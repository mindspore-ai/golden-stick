# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""RTNLayerPolicy."""

import abc

from mindspore.nn import Cell
from mindspore import dtype as msdtype
from mindspore_gs.quantization.layer_policy import LayerPolicy, PerChannelArgs
from mindspore_gs.ptq.fake_quantizer import FakeQuantizer
from mindspore_gs.ptq.ptq_config import InnerPTQConfig
from mindspore_gs.ptq import PTQMode
from mindspore_gs.common import logger
from ..fake_quantizer import MinMaxPerChannel, MinMaxPerLayer
from .quant_cells import LinearQuant, LinearDeploy, PagedAttentionDeploy, PagedAttentionQuant, \
    PagedAttentionDeployFusion


class RTNLayerPolicy(LayerPolicy, abc.ABC):
    """
    Derived class of LayerPolicy. Sim-QAT layer policy.
    Use linear perchannel fake quantizer as weight fake quantizer, linear perlayer fake quantizer as act fake quantizer.

    Supported Config:
        ``quant_delay`` ``quant_dtype`` ``per_channel`` ``symmetric`` ``narrow_range`` ``one_conv_fold``.
    """

    def __init__(self, weight_names: [], act_names: [], config: InnerPTQConfig = InnerPTQConfig()):
        super(RTNLayerPolicy, self).__init__()
        self._config: InnerPTQConfig = config
        if config.weight_dtype in (msdtype.int8, msdtype.uint8):
            self._num_bits = 8
        else:
            raise TypeError("Only support int8 weight quant now!")
        if config.act_per_channel:
            raise NotImplementedError("act quant only support perlayer now!")
        self._weight_names = weight_names
        self._act_names = act_names

    def get_weight_quantizer(self, weight_name="", perchannel_args: PerChannelArgs = PerChannelArgs(),
                             **kwargs) -> FakeQuantizer:
        if self._config.weight_dtype == msdtype.float_:
            return None
        strategy = kwargs.get('strategy', None)
        if self._config.weight_per_channel:
            channel_axis = perchannel_args.channel_axis
            num_channels = perchannel_args.num_channels
            rank = perchannel_args.rank
            if channel_axis == -1:
                raise RuntimeError("Please provide channel axis of weight for per-channel weight quantize.")
            if num_channels == -1:
                raise RuntimeError("Please provide channel number of weight for per-channel weight quantize.")
            if rank == -1:
                raise RuntimeError("Please provide rank of weight for per-channel weight quantize.")
            weight_quantizer = MinMaxPerChannel(symmetric=self._config.weight_symmetric, data_rank=rank,
                                                quant_dtype=self._config.weight_dtype,
                                                narrow_range=self._config.weight_narrow_range,
                                                axis=channel_axis, output_channel=num_channels, strategy=strategy)
        else:
            weight_quantizer = MinMaxPerLayer(symmetric=self._config.weight_symmetric,
                                              quant_dtype=self._config.weight_dtype,
                                              narrow_range=self._config.weight_narrow_range, strategy=strategy)
        weight_quantizer.set_attr("position", "weight")
        weight_only = self._config.weight_dtype == msdtype.int8 and self._config.act_dtype == msdtype.float_ and \
                      self._config.kvcache_dtype == msdtype.float_
        weight_quantizer.set_attr("weight_only_quant", weight_only)
        return weight_quantizer

    def _get_input_quantizer(self, input_index=-1, perchannel_args: PerChannelArgs = PerChannelArgs(),
                             **kwargs) -> FakeQuantizer:
        if self._config.act_dtype == msdtype.float_:
            return None
        quantizer = MinMaxPerLayer(symmetric=self._config.act_symmetric, quant_dtype=self._config.act_dtype,
                                   narrow_range=self._config.act_narrow_range, strategy=kwargs.get('strategy', None))
        quantizer.set_attr("position", "input")
        return quantizer

    def _get_output_quantizer(self, perchannel_args: PerChannelArgs = PerChannelArgs(), **kwargs) -> FakeQuantizer:
        if self._config.act_dtype == msdtype.float_:
            return None
        quantizer = MinMaxPerLayer(symmetric=self._config.act_symmetric, quant_dtype=self._config.act_dtype,
                                   narrow_range=self._config.act_narrow_range, strategy=kwargs.get('strategy', None))

        quantizer.set_attr("position", "output")
        return quantizer

    def get_config(self) -> InnerPTQConfig:
        return self._config

    @abc.abstractmethod
    def wrap_cell(self, handler: Cell) -> Cell:
        raise NotImplementedError


class LinearLayerPolicy(RTNLayerPolicy):
    """
    Derived class of SimulatedLayerPolicy. LayerPolicy used for nn.Dense.
    """
    def __init__(self, weight_names: [], act_names: [], config: InnerPTQConfig = InnerPTQConfig()):
        super().__init__(weight_names, act_names, config)
        self.set_input_number(1)
        if config.act_dtype == msdtype.float_:
            self.set_input_not_insert_fq()
            self.set_output_not_insert_fq()
        self.is_deploy = config.mode == PTQMode.DEPLOY

    def wrap_cell(self, handler) -> Cell:
        if self.is_deploy:
            return LinearDeploy(handler, self)
        return LinearQuant(handler, self)


class PagedAttentionMgrPolicy(RTNLayerPolicy):
    """
    Derived class of SimulatedLayerPolicy. LayerPolicy used for nn.Dense.
    """
    def __init__(self, weight_names: [], act_names: [], config: InnerPTQConfig = InnerPTQConfig()):
        super().__init__(weight_names, act_names, config)
        self.set_input_number(3)
        self.is_deploy = config.mode == PTQMode.DEPLOY
        self.enable_deploy_fusion = config.enable_deploy_fusion
        logger.info(f"PagedAttentionMgr Quant conifg: {config}.")

    def get_weight_quantizer(self, weight_name="", perchannel_args: PerChannelArgs = PerChannelArgs(),
                             **kwargs) -> FakeQuantizer:
        return None

    def _get_input_quantizer(self, input_index=-1, perchannel_args: PerChannelArgs = PerChannelArgs(),
                             **kwargs) -> FakeQuantizer:
        if self._config.kvcache_dtype == msdtype.float_:
            return None
        strategy = kwargs.get('strategy', None)
        channel_axis = perchannel_args.channel_axis
        num_channels = perchannel_args.num_channels
        rank = perchannel_args.rank
        if channel_axis == -1 and num_channels == -1 and rank == -1:
            return None
        if channel_axis == -1:
            raise RuntimeError("Please provide channel axis of input for per-channel input quantize.")
        if num_channels == -1:
            raise RuntimeError("Please provide channel number of input for per-channel input quantize.")
        if rank == -1:
            raise RuntimeError("Please provide rank of kvcache for per-channel weight quantize.")
        quantizer = MinMaxPerChannel(symmetric=self._config.weight_symmetric, data_rank=rank,
                                     quant_dtype=self._config.weight_dtype,
                                     narrow_range=self._config.weight_narrow_range, axis=channel_axis,
                                     output_channel=num_channels, strategy=strategy)
        quantizer.set_attr("position", "input")
        return quantizer

    def _get_output_quantizer(self, perchannel_args: PerChannelArgs = PerChannelArgs(), **kwargs) -> FakeQuantizer:
        if self._config.kvcache_dtype == msdtype.float_:
            return None
        strategy = kwargs.get('strategy', None)
        channel_axis = perchannel_args.channel_axis
        num_channels = perchannel_args.num_channels
        rank = perchannel_args.rank
        if channel_axis == -1 and num_channels == -1 and rank == -1:
            return None
        if channel_axis == -1:
            raise RuntimeError("Please provide channel axis of output for per-channel output quantize.")
        if num_channels == -1:
            raise RuntimeError("Please provide channel number of output for per-channel output quantize.")
        if rank == -1:
            raise RuntimeError("Please provide rank of kvcache for per-channel weight quantize.")
        quantizer = MinMaxPerChannel(symmetric=self._config.weight_symmetric, data_rank=rank,
                                     quant_dtype=self._config.weight_dtype,
                                     narrow_range=self._config.weight_narrow_range, axis=channel_axis,
                                     output_channel=num_channels, strategy=strategy)
        quantizer.set_attr("position", "output")
        return quantizer

    def wrap_cell(self, handler) -> Cell:
        if self.is_deploy:
            if self.enable_deploy_fusion:
                return PagedAttentionDeployFusion(handler, self)
            return PagedAttentionDeploy(handler, self)
        return PagedAttentionQuant(handler, self)
