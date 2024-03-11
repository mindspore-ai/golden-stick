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

from mindformers import Linear
from mindspore.nn import Cell
from mindspore.common.dtype import QuantDtype
from mindspore_gs.quantization.layer_policy import LayerPolicy, PerChannelArgs
from mindspore_gs.quantization.fake_quantizer import FakeQuantizer
from mindspore_gs.ptq.ptq_config import PTQConfig
from ..fake_quantizer import MinMaxPerChannel, MinMaxPerLayer
from ..quant_cells import LinearQuant


class RTNLayerPolicy(LayerPolicy, abc.ABC):
    """
    Derived class of LayerPolicy. Sim-QAT layer policy.
    Use linear perchannel fake quantizer as weight fake quantizer, linear perlayer fake quantizer as act fake quantizer.

    Supported Config:
        ``quant_delay`` ``quant_dtype`` ``per_channel`` ``symmetric`` ``narrow_range`` ``one_conv_fold``.
    """

    def __init__(self, weight_names: [], act_names: [], config: PTQConfig = PTQConfig()):
        super(RTNLayerPolicy, self).__init__()
        self._config: PTQConfig = config
        if config.weight_quant_dtype == QuantDtype.INT8:
            self._num_bits = 8
        else:
            raise TypeError("Only support int8 weight quant now!")
        if config.act_per_channel:
            raise NotImplementedError("act quant only support perlayer now!")
        self._weight_names = weight_names
        self._act_names = act_names

    def get_weight_quantizer(self, weight_name="", perchannel_args: PerChannelArgs = PerChannelArgs(),
                             **kwargs) -> FakeQuantizer:
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
                                                quant_dtype=self._config.weight_quant_dtype,
                                                narrow_range=self._config.weight_narrow_range,
                                                axis=channel_axis, output_channel=num_channels, strategy=strategy)
        else:
            weight_quantizer = MinMaxPerLayer(symmetric=self._config.weight_symmetric,
                                              quant_dtype=self._config.weight_quant_dtype,
                                              narrow_range=self._config.weight_narrow_range, strategy=strategy)
        weight_quantizer.set_attr("position", "weight")
        weight_quantizer.set_attr("weight_only_quant", self._config.weight_only)
        return weight_quantizer

    def _get_input_quantizer(self, input_index=-1, perchannel_args: PerChannelArgs = PerChannelArgs(),
                             **kwargs) -> FakeQuantizer:
        quantizer = MinMaxPerLayer(symmetric=self._config.act_symmetric, quant_dtype=self._config.act_quant_dtype,
                                   narrow_range=self._config.act_narrow_range, strategy=kwargs.get('strategy', None))
        quantizer.set_attr("position", "input")
        return quantizer

    def _get_output_quantizer(self, perchannel_args: PerChannelArgs = PerChannelArgs(), **kwargs) -> FakeQuantizer:
        quantizer = MinMaxPerLayer(symmetric=self._config.act_symmetric, quant_dtype=self._config.act_quant_dtype,
                                   narrow_range=self._config.act_narrow_range, strategy=kwargs.get('strategy', None))

        quantizer.set_attr("position", "output")
        return quantizer

    def get_config(self) -> PTQConfig:
        return self._config

    @abc.abstractmethod
    def wrap_cell(self, handler: Cell) -> Cell:
        raise NotImplementedError


class LinearLayerPolicy(RTNLayerPolicy):
    """
    Derived class of SimulatedLayerPolicy. LayerPolicy used for nn.Dense.
    """
    def __init__(self, weight_names: [], act_names: [], config: PTQConfig = PTQConfig()):
        super().__init__(weight_names, act_names, config)
        self.set_input_number(1)
        if config.weight_only:
            self.set_input_not_insert_fq()
            self.set_output_not_insert_fq()

    def wrap_cell(self, handler: Linear) -> Cell:
        return LinearQuant(handler, self)
