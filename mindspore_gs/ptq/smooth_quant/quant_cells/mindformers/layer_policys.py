# Copyright 2024 Huawei Technologies Co., Ltd
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
"""SQLayerPolicy for mindformers layers."""

from mindspore.nn import Cell
from mindspore import dtype as msdtype
from mindspore_gs.quantization.layer_policy import PerChannelArgs
from mindspore_gs.ptq.fake_quantizer import FakeQuantizer
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq import PTQMode
from mindspore_gs.ptq.smooth_quant.sq_layer_policy import SQLayerPolicy
from mindspore_gs.ptq.fake_quantizer import MinMaxPerChannel
from .quant_cells import SQLinearActObserver, SQLinearDeploy


class LinearLayerPolicy(SQLayerPolicy):
    """
    Derived class of SimulatedLayerPolicy. LayerPolicy used for nn.Dense.
    """
    def __init__(self, weight_names: [], act_names: [], config: InnerPTQConfig = InnerPTQConfig()):
        super().__init__(weight_names, act_names, config)
        self.set_input_number(1)
        self._is_deploy = config.mode == PTQMode.DEPLOY

    def create_observer_perchannel(self, perchannel_args: PerChannelArgs = PerChannelArgs(), **kwargs) -> FakeQuantizer:
        """create_observer_perchannel."""
        strategy = kwargs.get('strategy', None)
        channel_axis = perchannel_args.channel_axis
        num_channels = perchannel_args.num_channels
        rank = perchannel_args.rank
        if num_channels == -1:
            raise RuntimeError("Please provide channel number for observer.")
        perchannel_observer = MinMaxPerChannel(symmetric=True, narrow_range=False, axis=channel_axis, data_rank=rank,
                                               output_channel=num_channels, strategy=strategy)
        return perchannel_observer

    def wrap_cell(self, handler) -> Cell:
        if self._config.weight_quant_dtype != msdtype.int8 or self._config.act_quant_dtype != msdtype.int8:
            return None
        if self._is_deploy:
            return SQLinearDeploy(handler, self, self._config)
        return SQLinearActObserver(handler, self, self._config)
