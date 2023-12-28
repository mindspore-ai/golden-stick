# Copyright 2022 Huawei Technologies Co., Ltd
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
"""utils for nn cell."""
from mindspore.nn import Cell
from mindspore_gs.quantization.simulated_quantization.simulated_quantization_layer_policy import SimulatedLayerPolicy
from mindspore_gs.quantization.simulated_quantization.simulated_quantization_config import SimulatedQuantizationConfig


class TestLayerPolicy(SimulatedLayerPolicy):
    """
    Mock LayerPolicy for quant cell test.
    """
    def __init__(self, input_number, weight_per_channel=False, act_per_channel=False):
        config: SimulatedQuantizationConfig = SimulatedQuantizationConfig()
        config.weight_per_channel = weight_per_channel
        config.act_per_channel = act_per_channel
        super(TestLayerPolicy, self).__init__([], [], config)
        self.set_input_number(input_number)

    def get_input_quantizer(self):
        return None

    def get_output_quantizer(self):
        return None

    def wrap_cell(self, handler: Cell) -> Cell:
        raise RuntimeError("Mock LayerPolicy, should not call wrap_cell method.")
