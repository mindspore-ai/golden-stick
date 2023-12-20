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
"""Simulated Quantization Convert Utils."""

from mindspore.common.dtype import QuantDtype
from mindspore_gs.quantization.quant_cell import QuantCell


class ConvertToQuantInferNetwork:
    """
    Convert quantization aware network to infer network.

    Args:
        network (Cell): SimulatedQuantizationAwareTraining apply network.

    Returns:
        Cell, Infer network.
    """

    def __init__(self, network):
        self.quant_dtype = QuantDtype.INT8
        self.network = network

    def run(self):
        """Start to convert."""
        self.network.update_cell_prefix()
        return self._convert_quant2deploy(self.network)

    def _convert_quant2deploy(self, network):
        """Convert network's all quant subcell to deploy subcell."""
        cells = network.name_cells()
        for name in cells:
            subcell = cells[name]
            if subcell == network:
                continue
            if isinstance(subcell, QuantCell):
                subcell.convert()
            else:
                self._convert_quant2deploy(subcell)
        return network
