# Copyright 2025 Huawei Technologies Co., Ltd
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
"""telechat2 quant model"""

from mindspore.nn.cell import Cell
from mindspore_gs.ptq.models.mindformers_models.mf_model import MFModel, MFModelEnableSafeTensors
from mindspore_gs.ptq.utils import QuantType


@MFModel.reg_model('telechat2')
class Telechat2(MFModelEnableSafeTensors):
    """Telechat2"""

    def _get_quant_type(self, network):
        """_get_quant_type"""
        if not isinstance(network, Cell):
            raise TypeError(f"Input network should be a Cell, but got: {type(Cell)}.")
        results = {}
        def process(root: Cell, name_prefix):
            """Iterate the whole network and call callback function `process_cell`."""
            if root is None:
                return
            for name, cell in root.name_cells().items():
                full_cell_name = f"{name_prefix}.{name}"
                if not hasattr(cell, "quant_type_dict"):
                    process(cell, full_cell_name)
                    continue
                info = cell.quant_type_dict()
                results.update(info)
        process(network, 'network')
        return results

    def get_description_file(self, network):
        """
        Obtain the description of quantization type for each parameter in each layer of the network.
        Such as W8A8 or W4A8_DYNAMIC
        """
        results = self._get_quant_type(network)
        desc_info = {}
        param_dict = self.parameters_dict()
        for key, _ in param_dict.items():
            if key in results.keys():
                desc_info[key] = results[key]
            else:
                desc_info[key] = QuantType.FLOAT.value
        return desc_info
