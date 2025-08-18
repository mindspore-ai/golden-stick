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
"""qwen3 quant model"""

from mindspore.nn.cell import Cell
from mindspore_gs.ptq.models.mindformers_models.mf_model import MFModel
from mindspore_gs.ptq.utils import QuantType


@MFModel.reg_model('qwen3')
class QWen3(MFModel):
    """QWen3"""

    def _split_route_moe_weight(self, param_dict) -> dict:
        return param_dict

    def _convert_name(self, key):
        """_convert_name"""
        def replacer(string, src, dst):
            if src in string:
                string = string.replace(src, dst)
            return string
        new_key = key
        new_key = replacer(new_key, "._layer.matmul.", ".")
        new_key = replacer(new_key, "._layer.", ".")
        new_key = replacer(new_key, ".matmul.", ".")
        new_key = replacer(new_key, ".quant_op.", ".")
        new_key = replacer(new_key, ".input_zp", ".input_offset")
        new_key = replacer(new_key, ".weight_zp", ".weight_offset")
        new_key = replacer(new_key, ".dequant_scale", ".deq_scale")
        return new_key

    def _process_params_dict_before_save(self, param_dict) -> dict:
        """_process_params_dict_before_save"""
        new_param_dict = {}
        for key, param in param_dict.items():
            if "key_cache" in key or "value_cache" in key or "float_weight" in key:
                continue
            new_key = self._convert_name(key)
            new_param_dict[new_key] = param
        new_param_dict = self._split_route_moe_weight(new_param_dict)
        # merge weights by TP
        return new_param_dict

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
                for key, value in info.items():
                    new_key = self._convert_name(key)
                    results[new_key] = value
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
