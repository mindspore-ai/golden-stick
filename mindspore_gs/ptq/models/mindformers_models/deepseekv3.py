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


from mindspore_gs.ptq.models.mindformers_models.mf_model import MFModel


@MFModel.reg_model('deepseek_v3')
class DeepSeekV3(MFModel):
    """DeepSeekV3"""

    def _split_route_moe_weight(self, param_dict) -> dict:
        return param_dict

    def _process_params_dict_before_save(self, param_dict) -> dict:
        new_param_dict = {}
        def replacer(string, src, dst):
            if src in string:
                string = string.replace(src, dst)
            return string

        for key, param in param_dict.items():
            if "key_cache" in key or "value_cache" in key or "float_weight" in key:
                continue
            new_key = key
            new_key = replacer(new_key, "._layer.matmul.", ".")
            new_key = replacer(new_key, "._layer.", ".")
            new_key = replacer(new_key, ".matmul.", ".")
            new_key = replacer(new_key, ".quant_op.", ".")
            new_key = replacer(new_key, ".input_zp", ".input_offset")
            new_key = replacer(new_key, ".weight_zp", ".weight_offset")
            new_key = replacer(new_key, ".dequant_scale", ".deq_scale")
            new_param_dict[new_key] = param
        new_param_dict = self._split_route_moe_weight(new_param_dict)
        # merge weights by TP
        return new_param_dict

    def get_description_file(self, network):
        raise NotImplementedError
