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

from mindspore import ops as msops
from mindspore import Parameter
from mindspore_gs.ptq.models.mindformers_models.mf_model import MFModel
from mindspore_gs.ptq.utils import QuantType
from .qwen3 import QWen3


@MFModel.reg_model('qwen3_moe')
class QWen3MoE(QWen3):
    """QWen3"""

    def _split_route_moe_weight(self, param_dict) -> dict:
        """_split_route_moe_weight"""
        new_param_dict = {}
        for key, value in param_dict.items():
            if ".mlp.experts." in key:
                if "weight1" in key or 'weight2' in key:
                    continue
                num_experts = value.shape[0]
                for i in range(num_experts):
                    key_split = key.split('.')
                    new_name = f"{'.'.join(key_split[:6])}.{i}.{'.'.join(key_split[6:])}"
                    if len(value.shape) == 3:
                        new_value = value[i, :, :]
                    else:
                        new_value = value[i, :]
                    if key.endswith('.weight'):
                        new_value = msops.transpose(new_value, (1, 0))
                    new_param_dict[new_name] = Parameter(new_value)
            else:
                new_param_dict[key] = value
        return new_param_dict

    def _concat_route_moe_weight(self, param_dict) -> dict:
        """_concat_route_moe_weight"""
        new_param_dict = {}
        for key, value in param_dict.items():
            if ".mlp.experts." in key:
                key_split = key.split('.')
                prefix_str = '.'.join(key_split[:6])
                suffix_str = '.'.join(key_split[7:])
                new_name = f"{prefix_str}.{suffix_str}"
                if new_name in new_param_dict.keys():
                    continue
                experts_dict = {k: v for k, v in param_dict.items()
                                if k.startswith(prefix_str) and k.endswith(suffix_str)}
                num_experts = len(experts_dict.keys())
                value_list = []
                for i in range(num_experts):
                    key_ = f"{prefix_str}.{i}.{suffix_str}"
                    value_ = experts_dict[key_]
                    if key_.endswith('.weight'):
                        value_ = msops.transpose(value_, (1, 0))
                    value_ = value_.expand_dims(0)
                    value_list.append(value_)
                new_value = msops.cat(tuple(value_list), axis=0)
                new_param_dict[new_name] = Parameter(new_value)
            else:
                new_param_dict[key] = value
        return new_param_dict

    def _process_params_dict_before_load(self, param_dict) -> dict:
        """_process_params_dict_before_load"""
        param_dict = self._concat_route_moe_weight(param_dict)
        return param_dict

    def get_description_file(self, network):
        """
        Obtain the description of quantization type for each parameter in each layer of the network.
        Such as W8A8 or W4A8_DYNAMIC
        """
        results = self._get_quant_type(network)
        desc_info = {}
        param_dict = self.parameters_dict()
        for key, _ in param_dict.items():
            if "weight1" in key or 'weight2' in key:
                continue
            if ".mlp.experts." in key:
                key_split = key.split('.')
                experts_key = f"{'.'.join(key_split[:6])}.{'.'.join(key_split[7:])}"
                if experts_key in results.keys():
                    desc_info[key] = results[experts_key]
                    continue
            if key in results.keys():
                desc_info[key] = results[key]
            else:
                desc_info[key] = QuantType.FLOAT.value
        return desc_info
