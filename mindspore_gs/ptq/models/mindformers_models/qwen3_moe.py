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

    def _split_route_moe_weight(self, param_dict) -> tuple[dict, dict]:
        """_split_route_moe_weight"""
        new_param_dict = {}
        experts_dict = {k: v for k, v in param_dict.items()
                        if ".mlp.experts." in k}
        other_dict = dict(param_dict.items() - experts_dict.items())
        new_param_dict.update(other_dict)

        is_fc1_quant = any([".linear_fc1.weight_scale" in k for k in experts_dict.keys()])
        is_fc2_quant = any([".linear_fc2.weight_scale" in k for k in experts_dict.keys()])

        trace = {}

        if is_fc1_quant:
            experts_fc1_dict = {k: v for k, v in experts_dict.items()
                                if ".mlp.experts.linear_fc1" in k}
            experts_fc1_dict, cur_trace = self._split_experts(experts_fc1_dict,
                                                              True,
                                                              "linear_fc1")
            trace.update(cur_trace)

        else:
            experts_fc1_dict = {k: v for k, v in experts_dict.items()
                                if ".mlp.experts.weight1" in k}
            experts_fc1_dict, cur_trace = self._split_experts(experts_fc1_dict,
                                                              False,
                                                              "linear_fc1")
            trace.update(cur_trace)

        if is_fc2_quant:
            experts_fc2_dict = {k: v for k, v in experts_dict.items()
                                if ".mlp.experts.linear_fc2" in k}
            experts_fc2_dict, cur_trace = self._split_experts(experts_fc2_dict,
                                                              True,
                                                              "linear_fc2")
            trace.update(cur_trace)
        else:
            experts_fc2_dict = {k: v for k, v in experts_dict.items()
                                if ".mlp.experts.weight2" in k}
            experts_fc2_dict, cur_trace = self._split_experts(experts_fc2_dict,
                                                              False,
                                                              "linear_fc2")
            trace.update(cur_trace)
        new_param_dict.update(experts_fc1_dict)
        new_param_dict.update(experts_fc2_dict)
        return new_param_dict, trace

    def _split_experts(self, param_dict, is_quant, layer_name):
        """_split_experts"""
        new_param_dict = {}
        trace = {}
        for key, value in param_dict.items():
            num_experts = value.shape[0]
            for i in range(num_experts):
                key_split = key.split('.')
                if is_quant:
                    new_name = f"{'.'.join(key_split[:6])}.{i}.{'.'.join(key_split[6:])}"
                else:
                    new_name = f"{'.'.join(key_split[:6])}.{i}.{layer_name}.weight"
                if len(value.shape) == 3:
                    new_value = value[i, :, :]
                elif len(value.shape) == 2:
                    new_value = value[i, :]
                else:
                    raise ValueError(f"The shape {value.shape} of {key} is not suppose.")
                if new_name.endswith('.weight'):
                    new_value = msops.transpose(new_value, (1, 0))
                new_param_dict[new_name] = Parameter(new_value, new_name)
                trace[new_name] = key
        return new_param_dict, trace

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
