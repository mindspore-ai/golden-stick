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


from mindformers import Linear
from mindformers.parallel_core.inference.tensor_parallel.layers import RowParallelLinear, ColumnParallelLinear, QKVParallelLinear
from mindspore_gs.ptq.models.mindformers_models.mf_model import MFModel
from mindspore_gs.ptq.ptq.wrappers.mindformers.fq_linear_all_quant import FakeQuantLinearCell
from mindspore_gs.ptq.processor import network_replace


@MFModel.reg_model('qwen3')
class QWen3(MFModel):
    """QWen3"""
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
        # split moe linear weights by EP
        # merge weights by TP
        return new_param_dict

    def fake_quant(self, ptq_config, layers_policy, quant_safetensors_path: str = ""):
        # FIXME hangangqiang2@huawei.com
        # fake_quant should create with ptq_config
        src_layers = (Linear, RowParallelLinear, ColumnParallelLinear, QKVParallelLinear)
        network_replace(self.network, src_layers, FakeQuantLinearCell, FakeQuantLinearCell, ['fc2', 'output_layer'])
        self.network.update_parameters_name()
        self._load_tp_splited_safetensors(quant_safetensors_path)
