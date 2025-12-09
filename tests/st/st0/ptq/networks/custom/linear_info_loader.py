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
"""Load LinearDesc configurations from YAML file"""
import os
import yaml

from mindspore import dtype as msdtype


class LinearSpec:
    """Specification for linear layers"""

    def __init__(self, linear_type, compute_dtype, param_dtype_str,
                 input_size=None, output_size=None, hidden_size=None,
                 has_bias=True, bias=False, gather_output=False,
                 transpose_b=True, num_local_experts=None,
                 head_size=None, total_num_heads=None, total_num_kv_heads=None,
                 ffn_hidden_size=None, extra_params=None, quant_policy=None):
        self.linear_type = linear_type
        self.compute_dtype = compute_dtype
        self.param_dtype_str = param_dtype_str
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.ffn_hidden_size = ffn_hidden_size
        self.num_local_experts = num_local_experts
        self.has_bias = has_bias
        self.bias = bias
        self.gather_output = gather_output
        self.transpose_b = transpose_b
        self.quant_policy = quant_policy
        if extra_params:
            param_mapping = {
                'input_size': 'input_size',
                'output_size': 'output_size',
                'hidden_size': 'hidden_size',
                'num_local_experts': 'num_local_experts',
                'head_size': 'head_size',
                'total_num_heads': 'total_num_heads',
                'total_num_kv_heads': 'total_num_kv_heads',
                'ffn_hidden_size': 'ffn_hidden_size',
                'bias': 'bias',
                'gather_output': 'gather_output',
                'transpose_b': 'transpose_b',
            }
            for key, attr_name in param_mapping.items():
                if key in extra_params:
                    setattr(self, attr_name, extra_params[key])

    def name(self):
        """Return a string representation of the linear spec"""
        return f"{self.linear_type}-compute_dtype_{self.compute_dtype}-param_dtype_{self.param_dtype_str}"


def _load_linear_specs_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_linear_info_from_config(config_path=None):
    """Load and create LinearDesc instances from YAML configuration file"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'linear_specs_config.yaml')

    config = _load_linear_specs_config(config_path)
    model_config = config['model_config']
    default_precision = config['default_precision']
    layer_types = config['layer_types']

    dtype_map = {
        'float16': msdtype.float16,
        'bfloat16': msdtype.bfloat16,
        'float32': msdtype.float32
    }
    precision_thd = {}
    linear_specs = []
    for layer_type_config in layer_types:
        # precision threshold
        layer_precision_thd = layer_type_config.get('precision_threshold', default_precision)
        key = (layer_type_config['name'],
               dtype_map[layer_type_config['compute_dtype']],
               layer_type_config['quant_policy'])
        precision_thd[key] = layer_precision_thd

        # create linear spec
        linear_specs.append(
            LinearSpec(
                linear_type=layer_type_config['name'],
                quant_policy=layer_type_config['quant_policy'],
                compute_dtype=dtype_map[layer_type_config['compute_dtype']],
                param_dtype_str=layer_type_config['compute_dtype'],
                has_bias=layer_type_config.get('has_bias', False),
                **model_config))

    print(f"Created {len(linear_specs)} LinearSpec instances")
    return model_config, precision_thd, linear_specs
