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
"""Load LinearSpec configurations from YAML file"""
import os
import yaml

from mindspore import dtype as msdtype

from tests.st.st0.ptq.networks.custom.mixed_precision_network import LinearSpec

def _load_linear_specs_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _resolve_template(value, context):
    """Resolve template expressions like ${hidden_size} in configuration"""
    if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
        expr = value[2:-1].strip()
        try:
            # pylint: disable=eval-used
            return eval(expr, {}, context)
        except Exception as e:
            raise ValueError(f"Failed to evaluate template '{value}': {e}") from e
    return value


def _resolve_params(params, context):
    """Resolve all template expressions in params dictionary"""
    resolved = {}
    for key, value in params.items():
        if isinstance(value, dict):
            resolved[key] = _resolve_params(value, context)
        elif isinstance(value, list):
            resolved[key] = [_resolve_template(v, context) if isinstance(v, str) else v for v in value]
        else:
            resolved[key] = _resolve_template(value, context)
    return resolved


def load_linear_specs_from_config(hidden_size=512, num_experts=2, config_path=None):
    """Load and create LinearSpec instances from YAML configuration file"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'linear_specs_config.yaml')

    config = _load_linear_specs_config(config_path)
    linear_specs = []

    dtype_map = {
        'float16': msdtype.float16,
        'bfloat16': msdtype.bfloat16,
        'float32': msdtype.float32,
    }

    context = {
        'hidden_size': hidden_size,
        'num_experts': num_experts,
    }

    for layer_type_config in config['layer_types']:
        layer_name = layer_type_config['name']
        strategies_per_dtype = layer_type_config['strategies_per_dtype']
        params_template = layer_type_config.get('params', {})

        for data_type_config in config['data_types']:
            compute_dtype_str = data_type_config['compute_dtype']
            param_dtype_str = data_type_config['param_dtype_str']
            compute_dtype = dtype_map[compute_dtype_str]

            params = _resolve_params(params_template, context)

            for _ in range(strategies_per_dtype):
                linear_specs.append(LinearSpec(
                    linear_type=layer_name,
                    compute_dtype=compute_dtype,
                    param_dtype_str=param_dtype_str,
                    **params
                ))

    print(f"Created {len(linear_specs)} LinearSpec instances")
    return linear_specs
