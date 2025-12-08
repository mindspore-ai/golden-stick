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
"""Create PTQ layer policies from LinearSpec instances"""
from collections import OrderedDict
from typing import Any

from mindspore import dtype as msdtype

from mindspore_gs.ptq import PTQConfig, PTQMode
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq.ptq_config import (
    QuantGranularity, OutliersSuppressionType, PrecisionRecovery, GPTQQuantConfig
)


def _create_quantization_strategies():
    """Create standard and grouped quantization strategy configs"""
    a8w8_config = PTQConfig(
        mode=PTQMode.QUANTIZE,
        backend=BackendTarget.ASCEND,
        weight_quant_dtype=msdtype.int8,
        act_quant_dtype=msdtype.int8,
        act_quant_granularity=QuantGranularity.PER_TOKEN,
        weight_quant_granularity=QuantGranularity.PER_CHANNEL,
    )

    smoothquant_config = PTQConfig(
        mode=PTQMode.QUANTIZE,
        backend=BackendTarget.ASCEND,
        weight_quant_dtype=msdtype.int8,
        act_quant_dtype=msdtype.int8,
        act_quant_granularity=QuantGranularity.PER_TENSOR,
        weight_quant_granularity=QuantGranularity.PER_CHANNEL,
        outliers_suppression=OutliersSuppressionType.SMOOTH,
    )

    gptq_config_128 = GPTQQuantConfig(
        block_size=128, desc_act=True, static_groups=True, damp_percent=0.1)
    gptq_128_config = PTQConfig(
        mode=PTQMode.QUANTIZE,
        backend=BackendTarget.ASCEND,
        weight_quant_dtype=msdtype.qint4x2,
        act_quant_dtype=msdtype.int8,
        act_quant_granularity=QuantGranularity.PER_TOKEN,
        weight_quant_granularity=QuantGranularity.PER_GROUP,
        group_size=128,
        precision_recovery=PrecisionRecovery.GPTQ,
        algo_args=gptq_config_128,
    )

    a8w8_dynamic_grouped_config = PTQConfig(
        mode=PTQMode.QUANTIZE,
        backend=BackendTarget.ASCEND,
        weight_quant_dtype=msdtype.int8,
        act_quant_dtype=msdtype.int8,
        act_quant_granularity=QuantGranularity.PER_TOKEN,
    )

    standard_strategies = [a8w8_config, smoothquant_config]
    grouped_strategies = [a8w8_dynamic_grouped_config, gptq_128_config]

    return standard_strategies, grouped_strategies


def _is_grouped_linear(linear_type):
    return 'GroupedLinear' in linear_type


def create_layer_policies(linear_specs):
    """Create PTQ layer policies from LinearSpec instances"""
    layer_policies = OrderedDict[Any, Any]()
    standard_strategies, grouped_strategies = _create_quantization_strategies()

    strategy_counters = {}

    for layer_idx, spec in enumerate(linear_specs):
        is_grouped = _is_grouped_linear(spec.linear_type)
        strategies = grouped_strategies if is_grouped else standard_strategies

        key = (spec.linear_type, spec.param_dtype_str)
        if key not in strategy_counters:
            strategy_counters[key] = 0

        strategy_idx = strategy_counters[key] % len(strategies)

        if spec.linear_type == 'QKVParallelLinear':
            layer_policies[rf'.*layers\.{layer_idx}\.qkv(\..*)?$'] = strategies[strategy_idx]
            no_quant_config = PTQConfig(
                mode=PTQMode.QUANTIZE,
                backend=BackendTarget.ASCEND,
                weight_quant_dtype=None,
                act_quant_dtype=None,
            )
            layer_policies[rf'.*layers\.{layer_idx}\.proj(\..*)?$'] = no_quant_config
        elif spec.linear_type == 'MergedColumnParallelLinear':
            layer_policies[rf'.*layers\.{layer_idx}\.merged(\..*)?$'] = strategies[strategy_idx]
            no_quant_config = PTQConfig(
                mode=PTQMode.QUANTIZE,
                backend=BackendTarget.ASCEND,
                weight_quant_dtype=None,
                act_quant_dtype=None,
            )
            layer_policies[rf'.*layers\.{layer_idx}\.proj(\..*)?$'] = no_quant_config
        else:
            layer_policies[rf'.*layers\.{layer_idx}(?!\.proj)(\..*)?$'] = strategies[strategy_idx]

        strategy_counters[key] += 1

    proj_layers_count = sum(1 for spec in linear_specs
                            if spec.linear_type in ['QKVParallelLinear', 'MergedColumnParallelLinear'])
    expected_policies = len(linear_specs) + proj_layers_count
    expected_msg = (f"Expected {expected_policies} layer_policies "
                    f"({len(linear_specs)} main + {proj_layers_count} proj exclusions), "
                    f"but got {len(layer_policies)}")
    assert len(layer_policies) == expected_policies, expected_msg

    return layer_policies
