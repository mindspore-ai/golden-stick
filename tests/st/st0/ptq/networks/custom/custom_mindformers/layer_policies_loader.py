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


def get_layer_policy(quant_policy: str):
    """Get layer policy for mixed precision network"""
    if quant_policy == 'a8ptknw8pc':
        a8ptknw8pc = PTQConfig(
            mode=PTQMode.QUANTIZE,
            backend=BackendTarget.ASCEND,
            weight_quant_dtype=msdtype.int8,
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_TOKEN,
            weight_quant_granularity=QuantGranularity.PER_CHANNEL,
        )
        return a8ptknw8pc
    if quant_policy == 'a8ptnsw8pc_smoothquant':
        a8ptnsw8pc_smoothquant = PTQConfig(
            mode=PTQMode.QUANTIZE,
            backend=BackendTarget.ASCEND,
            weight_quant_dtype=msdtype.int8,
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_TENSOR,
            weight_quant_granularity=QuantGranularity.PER_CHANNEL,
            outliers_suppression=OutliersSuppressionType.SMOOTH,
        )
        return a8ptnsw8pc_smoothquant
    if quant_policy == "a8ptknw4pg":
        gptq_config_128 = GPTQQuantConfig(
            block_size=128, desc_act=True, static_groups=True, damp_percent=0.1)
        a8ptknw4pg = PTQConfig(
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
        return a8ptknw4pg
    if quant_policy == "":
        a8ptknw8pc = PTQConfig(
            mode=PTQMode.QUANTIZE,
            backend=BackendTarget.ASCEND,
            weight_quant_dtype=msdtype.int8,
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_TOKEN,
        )
        return a8ptknw8pc
    raise ValueError(f"Unsupported quant policy: {quant_policy}")


def create_layer_policies_for_mindformers(linear_specs):
    """Create PTQ layer policies for MindFormers"""
    layer_policies = OrderedDict[Any, Any]()

    for layer_idx, linear_spec in enumerate(linear_specs):
        layer_policy = get_layer_policy(linear_spec.quant_policy)
        if linear_spec.linear_type == 'QKVParallelLinear':
            layer_policies[rf'.*layers\.{layer_idx}\.qkv(\..*)?$'] = layer_policy
            no_quant_config = PTQConfig(
                mode=PTQMode.QUANTIZE,
                backend=BackendTarget.ASCEND,
                weight_quant_dtype=None,
                act_quant_dtype=None,
            )
            layer_policies[rf'.*layers\.{layer_idx}\.proj(\..*)?$'] = no_quant_config
        elif linear_spec.linear_type == 'MergedColumnParallelLinear':
            layer_policies[rf'.*layers\.{layer_idx}\.merged(\..*)?$'] = layer_policy
            no_quant_config = PTQConfig(
                mode=PTQMode.QUANTIZE,
                backend=BackendTarget.ASCEND,
                weight_quant_dtype=None,
                act_quant_dtype=None,
            )
            layer_policies[rf'.*layers\.{layer_idx}\.proj(\..*)?$'] = no_quant_config
        else:
            layer_policies[rf'.*layers\.{layer_idx}(?!\.proj)(\..*)?$'] = layer_policy
    return layer_policies
