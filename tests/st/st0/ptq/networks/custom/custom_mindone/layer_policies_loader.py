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
"""Create PTQ layer policies"""
from collections import OrderedDict
from typing import Any

from mindspore import dtype as msdtype

from mindspore_gs.ptq import PTQConfig
from mindspore_gs.ptq.ptq_config import (
    QuantGranularity, OutliersSuppressionType, PrecisionRecovery, GPTQQuantConfig
)


def get_layer_policy(quant_policy: str):
    """Get layer policy for mixed precision network"""
    if quant_policy == 'a16w4pc_awq':
        a16w4pc_awq = PTQConfig(
            weight_quant_dtype=msdtype.qint4x2,
            weight_quant_granularity=QuantGranularity.PER_CHANNEL,
            outliers_suppression=OutliersSuppressionType.AWQ,
            opname_blacklist=['pre_layer']
        )
        return a16w4pc_awq
    if quant_policy == 'a16w4pg_awq':
        a16w4pg_awq = PTQConfig(
            weight_quant_dtype=msdtype.qint4x2,
            weight_quant_granularity=QuantGranularity.PER_GROUP,
            group_size=128,
            outliers_suppression=OutliersSuppressionType.AWQ,
            opname_blacklist=['pre_layer']
        )
        return a16w4pg_awq
    if quant_policy == 'a16w4pc_gptq':
        a16w4pc_gptq = PTQConfig(
            weight_quant_dtype=msdtype.qint4x2,
            weight_quant_granularity=QuantGranularity.PER_CHANNEL,
            precision_recovery=PrecisionRecovery.GPTQ,
            algo_args=GPTQQuantConfig(),
            opname_blacklist=['pre_layer']
        )
        return a16w4pc_gptq
    if quant_policy == 'a16w4pg_gptq':
        a16w4pg_gptq = PTQConfig(
            weight_quant_dtype=msdtype.qint4x2,
            weight_quant_granularity=QuantGranularity.PER_GROUP,
            group_size=128,
            precision_recovery=PrecisionRecovery.GPTQ,
            algo_args=GPTQQuantConfig(),
            opname_blacklist=['pre_layer']
        )
        return a16w4pg_gptq
    if quant_policy == 'a16w8pc':
        a16w8pc = PTQConfig(
            weight_quant_dtype=msdtype.int8,
            weight_quant_granularity=QuantGranularity.PER_CHANNEL,
            opname_blacklist=['pre_layer']
        )
        return a16w8pc
    if quant_policy == 'a8ptknw8pc':
        a8ptknw8pc = PTQConfig(
            weight_quant_dtype=msdtype.int8,
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_TOKEN,
            weight_quant_granularity=QuantGranularity.PER_CHANNEL,
            opname_blacklist=['pre_layer']
        )
        return a8ptknw8pc
    if quant_policy == 'a8ptnsw8pc_smoothquant':
        a8ptnsw8pc_smoothquant = PTQConfig(
            weight_quant_dtype=msdtype.int8,
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_TENSOR,
            weight_quant_granularity=QuantGranularity.PER_CHANNEL,
            outliers_suppression=OutliersSuppressionType.SMOOTH,
            opname_blacklist=['pre_layer']
        )
        return a8ptnsw8pc_smoothquant
    if quant_policy == 'a8ptknw8pc_smoothquant':
        a8ptknw8pc_smoothquant = PTQConfig(
            weight_quant_dtype=msdtype.int8,
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_TOKEN,
            weight_quant_granularity=QuantGranularity.PER_CHANNEL,
            outliers_suppression=OutliersSuppressionType.SMOOTH,
            opname_blacklist=['pre_layer']
        )
        return a8ptknw8pc_smoothquant
    if quant_policy == 'a8ptnsw8pc_osl':
        a8ptnsw8pc_osl = PTQConfig(
            weight_quant_dtype=msdtype.int8,
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_TENSOR,
            weight_quant_granularity=QuantGranularity.PER_CHANNEL,
            outliers_suppression=OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE,
            opname_blacklist=['pre_layer']
        )
        return a8ptnsw8pc_osl
    if quant_policy == 'a8ptknw8pc_osl':
        a8ptknw8pc_osl = PTQConfig(
            weight_quant_dtype=msdtype.int8,
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_TOKEN,
            weight_quant_granularity=QuantGranularity.PER_CHANNEL,
            outliers_suppression=OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE,
            opname_blacklist=['pre_layer']
        )
        return a8ptknw8pc_osl
    if quant_policy == 'a8ptnsw8pc':
        a8ptnsw8pc = PTQConfig(
            weight_quant_dtype=msdtype.int8,
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_TOKEN,
            weight_quant_granularity=QuantGranularity.PER_CHANNEL,
            opname_blacklist=['pre_layer']
        )
        return a8ptnsw8pc
    raise ValueError(f"Unsupported quant policy: {quant_policy}")


def create_layer_policies_for_mindone(linear_specs):
    """Create PTQ layer policies for MindOne"""
    layer_policies = OrderedDict[Any, Any]()

    for layer_idx, linear_spec in enumerate(linear_specs):
        layer_policies[rf'.*layer\.linear_modules\.{layer_idx}\.linear.*'] = \
            get_layer_policy(linear_spec.quant_policy)
    return layer_policies
