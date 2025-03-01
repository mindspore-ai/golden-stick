# Copyright 2024 Huawei Technologies Co., Ltd
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
"""test layer configs."""

from collections import OrderedDict

import pytest

import mindspore as ms
from mindspore import nn, dtype, GRAPH_MODE
from mindspore_gs.ptq.ptq import PTQ
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQConfig, PTQMode, OutliersSuppressionType, PrecisionRecovery, QuantGranularity

from mindformers.modules.layers import Linear


class Attention(nn.Cell):
    """Attention"""
    def __init__(self, k, n):
        super(Attention, self).__init__()
        self.q = Linear(k, n)
        self.k = Linear(k, n)
        self.v = Linear(k, n)

    def construct(self, x):
        q = self.q(x)
        k = self.q(x)
        v = self.q(x)
        return q + k + v


class FeedForward(nn.Cell):
    """FeedForward"""
    def __init__(self, k, n):
        super(FeedForward, self).__init__()
        self.w1 = Linear(k, n)
        self.w2 = Linear(k, n)
        self.w3 = Linear(k, n)

    def construct(self, x):
        w1 = self.w1(x)
        w2 = self.w2(x)
        w3 = self.w3(x)
        return w1 + w2 + w3


class DecoderLayer(nn.Cell):
    """DecoderLayer"""
    def __init__(self, is_moe=False):
        super(DecoderLayer, self).__init__()
        self.attention = Attention(128, 128)
        if is_moe:
            self.moe = FeedForward(128, 128)
        else:
            self.ffn = FeedForward(128, 128)

    def construct(self, x):
        attn = self.attention(x)
        return self.ffn(attn)


class Network(nn.Cell):
    """Network"""
    def __init__(self):
        super(Network, self).__init__()
        self.layers = nn.CellList()
        for _ in range(3):
            layer = DecoderLayer()
            self.layers.append(layer)
        for _ in range(3):
            layer = DecoderLayer(True)
            self.layers.append(layer)

    def construct(self, x):
        for i in range(6):
            x = self.layers[i](x)
        return x


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_type():
    """
    Feature: test_type.
    Description: test_type.
    Expectation: test_type.
    """
    cfg = PTQConfig()
    with pytest.raises(TypeError):
        PTQ(config=cfg, layer_policies=1)
    with pytest.raises(TypeError):
        PTQ(config=cfg, layer_policies={1, 2})
    with pytest.raises(TypeError):
        PTQ(config=cfg, layer_policies=OrderedDict({1: cfg, 3: cfg}))
    with pytest.raises(TypeError):
        PTQ(config=cfg, layer_policies=OrderedDict({'1': 1, '3': 3}))

    ptq = PTQ(cfg)
    with pytest.raises(TypeError):
        ptq.summary(1)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_empty_summary():
    """
    Feature: summary.
    Description: test_empty_summary.
    Expectation: test_empty_summary.
    """
    ms.set_context(mode=GRAPH_MODE)
    ms.set_device('CPU')
    net = Network()
    cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, weight_quant_dtype=dtype.int8,
                    act_quant_dtype=dtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH,
                    opname_blacklist=['lkv2kv', 'lm_head'], precision_recovery=PrecisionRecovery.NONE,
                    act_quant_granularity=QuantGranularity.PER_TENSOR,
                    weight_quant_granularity=QuantGranularity.PER_CHANNEL)
    ptq = PTQ(config=cfg)
    ptq.decoder_layer_types.append(DecoderLayer)
    ptq.apply(net)
    ptq.summary(net)
    assert len(ptq._config.layer_quant_info_collect) == 36


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_quant_summary():
    """
    Feature: summary.
    Description: test_quant_summary.
    Expectation: test_quant_summary.
    """
    ms.set_context(mode=GRAPH_MODE)
    ms.set_device('CPU')
    net = Network()
    cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, weight_quant_dtype=dtype.int8,
                    act_quant_dtype=dtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH,
                    opname_blacklist=['lkv2kv', 'lm_head'], precision_recovery=PrecisionRecovery.NONE,
                    act_quant_granularity=QuantGranularity.PER_TENSOR,
                    weight_quant_granularity=QuantGranularity.PER_CHANNEL)
    wk_config = PTQConfig(weight_quant_dtype=dtype.int8, act_quant_dtype=dtype.int8,
                          outliers_suppression=OutliersSuppressionType.NONE,
                          precision_recovery=PrecisionRecovery.NONE,
                          act_quant_granularity=QuantGranularity.PER_TENSOR,
                          weight_quant_granularity=QuantGranularity.PER_CHANNEL)
    ffn_config = PTQConfig(weight_quant_dtype=dtype.int8, act_quant_dtype=dtype.int8,
                           outliers_suppression=OutliersSuppressionType.NONE,
                           precision_recovery=PrecisionRecovery.NONE,
                           act_quant_granularity=QuantGranularity.PER_TOKEN,
                           weight_quant_granularity=QuantGranularity.PER_CHANNEL)
    awq_config = PTQConfig(weight_quant_dtype=dtype.qint4x2,
                           outliers_suppression=OutliersSuppressionType.AWQ,
                           precision_recovery=PrecisionRecovery.NONE,
                           kvcache_quant_granularity=QuantGranularity.PER_CHANNEL,
                           weight_quant_granularity=QuantGranularity.PER_GROUP, group_size=64)
    ptq = PTQ(config=cfg, layer_policies=OrderedDict({r'.*attention\.k.*': wk_config,
                                                      r'.*[1,3,5].*moe.*': ffn_config,
                                                      r'.*ffn.*': ffn_config,
                                                      r'.*[0,2,4].*moe.*': awq_config,
                                                     }))
    ptq.decoder_layer_types.append(DecoderLayer)
    ptq.apply(net)
    ptq.summary(net)
    assert len(ptq._config.layer_quant_info_collect) == 36
    assert (ptq._config.layer_quant_info_collect['network.layers.0.attention.q'] ==
            'SmoothQuant-W8-per_channel-A8-per_tensor')
    assert ptq._config.layer_quant_info_collect['network.layers.0.attention.k'] == 'W8-per_channel-A8-per_tensor'
    assert ptq._config.layer_quant_info_collect['network.layers.0.ffn.w1'] == 'W8-per_channel-A8-per_token'
    assert ptq._config.layer_quant_info_collect['network.layers.4.moe.w1'] == 'AWQ-W4-per_group'


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_quant_policy_conflict_summary():
    """
    Feature: summary.
    Description: test_quant_policy_conflict_summary.
    Expectation: test_quant_policy_conflict_summary.
    """
    ms.set_context(mode=GRAPH_MODE)
    ms.set_device('CPU')
    net = Network()
    cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, weight_quant_dtype=dtype.int8,
                    act_quant_dtype=dtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH,
                    opname_blacklist=['lkv2kv', 'lm_head'], precision_recovery=PrecisionRecovery.NONE,
                    act_quant_granularity=QuantGranularity.PER_TENSOR,
                    weight_quant_granularity=QuantGranularity.PER_CHANNEL)
    w2_config = PTQConfig(weight_quant_dtype=dtype.int8, act_quant_dtype=dtype.int8,
                          outliers_suppression=OutliersSuppressionType.NONE,
                          precision_recovery=PrecisionRecovery.NONE,
                          act_quant_granularity=QuantGranularity.PER_TENSOR,
                          weight_quant_granularity=QuantGranularity.PER_CHANNEL)
    ffn_config = PTQConfig(weight_quant_dtype=dtype.int8, act_quant_dtype=dtype.int8,
                           outliers_suppression=OutliersSuppressionType.NONE,
                           precision_recovery=PrecisionRecovery.NONE,
                           act_quant_granularity=QuantGranularity.PER_TOKEN,
                           weight_quant_granularity=QuantGranularity.PER_CHANNEL)
    awq_config = PTQConfig(weight_quant_dtype=dtype.qint4x2,
                           outliers_suppression=OutliersSuppressionType.AWQ,
                           precision_recovery=PrecisionRecovery.NONE,
                           weight_quant_granularity=QuantGranularity.PER_GROUP, group_size=64)
    ptq = PTQ(config=cfg, layer_policies=OrderedDict({r'.*attention\.w2.*': w2_config,
                                                      r'.*moe.*': ffn_config,
                                                      r'.*ffn.*': ffn_config,
                                                      r'.*4.*moe.*': awq_config,
                                                     }))
    ptq.decoder_layer_types.append(DecoderLayer)
    with pytest.raises(RuntimeError):
        ptq.apply(net)
        ptq.summary(net)
