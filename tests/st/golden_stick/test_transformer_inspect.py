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
"""test NetworkAnalyser."""


import pytest
from mindspore_gs.ptq.ptq.transformer_inspect import TransformerInspect, LayerType

from .llama2 import llama2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_parse_noqkvconcat_network():
    """
    Feature: Parsing a network without qkv_concat.
    Description: Tests parsing a network where qkv_concat is set to False using the TransformerInspect class.
                 Verifies the counts of various nodes such as attention norm nodes, q, k, v, o nodes, etc.
    Expectation: The counts of all specified nodes match the expected values, ensuring correct parsing of the network
                 structure.
    """
    transformer_graph = TransformerInspect()
    network = llama2(num_layers=2, qkv_concat=False)
    transformer_graph.parse(network)
    assert len(transformer_graph.attn_norm_nodes) == 2
    assert len(transformer_graph.q_nodes) == 2
    assert len(transformer_graph.k_nodes) == 2
    assert len(transformer_graph.v_nodes) == 2
    assert len(transformer_graph.o_nodes) == 2
    assert len(transformer_graph.qkv_nodes) == 0
    assert len(transformer_graph.ffn_norm_nodes) == 2
    assert len(transformer_graph.mlp_w1_nodes) == 2
    assert len(transformer_graph.mlp_w2_nodes) == 2
    assert len(transformer_graph.mlp_w3_nodes) == 2
    assert len(transformer_graph.mlp_gate_hidden_nodes) == 0

    assert len(transformer_graph.q2l_nodes) == 0
    assert len(transformer_graph.kv2l_nodes) == 0
    assert len(transformer_graph.l2q_nodes) == 0
    assert len(transformer_graph.l2kv_nodes) == 0
    assert len(transformer_graph.q2lkv2l_nodes) == 0
    assert len(transformer_graph.route_w1_nodes) == 0
    assert len(transformer_graph.route_w2_nodes) == 0
    assert len(transformer_graph.route_w3_nodes) == 0
    assert len(transformer_graph.route_gate_hidden_nodes) == 0
    assert len(transformer_graph.share_w1_nodes) == 0
    assert len(transformer_graph.share_w2_nodes) == 0
    assert len(transformer_graph.share_w3_nodes) == 0
    assert len(transformer_graph.share_gate_hidden_nodes) == 0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_parse_qkvconcat_network():
    """
    Feature: Parsing a network with qkv_concat.
    Description: Tests parsing a network where qkv_concat is set to True using the TransformerInspect class.
                 Verifies the counts of various nodes such as attention norm nodes, qkv nodes, o nodes, etc.
                 Also checks the functionality of the get_pre_layer method for specific nodes.
    Expectation: The counts of all specified nodes match the expected values, and the get_pre_layer method returns the
                 correct nodes.
    """
    transformer_graph = TransformerInspect()
    network = llama2(num_layers=2, qkv_concat=True)
    transformer_graph.parse(network)
    assert len(transformer_graph.attn_norm_nodes) == 2
    assert len(transformer_graph.q_nodes) == 0
    assert len(transformer_graph.k_nodes) == 0
    assert len(transformer_graph.v_nodes) == 0
    assert len(transformer_graph.o_nodes) == 2
    assert len(transformer_graph.qkv_nodes) == 2
    assert len(transformer_graph.ffn_norm_nodes) == 2
    assert len(transformer_graph.mlp_w1_nodes) == 0
    assert len(transformer_graph.mlp_w2_nodes) == 2
    assert len(transformer_graph.mlp_w3_nodes) == 0
    assert len(transformer_graph.mlp_gate_hidden_nodes) == 2

    assert len(transformer_graph.q2l_nodes) == 0
    assert len(transformer_graph.kv2l_nodes) == 0
    assert len(transformer_graph.l2q_nodes) == 0
    assert len(transformer_graph.l2kv_nodes) == 0
    assert len(transformer_graph.q2lkv2l_nodes) == 0
    assert len(transformer_graph.route_w1_nodes) == 0
    assert len(transformer_graph.route_w2_nodes) == 0
    assert len(transformer_graph.route_w3_nodes) == 0
    assert len(transformer_graph.route_gate_hidden_nodes) == 0
    assert len(transformer_graph.share_w1_nodes) == 0
    assert len(transformer_graph.share_w2_nodes) == 0
    assert len(transformer_graph.share_w3_nodes) == 0
    assert len(transformer_graph.share_gate_hidden_nodes) == 0

    node = transformer_graph.get_pre_layer('network.model.layers.0.attention.w_qkv')
    assert node.name == 'network.model.layers.0.attention_norm'
    assert node.type_ == LayerType.ATTN_NORM
    node = transformer_graph.get_pre_layer('network.model.layers.0.attention.wo')
    assert node.name == 'network.model.layers.0.attention.w_qkv'
    assert node.type_ == LayerType.QKV
    node = transformer_graph.get_pre_layer('network.model.layers.0.attention.w2')
    assert node is None
