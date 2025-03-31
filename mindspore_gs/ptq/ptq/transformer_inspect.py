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
"""Transformer graph."""


import enum
from dataclasses import dataclass
from typing import Tuple
from collections import OrderedDict
from mindspore.nn import Cell

from mindspore_gs.common.utils import value_check
from mindspore_gs.ptq.processor import Processor
from mindspore_gs.common.name_prefix_tree import NameTree


class PreLayerType(enum.Enum):
    """
    Pre layer type Enum.

    - ``UNKNOWN`` : pre layer type is unknown.
    - ``NORM_LAYER`` : pre layer type is norm layer.
    - ``LINEAR_LAYER`` : pre layer type is linear layer.
    - ``CONCAT_LINEAR_LAYER`` : pre layer type is qkv concat linear layer.
    """
    UNKNOWN = 0
    NORM_LAYER = 1
    LINEAR_LAYER = 2
    CONCAT_LINEAR_LAYER = 3


class LayerType(enum.Enum):
    """LayerType"""
    UNKNOWN = 0
    Q = 1
    K = 2
    V = 3
    O = 3
    QKV = 4
    Q2L = 5
    L2Q = 6
    KV2L = 7
    L2KV = 8
    Q2LKV2L = 9
    MLP_W1 = 10
    MLP_W2 = 11
    MLP_W3 = 12
    MLP_GATE_HIDDEN = 13
    SHARED_EXPERT_W1 = 14
    SHARED_EXPERT_W2 = 15
    SHARED_EXPERT_W3 = 16
    SHARED_EXPERT_GATE_HIDDEN = 17
    ROUTED_EXPERT_W1 = 18
    ROUTED_EXPERT_W2 = 19
    ROUTED_EXPERT_W3 = 20
    ROUTED_EXPERT_GATE_HIDDEN = 21
    ATTN_NORM = 22
    Q_NORM = 23
    KV_NORM = 24
    FFN_NORM = 25


@dataclass
class LayerInfo:
    """
    Dataclass for recording layer information.

    Args:
        name (str) - name of layer.
        layer (Cell) - layer.
        type_ (LayerType) - type of layer, ``NORM_LAYER``is norm layer,
            ``LINEAR_LAYER`` is linear,``CONCAT_LINEAR_LAYER``is qkv concat linear layer,
            ``UNKNOWN``is unknown type.

    Raises:
        TypeError: `name` is not str.
        TypeError: `layer` type is not Cell.
        TypeError: `type_` not in [LayerType.UNKNOWN, LayerType.NORM_LAYER, LayerType.LINEAR_LAYER,
            LayerType.CONCAT_LINEAR_LAYER].
    """
    name: str = ""
    layer: Cell = None
    type_: LayerType = LayerType.UNKNOWN

    def __post_init__(self):
        value_check('name', self.name, str)
        value_check('layer', self.layer, Cell)
        value_check('type', self.type_, LayerType)


class TransformerInspect:
    """TransformerInspect"""
    q_keys = ('.wq',)
    k_keys = ('.wk',)
    v_keys = ('.wv',)
    qkv_keys = ('.wqkv', '.qkv', '.w_qkv')
    o_keys = ('.wo',)

    q2l_keys = ('.q2l',)
    l2q_keys = ('.l2q',)
    kv2l_keys = ('.kv2l',)
    l2kv_keys = ('.l2kv',)
    q2lkv2l_keys = ('.q2lkv2l',)

    w1_keys = ('.w1',)
    w2_keys = ('.w2',)
    w3_keys = ('.w3',)
    gate_hidden_keys = ('.w_gate_hidden',)

    norm_keys = ('norm', 'ln')

    def __init__(self):
        self.name_tree = NameTree()
        # MHA/GQA
        self.q_nodes = []
        self.k_nodes = []
        self.v_nodes = []
        self.qkv_nodes = []

        self.o_nodes = []
        # MLA
        self.q2l_nodes = []
        self.l2q_nodes = []
        self.kv2l_nodes = []
        self.l2kv_nodes = []
        self.q2lkv2l_nodes = []
        # MLP
        self.mlp_w1_nodes = []
        self.mlp_w3_nodes = []
        self.mlp_w2_nodes = []
        self.mlp_gate_hidden_nodes = []
        # route moe
        self.route_w1_nodes = []
        self.route_w3_nodes = []
        self.route_w2_nodes = []
        self.route_gate_hidden_nodes = []
        # share moe
        self.share_w1_nodes = []
        self.share_w3_nodes = []
        self.share_w2_nodes = []
        self.share_gate_hidden_nodes = []
        # attn norm
        self.attn_norm_nodes = []
        # ffn norm
        self.ffn_norm_nodes = []

    def _find_node(self, optional_keys, required_keys=None, exclude_keys=None):
        """_find_node"""
        if required_keys is None:
            required_keys = []
        if exclude_keys is None:
            exclude_keys = []
        results = []
        for optional_key in optional_keys:
            keys = required_keys + [optional_key]
            results += self.name_tree.find_leafs_with_keywords(keys, exclude_keys)
        return results

    def _find_layers(self):
        """_find_layers"""
        self.q_nodes = self._find_node(self.q_keys)
        self.k_nodes = self._find_node(self.k_keys)
        self.v_nodes = self._find_node(self.v_keys)
        self.qkv_nodes = self._find_node(self.qkv_keys)
        self.o_nodes = self._find_node(self.o_keys)

        self.q2l_nodes = self._find_node(self.q2l_nodes)
        self.l2q_nodes = self._find_node(self.l2q_nodes)
        self.kv2l_nodes = self._find_node(self.kv2l_keys)
        self.l2kv_nodes = self._find_node(self.l2kv_keys)
        self.q2lkv2l_nodes = self._find_node(self.q2lkv2l_keys)

        self.route_w1_nodes = self._find_node(self.w1_keys, ['route'], ['activation'])
        self.route_w2_nodes = self._find_node(self.w2_keys, ['route'], ['activation'])
        self.route_w3_nodes = self._find_node(self.w3_keys, ['route'], ['activation'])
        self.route_gate_hidden_nodes = self._find_node(self.gate_hidden_keys, ['route'], ['activation'])

        self.share_w1_nodes = self._find_node(self.w1_keys, ['share'], ['activation'])
        self.share_w2_nodes = self._find_node(self.w2_keys, ['share'], ['activation'])
        self.share_w3_nodes = self._find_node(self.w3_keys, ['share'], ['activation'])
        self.share_gate_hidden_nodes = self._find_node(self.gate_hidden_keys, ['share'], ['activation'])

        self.mlp_w1_nodes = self._find_node(self.w1_keys, [], ['route', 'share', 'activation'])
        self.mlp_w2_nodes = self._find_node(self.w2_keys, [], ['route', 'share', 'activation'])
        self.mlp_w3_nodes = self._find_node(self.w3_keys, [], ['route', 'share', 'activation'])
        self.mlp_gate_hidden_nodes = self._find_node(self.gate_hidden_keys, [], ['route', 'share', 'activation'])

        self.attn_norm_nodes = self._find_node(self.norm_keys, ['att'])
        self.ffn_norm_nodes = self._find_node(self.norm_keys, [], ['att', 'norm_out'])

    def _parse_name_tree(self, network: Cell):
        """_parse_name_tree"""
        class NetworkWalker(Processor):
            def __init__(self, name_tree: NameTree):
                self.name_tree = name_tree
                self.infos: OrderedDict[str, Cell] = OrderedDict()

            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                if not isinstance(cell, Cell):
                    return cell, True
                self.name_tree.add_name(cell_name, cell)
                return cell, False

        walker = NetworkWalker(self.name_tree)
        walker.process(network)

    def parse(self, network: Cell):
        """parse"""
        self._parse_name_tree(network)
        self._find_layers()

    def _get_layer_type(self, layer_name: str):
        """_get_layer_type"""
        if any(layer_name == node.name for node in self.q_nodes):
            return LayerType.Q
        if any(layer_name == node.name for node in self.k_nodes):
            return LayerType.K
        if any(layer_name == node.name for node in self.v_nodes):
            return LayerType.V
        if any(layer_name == node.name for node in self.o_nodes):
            return LayerType.O
        if any(layer_name == node.name for node in self.qkv_nodes):
            return LayerType.QKV
        if any(layer_name == node.name for node in self.q2l_nodes):
            return LayerType.Q2L
        if any(layer_name == node.name for node in self.kv2l_nodes):
            return LayerType.KV2L
        if any(layer_name == node.name for node in self.l2q_nodes):
            return LayerType.L2Q
        if any(layer_name == node.name for node in self.l2kv_nodes):
            return LayerType.L2KV
        if any(layer_name == node.name for node in self.q2lkv2l_nodes):
            return LayerType.Q2LKV2L
        if any(layer_name == node.name for node in self.mlp_w1_nodes):
            return LayerType.MLP_W1
        if any(layer_name == node.name for node in self.mlp_w3_nodes):
            return LayerType.MLP_W3
        if any(layer_name == node.name for node in self.mlp_w2_nodes):
            return LayerType.MLP_W2
        if any(layer_name == node.name for node in self.mlp_gate_hidden_nodes):
            return LayerType.MLP_GATE_HIDDEN
        if any(layer_name == node.name for node in self.share_w1_nodes):
            return LayerType.SHARED_EXPERT_W1
        if any(layer_name == node.name for node in self.share_w2_nodes):
            return LayerType.SHARED_EXPERT_W2
        if any(layer_name == node.name for node in self.share_w3_nodes):
            return LayerType.SHARED_EXPERT_W3
        if any(layer_name == node.name for node in self.share_gate_hidden_nodes):
            return LayerType.SHARED_EXPERT_GATE_HIDDEN
        if any(layer_name == node.name for node in self.route_w1_nodes):
            return LayerType.ROUTED_EXPERT_W1
        if any(layer_name == node.name for node in self.route_w2_nodes):
            return LayerType.ROUTED_EXPERT_W2
        if any(layer_name == node.name for node in self.route_w3_nodes):
            return LayerType.ROUTED_EXPERT_W3
        if any(layer_name == node.name for node in self.route_gate_hidden_nodes):
            return LayerType.ROUTED_EXPERT_GATE_HIDDEN
        if any(layer_name == node.name for node in self.attn_norm_nodes):
            return LayerType.ATTN_NORM
        if any(layer_name == node.name for node in self.ffn_norm_nodes):
            return LayerType.FFN_NORM
        return LayerType.UNKNOWN

    @staticmethod
    def _get_pre_layer_type(layer_type: LayerType):
        """_get_pre_layer_type"""
        map_ = {LayerType.Q: (LayerType.ATTN_NORM,),
                LayerType.K: (LayerType.ATTN_NORM,),
                LayerType.V: (LayerType.ATTN_NORM,),
                LayerType.QKV: (LayerType.ATTN_NORM,),
                LayerType.Q2L: (LayerType.ATTN_NORM,),
                LayerType.L2Q: (LayerType.Q_NORM,),
                LayerType.KV2L: (LayerType.ATTN_NORM,),
                LayerType.Q2LKV2L: (LayerType.ATTN_NORM,),
                LayerType.O: (LayerType.V, LayerType.QKV),
                LayerType.MLP_W1: (LayerType.FFN_NORM,),
                LayerType.MLP_W3: (LayerType.FFN_NORM,),
                LayerType.MLP_GATE_HIDDEN: (LayerType.FFN_NORM,),
                LayerType.SHARED_EXPERT_W1: (LayerType.FFN_NORM,),
                LayerType.SHARED_EXPERT_W3: (LayerType.FFN_NORM,),
                LayerType.SHARED_EXPERT_GATE_HIDDEN: (LayerType.FFN_NORM,),
                LayerType.ROUTED_EXPERT_W1: (LayerType.FFN_NORM,),
                LayerType.ROUTED_EXPERT_W3: (LayerType.FFN_NORM,),
                LayerType.ROUTED_EXPERT_GATE_HIDDEN: (LayerType.FFN_NORM,),
                }
        return map_.get(layer_type, (LayerType.UNKNOWN,))

    def _get_pre_layer_info(self, pre_layer_types: tuple[LayerType], cur_layer_name: str):
        """_get_pre_layer_info"""
        sibling_nodes = self.name_tree.get_sibling_leaf_nodes(cur_layer_name, 2)
        pre_layer_node = None
        pre_layer_type = LayerType.UNKNOWN
        for sibling_node in sibling_nodes:
            sibling_name = sibling_node.name
            sibling_layer_type = self._get_layer_type(sibling_name)
            if sibling_layer_type is not LayerType.UNKNOWN and sibling_layer_type in pre_layer_types:
                if not pre_layer_node:
                    pre_layer_node = sibling_node
                    pre_layer_type = sibling_layer_type
                else:
                    raise RuntimeError(f'Found multi pre_layer: {pre_layer_node.name}, {sibling_name} '
                                       f'for layer {cur_layer_name}')
        if pre_layer_node:
            return LayerInfo(pre_layer_node.name, pre_layer_node.value, pre_layer_type)
        return None

    def get_pre_layer(self, layer_name: str):
        """get_pre_layer"""
        cur_layer_type = self._get_layer_type(layer_name)
        pre_layer_types = self._get_pre_layer_type(cur_layer_type)
        return self._get_pre_layer_info(pre_layer_types, layer_name)

    def print_self(self):
        """print_self"""
        print("Query names:", self.q_nodes)
        print("Key names:", self.k_nodes)
        print("Value names:", self.v_nodes)
        print("Attention out proj names:", self.v_nodes)
        print("QKV names:", self.qkv_nodes)
        print("Q2L names:", self.q2l_nodes)
        print("L2Q names:", self.l2q_nodes)
        print("KV2L names:", self.kv2l_nodes)
        print("L2KV names:", self.l2kv_nodes)
        print("Q2LKV2L names:", self.q2lkv2l_nodes)
        print("Route W1 names:", self.route_w1_nodes)
        print("Route W2 names:", self.route_w2_nodes)
        print("Route W3 names:", self.route_w3_nodes)
        print("Route Gate Hidden names:", self.route_gate_hidden_nodes)
        print("Share W1 names:", self.share_w1_nodes)
        print("Share W2 names:", self.share_w2_nodes)
        print("Share W3 names:", self.share_w3_nodes)
        print("Share Gate Hidden names:", self.share_gate_hidden_nodes)
        print("MLP W1 names:", self.mlp_w1_nodes)
        print("MLP W2 names:", self.mlp_w2_nodes)
        print("MLP W3 names:", self.mlp_w3_nodes)
        print("MLP Gate Hidden names:", self.mlp_gate_hidden_nodes)
        print("Attention Norm names:", self.attn_norm_nodes)
        print("FFN Norm names:", self.ffn_norm_nodes)
