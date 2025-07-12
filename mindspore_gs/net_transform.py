# Copyright 2022 Huawei Technologies Co., Ltd
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
"""NetTransform."""
from typing import Union, Optional

import mindspore
from mindspore.nn.cell import Cell
from mindspore.rewrite import PatternEngine, SymbolTree, Node, ScopedValue, NodeType
from mindspore_gs.quantization.quant_cell import QuantCell


mindspore.rewrite.common.namespace._subtree_black_list.append(QuantCell)


class NetTransformer:
    """
    NetTransformer is define for transform network in MindSpore.

    Args:
        net (Cell): Network to be transformed.
    """

    def __init__(self, net: Cell, symbol_tree: SymbolTree = None):
        if symbol_tree is None:
            self._net = net
            try:
                self._symbol_tree = SymbolTree.create(net)
                self._symbol_tree.flatten_static_if_control_flow()
            except (RuntimeError, ValueError, TypeError, NotImplementedError):
                raise RuntimeError(f"For MindSpore Golden Stick, input network type '{type(net).__name__}' "
                                   f"is not supported right now.")
            except Exception as e:
                raise Exception(f"For MindSpore Golden Stick, analysis input network fail.\n{e}")
            return
        self._symbol_tree = symbol_tree

    @staticmethod
    def create_from_tree_node(node):
        """create_from_tree_node"""
        modify_tree = node.get_sub_tree()
        return NetTransformer(None, modify_tree)

    @staticmethod
    def create_node(cell: Cell, targets: [Union[ScopedValue, str]], args: [ScopedValue] = None,
                    kwargs: {str: ScopedValue}=None, name: str = "") -> Node:
        """create_node"""
        return Node.create_call_cell(cell, targets, args, kwargs, name)

    def get_network(self) -> Cell:
        """get_network"""
        return self._symbol_tree.get_network()

    def get_code(self):
        """get_code"""
        return self._symbol_tree.get_code()

    def unfolded_nodes(self) -> {}:
        """
        Get a generator to generate unfolded nodes of corresponding network.
        """

        for node in self._symbol_tree.nodes():
            for single_node in NodeUnfolder.unfold_nodes(node):
                yield single_node

    def nodes(self) -> {}:
        """
        Returns:
            a list of BaseNode corresponding to all layers in original network.
        """

        return self._symbol_tree.nodes()

    def before(self, node_or_name: Union[Node, str]):
        """before"""
        return self._symbol_tree.before(node_or_name)

    def after(self, node_or_name: Union[Node, str]):
        """after"""
        return self._symbol_tree.after(node_or_name)

    def insert(self, position, node: Node) -> Optional[Node]:
        """insert"""
        return self._symbol_tree.insert(position, node)

    def erase_node(self, node_or_name: Union[Node, str]) -> Optional[Node]:
        """
        Args:
            node_or_name (Node/str): node to be removed from original network.

        Returns:
            BaseNode has been removed, return None if failed
        """
        return self._symbol_tree.erase_node(node_or_name)

    @staticmethod
    def replace(old_node: Node, new_nodes: [Node]) -> Node:
        """
        Replace an old_node with new_node from rewrite. Can only erase a node not being depended on.

        Args:
            old_node (Node): Node to be replaced.
            new_nodes (list[Node]): Node to replace

        Returns:
            new node.

        Raises:
            RuntimeError: old node is isolated.
        """
        stree = SymbolTree(old_node.get_handler().get_belong_symbol_tree())
        return stree.replace(old_node, new_nodes)

    def dump(self):
        """dump"""
        self._symbol_tree.dump()

    # replace src_pattern with target_nodes.
    # target_nodes should has same inputs and outputs with src_pattern.
    def pattern_transform(self, pattern_engine: PatternEngine) -> bool:
        """
        Args:
            pattern_engine (PatternEngine): Instance of PatternEngine. Apply `pattern_engine` on current network.

        Returns:
            a bool value indicating if transform occurred
        """
        return pattern_engine.apply(self._symbol_tree)


class NodeUnfolder:
    """
    Unfold nodes from symbol tree.
    node_type_to_unfold_list: [NodeType.Tree, NodeType.CellContainer]
    """

    @staticmethod
    def _get_nodes_from_cell_container(node: Node):
        cell_container = node.get_handler()
        for i, sub_node in enumerate(cell_container.nodes()):
            if i == 0 and not sub_node.get_inputs():
                sub_node.set_arg_providers(0, (cell_container.get_inputs()[0], 0))
            for single_node in NodeUnfolder.unfold_nodes(Node(sub_node)):
                yield single_node

    @staticmethod
    def _get_nodes_from_sub_tree(node: Node):
        sub_tree: SymbolTree = node.get_handler().symbol_tree
        for sub_node in sub_tree.nodes():
            for single_node in NodeUnfolder.unfold_nodes(Node(sub_node)):
                yield single_node

    @staticmethod
    def unfold_nodes(node: Node):
        """
        Unfold cell container or subtree.
        """
        node_type: NodeType = node.get_node_type()
        if node_type == NodeType.CellContainer:
            for unfolded_node in NodeUnfolder._get_nodes_from_cell_container(node):
                yield unfolded_node
        elif node_type == NodeType.Tree:
            for unfolded_node in NodeUnfolder._get_nodes_from_sub_tree(node):
                yield unfolded_node
        else:
            yield node
