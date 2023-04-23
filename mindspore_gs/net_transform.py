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

from mindspore.nn.cell import Cell
from mindspore.rewrite import PatternEngine, SymbolTree, Node, ScopedValue, TreeNodeHelper


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
            except (RuntimeError, ValueError, TypeError, NotImplementedError):
                raise RuntimeError(f"For MindSpore Golden Stick, input network type '{type(net).__name__}' "
                                   f"is not supported right now.")
            except Exception:
                raise Exception("For MindSpore Golden Stick, analysis input network fail.")
            return
        self._symbol_tree = symbol_tree

    @staticmethod
    def create_from_tree_node(node):
        modify_tree = TreeNodeHelper.get_sub_tree(node)
        return NetTransformer(None, modify_tree)

    def get_network(self) -> Cell:
        return self._symbol_tree.get_network()

    def get_code(self):
        return self._symbol_tree.get_code()

    def nodes(self) -> {}:
        """
        Returns:
            a list of BaseNode corresponding to all layers in original network.
        """

        return self._symbol_tree.nodes()

    def before(self, node_or_name: Union[Node, str]):
        return self._symbol_tree.before(node_or_name)

    def after(self, node_or_name: Union[Node, str]):
        return self._symbol_tree.after(node_or_name)

    @staticmethod
    def create_node(cell: Cell, targets: [Union[ScopedValue, str]], args: [ScopedValue] = None,
                    kwargs: {str: ScopedValue}=None, name: str = "") -> Node:
        return Node.create_call_cell(cell, targets, args, kwargs, name)

    def insert(self, position, node: Node) -> Optional[Node]:
        return self._symbol_tree.insert(position, node)

    def erase_node(self, node_or_name: Union[Node, str]) -> Optional[Node]:
        """
        Args:
            node_or_name (Node/str): node to be removed from original network.

        Returns:
            BaseNode has been removed, return None if failed
        """
        return self._symbol_tree.erase_node(node_or_name)

    def replace(self, old_node: Node, new_nodes: [Node]) -> Node:
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
        return self._symbol_tree.replace(old_node, new_nodes)

    def dump(self):
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
