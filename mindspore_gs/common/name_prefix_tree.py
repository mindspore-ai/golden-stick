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
"""name prefix tree"""


import os
from mindspore_gs.common.logger import logger


class Node:
    """Node"""
    def __init__(self, segment, name):
        self.segment = segment
        self.children = {}
        self.name = name
        self.value = None

    def __lt__(self, other):
        return self.name.__lt__(other.name)

    def __gt__(self, other):
        return self.name.__gt__(other.name)


class NameTree:
    """NameTree"""
    def __init__(self, root_name='network'):
        self.root = Node(root_name, root_name)

    def add_name(self, name, value):
        """add_name"""
        segments = name.split('.')
        current = self.root
        current_name = ''
        for seg in segments:
            current_name = current_name + '.' + seg if current_name else seg
            if seg not in current.children:
                current.children[seg] = Node(seg, current_name)
            current = current.children[seg]
        if current.value is not None:
            raise RuntimeError(f'Name duplicated: {name}')
        current.value = value

    def _get_node_by_segments(self, segments):
        """_get_node_by_segments"""
        current = self.root
        for seg in segments:
            if seg not in current.children:
                return None
            current = current.children[seg]
        return current

    def _get_node_by_name(self, name):
        """_get_node_ny_name"""
        if name == self.root.name:
            return self.root
        segments = name.split('.')
        return self._get_node_by_segments(segments)

    def find_leafs_with_keywords(self, keywords, blacklist=None):
        """find_leafs_with_keywords"""
        if blacklist is None:
            blacklist = []
        results = []

        def traverse(node):
            if node.value is not None:
                if (all(keyword in node.name for keyword in keywords) and
                        not any(exclude_name in node.name for exclude_name in blacklist)):
                    results.append(node)
            for child in node.children.values():
                traverse(child)

        traverse(self.root)
        return results

    def find_leafs_with_prefix(self, prefix):
        """Find leaf nodes with prefix"""
        target_node = self._get_node_by_name(prefix)
        return self._get_leaf_nodes(target_node)

    def _get_parent_node(self, node):
        """_get_node_and_parent"""
        segments = node.name.split('.')
        current = self.root
        parent = None
        for seg in segments:
            if seg not in current.children:
                return None
            parent = current
            current = current.children[seg]
        return parent

    @staticmethod
    def _get_leaf_nodes(node):
        """Get all leaf nodes from the given node"""
        if not node:
            return []
        leaf_nodes = []

        def traverse(current_node):
            if not current_node.children:  # 如果当前节点是叶子节点
                leaf_nodes.append(current_node)
            for child in current_node.children.values():
                traverse(child)

        traverse(node)
        return leaf_nodes

    def get_sibling_leaf_nodes(self, name, level=1):
        """Get sibling leaf nodes"""
        target_node = self._get_node_by_name(name)
        if not target_node:
            return []
        if target_node.children:
            raise RuntimeError(f'Only leaf node has sibling leaf nodes, but got {name}')

        # Special case when level is 0
        if level == 0:
            return []

        # Traverse up the specified number of levels to find the ancestor node
        ancestor = target_node
        for _ in range(level):
            if ancestor is None:
                break
            ancestor = self._get_parent_node(ancestor)

        if ancestor is None:
            return []

        sibling_leaves = self._get_leaf_nodes(ancestor)
        index = 0
        while True:
            if sibling_leaves[index] is target_node:
                break
            index += 1
        sibling_leaves.pop(index)
        return sibling_leaves

    def generate_dot(self, file_path):
        """generate_dot"""
        dot = "digraph Tree {\n"
        dot += 'node [shape=box, style="filled", color="lightgrey"];\n'

        def traverse(node, parent_id=None):
            nonlocal dot
            current_id = f'node_{id(node)}'
            dot += f'{current_id} [label="{node.segment}"];\n'
            if parent_id is not None:
                dot += f'{parent_id} -> {current_id};\n'
            for child in node.children.values():
                traverse(child, current_id)

        traverse(self.root)
        dot += "}"

        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        with open(file_path, "w") as f:
            f.write(dot)

        logger.info(f"DOT save to: {file_path}")
