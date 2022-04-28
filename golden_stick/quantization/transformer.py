# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Transformer."""
from mindspore.rewrite import PatternEngine, PatternNode, Node, SymbolTree
from typing import List, Union
from .layer_policy import LayerPolicy, layer_policy_key


# Only support for fusion currently
class Transformer(PatternEngine):
    def __init__(self, pattern: Union[PatternNode, List]):
        super().__init__(pattern, None)
        self._node_visited_key: str = "is_node_visited"

    def _is_node_visited(self, node: Node) -> bool:
        assert node is not None
        return node.get_attribute(self._node_visited_key)

    def _set_node_visited(self, node: Node):
        assert node is not None
        node.set_attribute(self._node_visited_key, True)

    def apply(self, symbol_tree: SymbolTree) -> bool:
        root: Node = symbol_tree.get_return_node()
        # IR match
        queue: [Node] = [root]
        while len(queue) > 0:
            cur_node: Node = queue.pop(0)
            cur_node_inputs = cur_node.get_inputs()
            matched, matched_dict = self._match(self._pattern, cur_node)
            if not matched or not PatternEngine._check_match(self._pattern, matched_dict):
                for cur_node_input in cur_node_inputs:
                    queue.append(cur_node_input)
                continue
            matched_list = list(matched_dict.values())
            overlapped = False
            for matched_node in matched_list:
                if self._is_node_visited(matched_node):
                    overlapped = True
                    break
            if overlapped:
                for cur_node_input in cur_node_inputs:
                    queue.append(cur_node_input)
                continue
            for matched_node in matched_list:
                self._set_node_visited(matched_node)

            # modify layer policy
            # find input output
            inputs_of_matched: [Node] = []
            output = matched_dict.get(self._pattern.name())
            inputs: [] = []
            for matched_node in matched_list:
                node_inputs = matched_node.inputs
                is_input_node = False
                for node_input in node_inputs:
                    if node_input in matched_dict.values():
                        continue
                    inputs_of_matched.append(node_input)
                    is_input_node = True
                if is_input_node:
                    inputs.append(matched_node)
            # remove inter-matched-node-policy
            for matched_node in matched_list:
                node_policy: LayerPolicy = matched_node.get_attribute(layer_policy_key)
                if node_policy is None:
                    continue
                is_input = matched_node in inputs
                is_output = matched_node == output
                if not is_input and not is_output:
                    node_policy.set_input_not_insert_fq()
                    node_policy.set_output_not_insert_fq()
                    continue
                if is_input and not is_output:
                    node_policy.set_output_not_insert_fq()
                    continue
                if is_output and not is_input:
                    node_policy.set_input_not_insert_fq()
                    continue
                for i in range(0, len(matched_node.inputs)):
                    node_input = matched_node.inputs[i]
                    if node_input in inputs_of_matched:
                        continue
                    node_policy.set_input_not_insert_fq(i)

            for input_of_matched in inputs_of_matched:
                queue.append(input_of_matched)
        return False
