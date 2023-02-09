# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
"""Class for model's computational graph."""
import re
from itertools import count
from mindspore.train.mind_ir_pb2 import GraphProto


class Node:
    """Class for representing node."""
    _ids = count(0)

    def __init__(self, node, graph_name):
        self._index = next(self._ids)
        self._graph_name = graph_name
        self._name = self.__delete_graph_name(node.name)
        self._op_type = re.search(r"::(.*):", node.op_type).group(1) if \
            re.search(r"::(.*):", node.op_type) else node.op_type
        self._inputs = [self.__delete_graph_name(inp) for inp in node.input]
        self._outputs = [self.__delete_graph_name(out) for out in node.output]
        if self._name in self._outputs:
            self._outputs.remove(self._name)

    def __delete_graph_name(self, name):
        """Delete graph name from node name to express node more clearly."""
        return name.replace(self._graph_name + ":", "")

    def __repr__(self):
        return f"name: {self._name}, " \
               f"op_type: {self._op_type}, " \
               f"inputs: {self._inputs}, " \
               f"outputs: {self._outputs} "

    @property
    def index(self):
        """Return Node index"""
        return self._index

    @property
    def name(self):
        """Return Node name"""
        return self._name

    @name.setter
    def name(self, name):
        """Set Node name"""
        self._name = name

    @property
    def op_type(self):
        """Return Node type"""
        return self._op_type

    @property
    def inputs(self):
        """Return Nodes inputs list"""
        return self._inputs

    @property
    def outputs(self):
        """Return Nodes outputs list"""
        return self._outputs


class Graph:
    """Represent graph with ordered nodes."""
    def __init__(self, mindir_graph: GraphProto):
        self._nodes = {}
        self.__init_graph(mindir_graph)

    def __init_graph(self, mindir_graph: GraphProto):
        """Init nodes list with mindir graph."""
        for mindir_node in mindir_graph.node:
            node = Node(mindir_node, mindir_graph.name)
            self._nodes[node.name] = node

        for node in self._nodes.values():
            for inp in node.inputs:
                if inp in self._nodes:
                    self._nodes[inp].outputs.append(node.name)

        # rename node with op_type+index to express the node more clearly
        for node in list(self._nodes.values()):
            new_name = f"{node.op_type}_{node.index}"
            self.rename_node(node, new_name)

    def remove_node(self, node_to_remove: Node):
        """Remove node."""
        for node in self.nodes.values():
            if node_to_remove.name in node.inputs:
                node.inputs.remove(node_to_remove.name)
            if node_to_remove.name in node.outputs:
                node.outputs.remove(node_to_remove.name)
        del self.nodes[node_to_remove.name]

    def rename_node(self, cur_node: Node, new_name):
        """Rename node."""
        # create an element with new name key to origin node
        del self.nodes[cur_node.name]
        self.nodes[new_name] = cur_node
        for node in self.nodes.values():
            if cur_node.name in node.inputs:
                node.inputs.remove(cur_node.name)
                node.inputs.append(new_name)
            if cur_node.name in node.outputs:
                node.outputs.remove(cur_node.name)
                node.outputs.append(new_name)
        self.nodes[new_name].name = new_name

    def merge_nodes(self, node: Node, node_to_merge: Node):
        """Merge nodes."""
        if node_to_merge.name in node.outputs:
            node.outputs.remove(node_to_merge.name)
        for inp in node_to_merge.inputs:
            if (inp not in node.inputs) and (inp != node.name):
                node.inputs.append(inp)
                self.nodes[inp].outputs.remove(node_to_merge.name)
                self.nodes[inp].outputs.append(node.name)
        for out in node_to_merge.outputs:
            if out not in node.outputs:
                node.outputs.append(out)
                self.nodes[out].inputs.remove(node_to_merge.name)
                self.nodes[out].inputs.append(node.name)
        del self.nodes[node_to_merge.name]

    @property
    def nodes(self):
        """Return Graph nodes."""
        return self._nodes
