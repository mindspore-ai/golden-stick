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
"""Class for model's computational graph analysis."""
from mindspore.common.api import _cell_graph_executor as _executor
from mindspore.nn import Cell
from mindspore.train.mind_ir_pb2 import ModelProto as mindir_model

from mindspore_gs.validator import Validator
from .graph import Graph
from .equichannel_group import EquichannelGroup
from .equichannel_group import prunable_layers

# pylint: disable=protected-access
# pylint: disable=broad-except

class GraphAnalyzer:
    """
    Class for model's computational graph analysis. It parses the computation graph, find groups of
    layers with associated chennels and return them to Pruning Algorithm.

    Args:
        net (Cell): network,
        *inputs: network's inputs
    """

    def __init__(self, net, *inputs):
        Validator.check_value_type("net", net, [Cell])
        self._net = net
        self._mindir = None
        self._inputs = inputs
        self._cells = {name: cell for name, cell in self._net.cells_and_names() if not cell.cells()}
        self._graph = None
        self._groups = []
        self.__analyze()

    @property
    def mindir(self):
        """Return model's MindIR graph."""
        return self._mindir

    @property
    def groups(self):
        """Return model's equichannel groups."""
        return self._groups

    @property
    def cells(self):
        """Return model's cells."""
        return self._cells

    def __analyze(self):
        """Analyze graph."""
        self.__parse_mindir()
        self.__shrink()
        self.__split_groups()

    def __parse_mindir(self):
        """Analyze computational graph from .MINDIR."""
        model = mindir_model()
        phase_name = "export.mindir"
        graph_id, _ = _executor.compile(self._net, *self._inputs, phase=phase_name,
                                        do_convert=False)
        mindir_stream = _executor._get_func_graph_proto(self._net, graph_id, 'mind_ir')
        model.ParseFromString(mindir_stream)
        self._mindir = model
        self._graph = Graph(self._mindir.graph)

    def __shrink(self):
        """Shrink function."""
        for_shrink = ["Depend", "UpdateState", "MakeTuple", "Constant"]
        for node in list(self._graph.nodes.values()):
            if node.op_type in for_shrink:
                self._graph.remove_node(node)

        for node in list(self._graph.nodes.values()):
            if node.op_type == "Load" and len(node.inputs) == 1:
                self._graph.rename_node(node, node.inputs[0])
                node.inputs.pop(0)

        # For the network cell, like: the conv2d cell with bias will be compiled to two node: Conv2D, BiasAdd
        # 1. Rename node name with cell name.
        # 2. Merge two nodes to the front node.
        for node in list(self._graph.nodes.values()):
            load_names = set()
            for inp in node.inputs:
                if (inp in self._graph.nodes) and (self._graph.nodes[inp].op_type == "Load"):
                    load_name = self._graph.nodes[inp].name
                    load_name = load_name[:load_name.rfind(".")]
                    load_names.add(load_name)
            if len(load_names) == 1:
                new_name = list(load_names)[0]
                for inp in list(node.inputs):
                    if (inp in self._graph.nodes) and (self._graph.nodes[inp].op_type == "Load"):
                        self._graph.remove_node(self._graph.nodes[self._graph.nodes[inp].name])
                if new_name in self._graph.nodes:
                    self._graph.merge_nodes(self._graph.nodes[new_name], node)
                else:
                    self._graph.rename_node(node, new_name)

    def __split_groups(self):
        """Separate layer groups."""
        def get_all_connected_groups(merge_group_idx):
            already_seen = set()
            result = []
            for node in merge_group_idx:
                if node not in already_seen:
                    connected_group, already_seen = get_connected_group(merge_group_idx, node, already_seen)
                    result.append(connected_group)
            return result

        def get_connected_group(merge_group_idx, node, already_seen):
            result = []
            nodes = {node}
            while nodes:
                node = nodes.pop()
                already_seen.add(node)
                nodes = nodes or merge_group_idx[node] - already_seen
                result.append(node)
            return result, already_seen

        groups = []
        merge_groups = []
        for inp in self.mindir.graph.input:
            name = inp.name.replace(self.mindir.graph.name + ":", "")
            group = EquichannelGroup(name, self._graph)
            if group.induce_layers:
                merge_groups.append(group)
            else:
                groups.append(group)

        for node in self._graph.nodes.values():
            if node.op_type in prunable_layers:
                group = EquichannelGroup(node.name, self._graph)
                if group.induce_layers:
                    merge_groups.append(group)
                else:
                    groups.append(group)

        merge_group_idx = {}
        for i, gr1 in enumerate(merge_groups):
            merge_group_idx[i] = set()
            for j, gr2 in enumerate(merge_groups):
                if i == j:
                    continue
                if set(gr1.induce_layers) & set(gr2.induce_layers):
                    merge_group_idx[i].add(j)

        connected_groups = get_all_connected_groups(merge_group_idx)
        for component in connected_groups:
            group = merge_groups[component.pop(0)]
            for grp in component:
                group.merge(merge_groups[grp])
            groups.append(group)
        self._groups = groups

        for group in self._groups:
            group.map_to_network_cell(self._cells)
