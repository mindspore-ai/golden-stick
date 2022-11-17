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
"""Class for model's computational graph analysis."""
import re
from itertools import count

from mindspore import log as logger
from mindspore._checkparam import Validator
from mindspore.common.api import _cell_graph_executor as _executor
from mindspore.nn import Cell
from mindspore.train.mind_ir_pb2 import ModelProto as mindir_model

#pylint: disable=too-many-nested-blocks
#pylint: disable=protected-access
#pylint: disable=broad-except


class GraphAnalyzer:
    """
    Class for model's computational graph analysis. It parses the computation graph, find groups of
    layers with associated chennels and return them to Pruning Algorithm.

    Args:
        net (Cell): network,
        *inputs: network's inputs
    """

    class Node:
        """Class for representing mindir node."""

        _ids = count(0)

        def __init__(self, node, analyzer):
            self._index = next(self._ids)
            self._analyzer = analyzer
            self._name = self.__delete_graph_name(node.name)
            self._op_type = re.search(r"::(.*):", node.op_type).group(1) if \
                re.search(r"::(.*):", node.op_type) else node.op_type
            self._inputs = [self.__delete_graph_name(inp) for inp in node.input]
            self._outputs = [self.__delete_graph_name(out) for out in node.output]
            if self._name in self._outputs:
                self._outputs.remove(self._name)

        def __delete_graph_name(self, name):
            return name.replace(self._analyzer.mindir.graph.name + ":", "")

        def __repr__(self):
            return f"[name: {self._name}, op_type: {self._op_type}]"

        def rename(self, new_name):
            """Rename node."""
            del self._analyzer.nodes[self.name]
            self._analyzer.nodes[new_name] = self
            for node in self._analyzer.nodes.values():
                if self._name in node.inputs:
                    node.inputs.remove(self._name)
                    node.inputs.append(new_name)
                if self._name in node.outputs:
                    node.outputs.remove(self._name)
                    node.outputs.append(new_name)
            self._name = new_name

        def merge_next(self, node_to_merge):
            """Merge nodes."""
            self.outputs.remove(node_to_merge.name)
            for inp in node_to_merge.inputs:
                if (inp not in self.inputs) and (inp != self.name):
                    self.inputs.append(inp)
                    self._analyzer.nodes[inp].outputs.remove(node_to_merge.name)
                    self._analyzer.nodes[inp].outputs.append(self.name)
            for out in node_to_merge.outputs:
                if out not in self.outputs:
                    self.outputs.append(out)
                    self._analyzer.nodes[out].inputs.remove(node_to_merge.name)
                    self._analyzer.nodes[out].inputs.append(self.name)
            del self._analyzer.nodes[node_to_merge.name]

        @property
        def index(self):
            """Node's index"""
            return self._index

        @property
        def name(self):
            """Node's name"""
            return self._name

        @property
        def op_type(self):
            """Node's operation type"""
            return self._op_type

        @property
        def inputs(self):
            """Node's inputs"""
            return self._inputs

        @property
        def outputs(self):
            """Node's outputs"""
            return self._outputs

    class EquichannelGroup:
        """Class for representing pruning group."""

        def __init__(self, initial_layer, graph):
            self.__graph = graph
            self.__subgraph = set()
            self.__initial_layer = initial_layer
            self.__starts = {initial_layer}
            self.__middles = set()
            self.__ends = set()
            self.__conv_bn_pairs = []
            self.__nodes = graph.nodes
            self.__induce_layers = set()
            self.__closure()

        @property
        def ms_starts(self):
            """Returns Mindspore start layers of group."""
            return self.__starts

        @property
        def ms_middles(self):
            """Returns Mindspore middle layers of group."""
            return self.__middles

        @property
        def ms_ends(self):
            """Returns Mindspore end layers of group."""
            return self.__ends

        @property
        def subgraph(self):
            """Returns subgraph layers."""
            return self.__subgraph

        @property
        def induce_layers(self):
            """Returns induce layers."""
            return self.__induce_layers

        @property
        def ms_conv_bn_pairs(self):
            """Returns conv-bn pairs."""
            return self.__conv_bn_pairs

        def merge(self, group):
            """Merges two groups."""
            self.__starts |= group.ms_starts
            self.__middles |= group.ms_middles
            self.__ends |= group.ms_ends
            self.__induce_layers |= group.induce_layers
            self.__subgraph |= group.subgraph
            self.__conv_bn_pairs += group.ms_conv_bn_pairs

        def __get_all_paths(self, node_id, path=None):
            """Get all paths."""
            paths = []
            if path is None:
                path = []
                self.__subgraph.add(node_id)
            path.append(node_id)

            if node_id in self.__nodes:
                outputs = list(self.__nodes[node_id].outputs)
            else:
                outputs = []
                for node in self.__nodes.values():
                    if node_id in node.inputs:
                        outputs.append(node.name)

            if outputs:
                for out in outputs:
                    if out in self.__nodes:
                        self.__subgraph.add(out)
                        #TODO process depthwise convs and PRelu
                        if self.__nodes[out].op_type in self.__graph.prunable_layer_types():
                            self.ms_ends.add(out)
                            continue
                        if self.__nodes[out].op_type in self.__graph.affectable_layer_types():
                            if self.__nodes[out].op_type == "BatchNorm":
                                if self.__nodes[path[-1]].op_type == "BiasAdd":
                                    if self.__nodes[path[-2]].op_type in ["Conv2D", "MatMul"]:
                                        self.__conv_bn_pairs.append((path[-2], out))
                                else:
                                    if self.__nodes[path[-1]].op_type in ["Conv2D", "MatMul"]:
                                        self.__conv_bn_pairs.append((path[-1], out))
                            self.ms_middles.add(out)
                        if self.__nodes[out].op_type in self.__graph.induce_constraint_ops():
                            self.induce_layers.add(out)
                    paths.extend(self.__get_all_paths(out, path[:]))
            else:
                paths.append(path)
            return paths

        def __closure(self):
            """Get all paths from start layer."""
            self.__get_all_paths(self.__initial_layer)

        def map_mindspore(self):
            """Map onnx layers to Mindspore layers."""
            self.__starts = {key: self.__graph.cells[key] for key in self.__starts if key in \
                self.__graph.cells}
            self.__middles = {key: self.__graph.cells[key] for key in self.__middles}
            self.__ends = {key: self.__graph.cells[key] for key in self.__ends}

    def __init__(self, net, *inputs):
        Validator.check_value_type("net", net, [Cell])
        self._net = net
        self._mindir = None
        self._inputs = inputs
        self._nodes = {}
        self._input_nodes = []
        self._cells = {name: cell for name, cell in self._net.cells_and_names() if not cell.cells()}
        self._layers_map = {}
        self._groups = []

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

    @property
    def nodes(self):
        """Return model's nodes."""
        return self._nodes

    @property
    def input_nodes(self):
        """Return model's input_nodes."""
        return self._input_nodes

    def analyze(self):
        """Analyze graph."""
        try:
            self.__parse_mindir()
            self.__shrink()
            self.__split_groups()
            return True
        except Exception as err:
            logger.error(err)
            return False

    @staticmethod
    def prunable_layer_types():
        """Return all prunable layer types."""
        return [
            "Conv2D", "MatMul" #"ConvTranspose", "Gemm"
        ]

    @staticmethod
    def affectable_layer_types():
        """Return all operations which induce constraints between different layers pruning."""
        return [
            "BatchNorm", "BiasAdd", #, "PRelu"
        ]

    @staticmethod
    def induce_constraint_ops():
        """Return all operations which induce constraints between different layers pruning."""
        return [
            "Add" #, "Mul", "Sub", "Pow", "Max", "Min"
        ]

    def __parse_mindir(self):
        """Analyze computational graph from .MINDIR."""
        model = mindir_model()
        phase_name = "export.mindir"
        graph_id, _ = _executor.compile(self._net, *self._inputs, phase=phase_name,
                                        do_convert=False)
        mindir_stream = _executor._get_func_graph_proto(self._net, graph_id, 'mind_ir')
        model.ParseFromString(mindir_stream)
        self._mindir = model
        for mindir_node in model.graph.node:
            node = self.Node(mindir_node, self)
            self._nodes[node.name] = node
        for node in self._nodes.values():
            for inp in node.inputs:
                if inp in self._nodes:
                    self._nodes[inp].outputs.append(node.name)
        for node in list(self._nodes.values()):
            new_name = f"{node.op_type}_{node.index}"
            node.rename(new_name)

    def __remove_node(self, node_to_remove: Node):
        for node in self.nodes.values():
            if node_to_remove.name in node.inputs:
                node.inputs.remove(node_to_remove.name)
            if node_to_remove.name in node.outputs:
                node.outputs.remove(node_to_remove.name)
        del self.nodes[node_to_remove.name]

    def __remove_optype_nodes(self, op_type):
        for node in list(self.nodes.values()):
            if node.op_type == op_type:
                self.__remove_node(node)

    def __shrink(self):
        """Shrink function."""
        for_shrink = ["Depend", "UpdateState", "MakeTuple", "Constant"]
        for op_type in for_shrink:
            self.__remove_optype_nodes(op_type)

        for node in list(self.nodes.values()):
            if node.op_type == "Load" and len(node.inputs) == 1:
                node.rename(node.inputs[0])
                node.inputs.pop(0)

        for node in list(self.nodes.values()):
            load_names = set()
            for inp in node.inputs:
                if (inp in self.nodes) and (self.nodes[inp].op_type == "Load"):
                    load_name = self.nodes[inp].name
                    load_name = load_name[:load_name.rfind(".")]
                    load_names.add(load_name)
            if len(load_names) == 1:
                new_name = list(load_names)[0]
                for inp in list(node.inputs):
                    if (inp in self.nodes) and (self.nodes[inp].op_type == "Load"):
                        self.__remove_node(self.nodes[self.nodes[inp].name])
                if new_name in self.nodes:
                    self.nodes[new_name].merge_next(node)
                else:
                    node.rename(new_name)

    def __split_groups(self):
        """Separate layer groups."""
        def get_all_connected_groups(graph):
            already_seen = set()
            result = []
            for node in graph:
                if node not in already_seen:
                    connected_group, already_seen = get_connected_group(graph, node, already_seen)
                    result.append(connected_group)
            return result

        def get_connected_group(graph, node, already_seen):
            result = []
            nodes = set([node])
            while nodes:
                node = nodes.pop()
                already_seen.add(node)
                nodes = nodes or graph[node] - already_seen
                result.append(node)
            return result, already_seen

        groups = []
        merge_groups = []
        for inp in self.mindir.graph.input:
            name = inp.name.replace(self.mindir.graph.name + ":", "")
            self._input_nodes.append(name)
            group = self.EquichannelGroup(name, self)
            if group.induce_layers:
                merge_groups.append(group)
            else:
                groups.append(group)

        for node in self._nodes.values():
            if node.op_type in self.prunable_layer_types():
                group = self.EquichannelGroup(node.name, self)
                if group.induce_layers:
                    merge_groups.append(group)
                else:
                    groups.append(group)

        merge_graph = {}
        for i, gr1 in enumerate(merge_groups):
            merge_graph[i] = set()
            for j, gr2 in enumerate(merge_groups):
                if i == j:
                    continue
                if set(gr1.induce_layers) & set(gr2.induce_layers):
                    merge_graph[i].add(j)

        connected_groups = get_all_connected_groups(merge_graph)
        for component in connected_groups:
            group = merge_groups[component.pop(0)]
            for grp in component:
                group.merge(merge_groups[grp])
            groups.append(group)
        self._groups = groups

        for group in self._groups:
            group.map_mindspore()
