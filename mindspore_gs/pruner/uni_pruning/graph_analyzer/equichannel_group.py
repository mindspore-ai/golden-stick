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
"""Class for model's equal channel group."""

from .graph import Graph


# all prunable layers
prunable_layers = ["Conv2D", "MatMul"]  # "ConvTranspose", "Gemm"

# layers which will be affected by prunable layers.
affectable_layers = ["BatchNorm", "BiasAdd"]  # "PRelu"

# layers which induce constraints between different layers pruning.
induce_constraint_layers = ["Add"]  # "Mul", "Sub", "Pow", "Max", "Min", "Concat"


class EquichannelGroup:
    """Class for representing pruning group."""

    def __init__(self, initial_layer, graph: Graph):
        self.__initial_layer = initial_layer
        self.__starts = {initial_layer}
        self.__middles = set()
        self.__ends = set()
        self.__subgraph = set()
        self.__induce_layers = set()
        self.__conv_bn_pairs = []
        self.__closure(graph.nodes)

    def __repr__(self):
        return f"starts: {self.__starts}, \n" \
               f"middles: {self.__middles}, \n" \
               f"ends: {self.__ends}, \n" \
               f"induce_layers: {self.__induce_layers}, \n" \
               f"subgraph: {self.__subgraph} \n"

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

    def merge(self, group):
        """Merges two groups."""
        self.__starts |= group.ms_starts
        self.__middles |= group.ms_middles
        self.__ends |= group.ms_ends
        self.__induce_layers |= group.induce_layers
        self.__subgraph |= group.subgraph

    def map_to_network_cell(self, cells):
        """Map mindir layers to network cells."""
        self.__starts = {key: cells[key] for key in self.__starts if key in cells}
        self.__middles = {key: cells[key] for key in self.__middles if key in cells}
        self.__ends = {key: cells[key] for key in self.__ends if key in cells}

    def __get_all_paths(self, node_id, nodes, path=None):
        """Get all paths."""
        paths = []
        if path is None:
            path = []
            self.__subgraph.add(node_id)
        path.append(node_id)

        if node_id in nodes:
            outputs = list(nodes[node_id].outputs)
        else:
            outputs = []
            for node in nodes.values():
                if node_id in node.inputs:
                    outputs.append(node.name)

        if not outputs:
            paths.append(path)

        for out in outputs:
            if out in nodes:
                self.__subgraph.add(out)
                if nodes[out].op_type in prunable_layers:
                    self.ms_ends.add(out)
                    continue
                if nodes[out].op_type in affectable_layers:
                    self.ms_middles.add(out)
                if nodes[out].op_type in induce_constraint_layers:
                    self.induce_layers.add(out)
            paths.extend(self.__get_all_paths(out, nodes, path[:]))
        return paths

    def __closure(self, nodes):
        """Get all paths from start layer."""
        _ = self.__get_all_paths(self.__initial_layer, nodes)
