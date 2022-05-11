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
"""test Quantize."""

import unittest
from test_common import TestCommon
from mindspore.golden_stick import Transformer
from mindspore.rewrite import Graph
from mindspore.nn import Cell, Conv2d, BatchNorm2d


class TransformerTestCase(unittest.TestCase):
    @staticmethod
    def network():
        placeholder = TestCommon.create_placeholder_layer()

        conv1 = TestCommon.create_conv_layer("conv1", [placeholder])
        placeholder.outputs = [conv1]

        bn1 = TestCommon.create_bn_layer("bn1", [conv1])
        conv1.outputs = [bn1]

        pool1 = TestCommon.create_pool_layer("pool1", [bn1])
        bn1.outputs = [pool1]

        conv2 = TestCommon.create_conv_layer("conv2", [pool1])
        pool1.outputs = [conv2]

        bn2 = TestCommon.create_bn_layer("bn2", [conv2])
        conv2.outputs = [bn2]

        pool2 = TestCommon.create_pool_layer("pool2", [bn2])
        bn2.outputs = [pool2]

        graph = Graph(Cell())
        graph.set_root(pool2)
        return graph

    @staticmethod
    def network_intra_overlapped():
        placeholder = TestCommon.create_placeholder_layer()

        conv1 = TestCommon.create_conv_layer("conv1", [placeholder])
        placeholder.outputs = [conv1]

        conv2 = TestCommon.create_conv_layer("conv2", [conv1])
        conv1.outputs = [conv2]

        conv3 = TestCommon.create_conv_layer("conv3", [conv2])
        conv2.outputs = [conv3]

        bn = TestCommon.create_bn_layer("bn", [conv3])
        conv3.outputs = [bn]

        pool = TestCommon.create_pool_layer("pool", [bn])
        bn.outputs = [pool]

        graph = Graph(Cell())
        graph.set_root(pool)
        return graph

    @staticmethod
    def network_inter_overlapped():
        placeholder = TestCommon.create_placeholder_layer()

        conv1 = TestCommon.create_conv_layer("conv1", [placeholder])
        placeholder.outputs = [conv1]

        conv2 = TestCommon.create_conv_layer("conv2", [conv1])
        conv1.outputs = [conv2]

        bn = TestCommon.create_bn_layer("bn", [conv2])
        conv2.outputs = [bn]

        pool = TestCommon.create_pool_layer("pool", [bn])
        bn.outputs = [pool]

        graph = Graph(Cell())
        graph.set_root(pool)
        return graph

    def test_inter_overlap(self):
        transformer1: Transformer = Transformer([Conv2d, Conv2d])
        transformer2: Transformer = Transformer([Conv2d, BatchNorm2d])
        graph: Graph = TransformerTestCase.network_inter_overlapped()
        conv1_in_fq = TransformerTestCase._get_node_of_graph_inout_fq(graph, 1, True)
        conv1_out_fq = TransformerTestCase._get_node_of_graph_inout_fq(graph, 1, False)
        conv2_in_fq = TransformerTestCase._get_node_of_graph_inout_fq(graph, 2, True)
        conv2_out_fq = TransformerTestCase._get_node_of_graph_inout_fq(graph, 2, False)
        bn_in_fq = TransformerTestCase._get_node_of_graph_inout_fq(graph, 3, True)
        bn_out_fq = TransformerTestCase._get_node_of_graph_inout_fq(graph, 3, False)
        self.assertNotEqual(conv1_in_fq, None)
        self.assertNotEqual(conv1_out_fq, None)
        self.assertNotEqual(conv2_in_fq, None)
        self.assertNotEqual(conv2_out_fq, None)
        self.assertNotEqual(bn_in_fq, None)
        self.assertNotEqual(bn_out_fq, None)
        transformer1.apply(graph)
        transformer2.apply(graph)
        conv1_in_fq = TransformerTestCase._get_node_of_graph_inout_fq(graph, 1, True)
        conv1_out_fq = TransformerTestCase._get_node_of_graph_inout_fq(graph, 1, False)
        conv2_in_fq = TransformerTestCase._get_node_of_graph_inout_fq(graph, 2, True)
        conv2_out_fq = TransformerTestCase._get_node_of_graph_inout_fq(graph, 2, False)
        bn_in_fq = TransformerTestCase._get_node_of_graph_inout_fq(graph, 3, True)
        bn_out_fq = TransformerTestCase._get_node_of_graph_inout_fq(graph, 3, False)
        self.assertNotEqual(conv1_in_fq, None)
        self.assertEqual(conv1_out_fq, None)
        self.assertEqual(conv2_in_fq, None)
        self.assertNotEqual(conv2_out_fq, None)
        self.assertNotEqual(bn_in_fq, None)
        self.assertNotEqual(bn_out_fq, None)


if __name__ == '__main__':
    unittest.main()
