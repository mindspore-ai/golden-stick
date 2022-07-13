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
"""Quantize."""
import copy
from typing import Optional
from mindspore.rewrite import Node, NodeType
from mindspore.nn import Cell
from .net_policy import NetPolicy
from .layer_policy import LayerPolicy, layer_policy_key
from .transformer import Transformer
from .quantize_wrapper_cell import QuantizeWrapperCell
from ..comp_algo import CompAlgo
from ..net_transform import NetTransformer


class QuantizationAwareTraining(CompAlgo):
    """
    Derived class of `CompAlgo`. Base class of QAT-algorithm.
    """

    def __init__(self, config=None):
        super(QuantizationAwareTraining, self).__init__(config)
        if config is None:
            config = {}
        self._qat_policy = None
        self._custom_transforms = None
        self._custom_layer_policy_map = None
        self.net_transformer = None

    def _propagate_layer_policy(self, net_transformer: NetTransformer):
        """
        Set layer_policy for every layer according to custom_layer_policy_map, layer_policy_map and net_layer_policy in
        QuantAwareTraining. custom_layer_policy_map is in first priority, layer_policy_map is in second priority and
        net_layer_policy is in last priority.

        Args:
            nodes (List[Node]): nodes to be checked between which may find redundant fake-quantizer
        """

        # step1 apply net layer-policy first
        net_layer_policy: Optional[LayerPolicy] = self._qat_policy.get_net_layer_policy()
        if net_layer_policy:
            for node in net_transformer.nodes():
                node.set_attribute(layer_policy_key, copy.copy(net_layer_policy))
        # step2 then apply layer-policy map, override policy if need
        layer_policy_map = self._qat_policy.get_layer_policy_map()
        for node in net_transformer.nodes():
            if not isinstance(node, Node):
                continue
            layer_policy: LayerPolicy = self._custom_layer_policy_map.get(node.get_instance_type())
            if layer_policy is None:
                layer_policy = layer_policy_map.get(node.get_instance_type())
            if isinstance(layer_policy, LayerPolicy):
                new_layer_policy = copy.copy(layer_policy)
                new_layer_policy.set_input_number(len(node.get_inputs()))
                node.set_attribute(layer_policy_key, new_layer_policy)

        for node in net_transformer.nodes():
            if node.get_node_type() == NodeType.Tree:
                sub_net_trans = NetTransformer.create_from_tree_node(node)
                self._propagate_layer_policy(sub_net_trans)

    @staticmethod
    def _reduce_redundant_fake_quant(net_transformer: NetTransformer):
        """
        Reduce redundant fake-quantizer node between nodes. It usually occurs when pre-node inserted output
        fake-quantizer and post-node inserted input fake-quantizer at the same time.

        Args:
            nodes (List[Node]): nodes to be checked between which may find redundant fake-quantizer
        """

        for node in net_transformer.nodes():
            if not isinstance(node, Node):
                continue
            if node.get_node_type() == NodeType.Tree:
                sub_net_trans = NetTransformer.create_from_tree_node(node)
                QuantizationAwareTraining._reduce_redundant_fake_quant(sub_net_trans)
            cur_policy: LayerPolicy = node.get_attribute(layer_policy_key)
            # cur-node has no quant policy, so no fq will insert into its inputs
            if cur_policy is None:
                continue
            input_nodes = node.get_inputs()
            for i in range(0, len(input_nodes)):
                cur_in_quantizer = cur_policy.get_input_quantizer()
                # cur-node's input quantizer is None, so no fq will insert into its inputs
                if cur_in_quantizer is None:
                    continue
                input_node: Node = input_nodes[i]
                pre_policy: LayerPolicy = input_node.get_attribute(layer_policy_key)
                # pre-node has no quant policy, so no fq will insert into its outputs
                if pre_policy is None:
                    continue
                output_nodes_of_input_node = input_node.get_targets()
                for j in range(0, len(output_nodes_of_input_node)):
                    output_node_of_input_node = output_nodes_of_input_node[j]
                    if output_node_of_input_node is not node:
                        continue
                    pre_out_quantizer = pre_policy.get_output_quantizer()
                    # pre-node's output quantizer is None, so no fq will insert into its outputs
                    # or input fq of cur-node and output fq of pre-node are different
                    if type(pre_out_quantizer) is not type(cur_in_quantizer):
                        continue
                    # input fq of cur-node and output fq of pre-node are same type
                    # so we mark input fq of cur-node as redundant
                    cur_policy.set_input_not_insert_fq(i)

    def _apply_fuse_patterns(self, net_transformer: NetTransformer):
        """
        Apply transforms to corresponding layer.
        Replace layer with return value of wrap_cell of layer-policy by default.

        Args:
            net_transformer (NetTransformer): net_transformer is used to apply transforms to graph.
        """

        transformers: [Transformer] = self._qat_policy.get_transformers()
        if isinstance(self._custom_transforms, list):
            for transform in self._custom_transforms:
                if isinstance(transform, Transformer):
                    transformers.append(transform)
        for transformer in transformers:
            # Transformer always return False
            net_transformer.pattern_transform(transformer)

    @staticmethod
    def _replace_node(net_transformer: NetTransformer, target_node: Node, result_cell: Cell):
        if isinstance(result_cell, QuantizeWrapperCell):
            node_name = type(getattr(result_cell, '_handler')).__name__
        else:
            node_name = target_node.get_name()
        node = NetTransformer.create_node(result_cell, target_node.get_targets(), target_node.get_args(),
                                          target_node.get_kwargs(), node_name)
        net_transformer.replace(target_node, [node])

    @staticmethod
    def _apply_layer_policy(net_transformer: NetTransformer):
        """
        Apply layer-policy to corresponding layer.
        Replace layer with return value of wrap_cell of layer-policy by default.

        Args:
            net_transformer (NetTransformer): net_transformer is used to transform node according to layer policy.
        """
        nodes = list(net_transformer.nodes())
        for node in nodes:
            if node.get_node_type() == NodeType.Tree:
                sub_net_transformer = NetTransformer.create_from_tree_node(node)
                QuantizationAwareTraining._apply_layer_policy(sub_net_transformer)
                continue
            layer_policy = node.get_attribute(layer_policy_key)
            if isinstance(layer_policy, LayerPolicy):
                wrapped_cell = layer_policy.wrap_cell(node.get_instance())
                wrapped_cell.update_parameters_name(node.get_name() + '.')
                QuantizationAwareTraining._replace_node(net_transformer, node, wrapped_cell)

    def apply(self, network: Cell) -> Cell:
        """
        Apply QAT-Algorithm on `network`, use the following steps to make `network` available for quantization aware
        training:

        1. Fuse certain cells in `network` using pattern engine which is defined by net policy.
        2. Propagate layer policies defined through cells.
        3. Reduce redundant fake quantizers when they are redundant.
        4. Apply layer policies to convert normal cell to `QuantizeWrapperCell`.

        Args:
            network (Cell): Network to be quantized.

        Returns:
            Quantized network.
        """

        if not isinstance(self._qat_policy, NetPolicy):
            raise RuntimeError("Derived class should provide net policy")
        self.net_transformer = NetTransformer(network)
        self._apply_fuse_patterns(self.net_transformer)
        self._propagate_layer_policy(self.net_transformer)
        QuantizationAwareTraining._reduce_redundant_fake_quant(self.net_transformer)
        QuantizationAwareTraining._apply_layer_policy(self.net_transformer)
        return self.net_transformer.get_network()
