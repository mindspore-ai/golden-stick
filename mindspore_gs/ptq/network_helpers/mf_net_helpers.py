# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Network helper for network from MindFormers."""
import os.path

from typing import Union, List

import math
from collections import OrderedDict
import numpy as np
import mindspore as ms
from mindspore import dtype as mstype
from mindspore.communication.management import init
from mindspore import Tensor, Model, nn
from mindformers import MindFormerConfig, build_context, AutoModel, build_parallel_config
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.models.build_tokenizer import build_tokenizer
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.modules import Linear
from mindformers.models.llama import LlamaForCausalLM, LlamaModel
from mindformers.models.llama.llama_transformer import LLamaDecodeLayer, LLamaAttention
from mindformers.models.llama.llama_layer import LlamaFeedForward, LlamaRMSNorm
from mindformers.experimental.infer.models.llama.llama import ParallelLlamaForCausalLM
from mindformers.experimental.infer.core.transformer import (ParallelTransformer,
                                                             ParallelTransformerLayer,
                                                             ParallelAttention,
                                                             ParallelMLP)
from mindformers.experimental.infer.core.norm import RMSNorm
from mindformers.experimental.infer.core.layers import ColumnParallelLinear, RowParallelLinear
from mindformers.experimental.distri_cores.create_comm import initialize_model_parallel
from mindspore_gs.common.utils import value_check
from mindspore_gs.ptq.processor import Processor
from .network_helper import NetworkHelper, DecoderGroupInfo, LayerInfo, LayerType


class MFNetworkHelper(NetworkHelper):
    """
    Network helper for network from MindFormers.

    Args:
        config (Union[str, MindFormerConfig]): MindFormerConfig or path of config file for network.

    Raises:
        TypeError: If input `config` is not an instance of `MindFormerConfig` neither a str.
        ValueError: If input `config` is not a valid file path when input `config` is a str.
    """
    def __init__(self, config: Union[str, MindFormerConfig] = None):
        value_check("config", config, (MindFormerConfig, str))
        if isinstance(config, MindFormerConfig):
            self.mf_config = config
        else:
            if not os.path.isfile(config):
                raise ValueError(f"Input `config`({config}) is not a valid file path.")
            self.mf_config = MindFormerConfig(config)
        build_parallel_config(self.mf_config)
        self.mf_config.model.model_config.parallel_config = self.mf_config.parallel_config

    def create_network(self):
        """
        Create network of type LlamaForCasualLM.

        Returns:
            Network of type LlamaForCasualLM.
        """
        build_context(self.mf_config)
        network = AutoModel.from_config(self.mf_config, download_checkpoint=False)
        network.set_train(False)
        network.phase = 'predict'
        model = Model(network)
        ckpt_path = self.mf_config.load_checkpoint
        if ckpt_path:
            input_ids = np.ones(shape=[self.get_spec('batch_size'), self.get_spec('seq_length')], dtype=np.int32)
            infer_data = network.prepare_inputs_for_predict_layout(input_ids)
            if os.path.isdir(ckpt_path):
                network.phase = 'infer_predict_layout'
                model.infer_predict_layout(*infer_data)
            transform_and_load_checkpoint(self.mf_config, model, network, infer_data, do_predict=True)
        ms.ms_memory_recycle()
        network.phase = 'predict'
        return network

    def get_spec(self, name: str):
        """
        Get network specific, such as batch_size, seq_length and so on.

        Args:
            name (str): Name of specific.

        Returns:
            Object as network specific.
        """
        value_check('name', name, str)
        if name == 'vocab_file':
            return self.mf_config.processor.tokenizer.vocab_file
        model_config = self.mf_config.model.model_config
        if hasattr(model_config, name):
            return getattr(model_config, name)
        raise KeyError(f"Can not find network specific: {name}.")

    # pylint: disable=arguments-differ
    def create_tokenizer(self):
        """
        Get network tokenizer.

        Returns:
            Object as network tokenizer.
        """
        return build_tokenizer(self.mf_config.processor.tokenizer)

    # pylint: disable=arguments-differ
    def generate(self, mf_network, input_ids: Union[np.ndarray, List[int], List[List[int]]],
                 max_new_tokens=None, **kwargs):
        """
        Invoke `network` and generate tokens.

        Args:
            mf_network (Cell): Network to generate tokens.
            input_ids (numpy.ndarray): Input tokens for generate.
            max_new_tokens (int): Max number of tokens to be generated, default ``1``.
            kwargs (Dict): Extensible parameter for subclasses.

        Returns:
            A list as generated tokens.
        """
        value_check('mf_network', mf_network, PreTrainedModel)
        value_check('input_ids', input_ids, (np.ndarray, List))
        if max_new_tokens:
            value_check('max_new_tokens', max_new_tokens, int)
        do_sample = self.get_spec('do_sample')
        seq = self.get_spec('seq_length')
        top_p = self.get_spec('top_p')
        top_k = self.get_spec('top_k')
        return mf_network.generate(input_ids, do_sample=do_sample, max_length=seq, max_new_tokens=max_new_tokens,
                                   top_p=top_p, top_k=top_k)

    def assemble_inputs(self, input_ids: np.ndarray, **kwargs):
        raise NotImplementedError


class MFLlama2Helper(MFNetworkHelper):
    """
    Derived from 'NetworkHelper', a utility class for the MindFormers framework Llama2 network.

    Args:
        config (MindFormerConfig): MindFormerConfig for network.

    Raises:
        TypeError: If input `config` is not an instance of `MindFormerConfig`.
    """
    def __init__(self, config: Union[str, MindFormerConfig] = None):
        super().__init__(config)
        self._decoder_infos = OrderedDict()

    @staticmethod
    def _get_slots(bs, block_size, prefill_max_len, is_prefill, block_tables, valid_length_example):
        """get_slots."""
        slot_mapping = []
        for i in range(bs):
            block_table = block_tables[i]
            if is_prefill:
                slots = [block_table[k // block_size] * block_size + k % block_size
                         for k in range(valid_length_example[i])]
                null_slot_idx = -1
                num_elements_to_add = prefill_max_len - valid_length_example[i]
                for _ in range(num_elements_to_add):
                    slots.append(null_slot_idx)
            else:
                current_idx = valid_length_example[i] - 1
                slots = [block_table[current_idx // block_size] * block_size + current_idx % block_size]
            slot_mapping = slot_mapping + slots

        return np.array(slot_mapping, copy=False, dtype=np.int32)

    @staticmethod
    def _get_pa_inputs(bs, seq, block_size, valid_length):
        """_get_pa_inputs"""
        valid_length_each_example = np.array([valid_length])
        prefill_max_len = max(valid_length_each_example)
        required_block_num = math.ceil(seq / block_size)
        block_tables = np.arange(required_block_num, dtype=np.int32).reshape(bs, -1)
        slot_mapping = MFLlama2Helper._get_slots(bs, block_size, prefill_max_len, True, block_tables,
                                                 valid_length_each_example)
        block_tables = Tensor(block_tables, mstype.int32)
        slot_mapping = Tensor(slot_mapping, mstype.int32)
        return block_tables, slot_mapping

    def assemble_inputs(self, input_ids: np.ndarray, **kwargs):
        """
        Assemble network inputs for predict from input tokens in numpy ndarray format.

        Args:
            input_ids (numpy.ndarray): Input tokens.
            kwargs (Dict): Extensible parameter for subclasses.

        Returns:
            A list of `mindspore.Tensor` as inputs of network predict.
        """
        value_check('input_ids', input_ids, np.ndarray)
        shape = input_ids.shape
        if len(shape) > 2:
            raise ValueError(f"Only support two-dimension(bs, seq_length) input_ids, got: {shape}.")
        bs = self.mf_config.model.model_config.batch_size
        seq = self.mf_config.model.model_config.seq_length
        if shape[0] > bs or shape[1] > seq:
            raise ValueError(f"Input input_ids shape({shape}) out of max shape({bs}, {seq}).")
        pad_token_id = self.mf_config.model.model_config.pad_token_id
        use_past = self.mf_config.model.model_config.use_past
        input_ids = np.pad(input_ids, ((0, bs - shape[0]), (0, seq - shape[1])), 'constant',
                           constant_values=pad_token_id)
        t_input_ids = Tensor(input_ids)
        if not use_past:
            return t_input_ids, None, None, None, None, None, None, None, None, None, None, None
        block_size = self.mf_config.model.model_config.block_size

        block_tables, slot_mapping = MFLlama2Helper._get_pa_inputs(bs, seq, block_size, shape[1])
        return t_input_ids, None, None, None, None, None, None, None, None, None, block_tables, slot_mapping

    def get_decoder_layers(self, network: LlamaForCausalLM):
        """
        Get decoder layers from network.

        Args:
            network (LlamaForCausalLM): Network to get decoder layers.

        Returns:
            A list of tuples (cell_name, `Cell`) as decoder layers of network.
        """
        value_check('network', network, LlamaForCausalLM)
        model: LlamaModel = network.model
        layers = []
        for i, layer in enumerate(model.layers):
            layers.append((f"root.model.layers.{i}", layer))
        return layers

    def get_linears(self, decoder_layer: LLamaDecodeLayer):
        """
        Get linears from decoder_layer.

        Args:
            decoder_layer (LLamaDecodeLayer): Decoder_layer to get linears.

        Returns:
            A list of `Cell` as linears of decoder layers.
        """
        value_check('decoder_layer', decoder_layer, LLamaDecodeLayer)
        attention: LLamaAttention = decoder_layer.attention
        if not isinstance(attention, LLamaAttention):
            raise RuntimeError(f"Only support LLamaAttention as attention but got {attention}")
        qkv_concat = attention.qkv_concat
        ffn: LlamaFeedForward = decoder_layer.feed_forward
        if not isinstance(ffn, LlamaFeedForward):
            raise RuntimeError(f"Only support LlamaFeedForward as FFN but got {ffn}")
        ffn_concat = ffn.ffn_concat
        linears = []
        if qkv_concat:
            linears.extend([attention.w_qkv, attention.wo])
        else:
            linears.extend([attention.wq, attention.wk, attention.wv, attention.wo])

        if ffn_concat:
            linears.extend([ffn.w_gate_hidden, ffn.w2])
        else:
            linears.extend([ffn.w1, ffn.w3, ffn.w2])
        return qkv_concat, ffn_concat, linears

    def get_page_attention_mgr(self, decoder_layer: LLamaDecodeLayer):
        """
        Get PageAttentionMgr layers from decoder layer.

        Args:
            decoder_layer (LLamaDecodeLayer): Decoder layer to get PageAttentionMgr layers.

        Returns:
            A list of `Cell` as PageAttentionMgr layers of decoder layer.
        """
        value_check('decoder_layer', decoder_layer, LLamaDecodeLayer)
        if not self.mf_config.model.model_config.use_past:
            raise ValueError("use_path need be True when doing kv cache quantizer.")
        attention: LLamaAttention = decoder_layer.attention
        if not isinstance(attention, LLamaAttention):
            raise RuntimeError(f"Only support LLamaAttention as attention but got {attention}")
        return attention.infer_attention.paged_attention_mgr

    @staticmethod
    def _ffn_analysis(decoder_info: DecoderGroupInfo):
        """_ffn_analysis"""
        ffn_info: LayerInfo = decoder_info.ffn
        ffn: LlamaFeedForward = ffn_info.layer
        decoder_info.ffn_concat = ffn.ffn_concat
        for name, cell in ffn.name_cells().items():
            full_cell_name = f"{ffn_info.name}.{name}"
            if ffn.ffn_concat:
                if isinstance(cell, Linear):
                    if "gate_hidden" in name:
                        decoder_info.gate_hidden_mm = LayerInfo(full_cell_name, cell, LayerType.CONCAT_LINEAR_LAYER)
                        continue
                    if "w2" in name:
                        decoder_info.w2_mm = LayerInfo(full_cell_name, cell, LayerType.LINEAR_LAYER)
                        continue
            else:
                if isinstance(cell, Linear):
                    if "w1" in name:
                        decoder_info.hidden_mm = LayerInfo(full_cell_name, cell, LayerType.LINEAR_LAYER)
                        continue
                    if "w3" in name:
                        decoder_info.gate_mm = LayerInfo(full_cell_name, cell, LayerType.LINEAR_LAYER)
                        continue
                    if "w2" in name:
                        decoder_info.w2_mm = LayerInfo(full_cell_name, cell, LayerType.LINEAR_LAYER)
                        continue

    @staticmethod
    def _attention_analysis(decoder_info: DecoderGroupInfo):
        """_attention_analysis"""
        attention_info: LayerInfo = decoder_info.attention
        attention: LLamaAttention = attention_info.layer
        decoder_info.qkv_concat = attention.qkv_concat
        for name, cell in attention.name_cells().items():
            full_cell_name = f"{attention_info.name}.{name}"
            if attention.qkv_concat:
                if isinstance(cell, Linear):
                    if "qkv" in name:
                        decoder_info.qkv_mm = LayerInfo(full_cell_name, cell, LayerType.CONCAT_LINEAR_LAYER)
                        continue
                    if "wo" in name:
                        decoder_info.o_mm = LayerInfo(full_cell_name, cell, LayerType.LINEAR_LAYER)
                        continue
            else:
                if isinstance(cell, Linear):
                    if "wq" in name:
                        decoder_info.q_mm = LayerInfo(full_cell_name, cell, LayerType.LINEAR_LAYER)
                        continue
                    if "wk" in name:
                        decoder_info.k_mm = LayerInfo(full_cell_name, cell, LayerType.LINEAR_LAYER)
                        continue
                    if "wv" in name:
                        decoder_info.v_mm = LayerInfo(full_cell_name, cell, LayerType.LINEAR_LAYER)
                        continue
                    if "wo" in name:
                        decoder_info.o_mm = LayerInfo(full_cell_name, cell, LayerType.LINEAR_LAYER)
                        continue

    @staticmethod
    def _decoder_analysis(decoder_name: str, decoder: LLamaDecodeLayer) -> DecoderGroupInfo:
        """_decoder_analysis"""
        info: DecoderGroupInfo = DecoderGroupInfo(decoder_name, decoder)
        for name, cell in decoder.name_cells().items():
            full_cell_name = f"{decoder_name}.{name}"
            if isinstance(cell, LlamaRMSNorm):
                if "attention" in name:
                    info.attention_norm = LayerInfo(full_cell_name, cell, LayerType.NORM_LAYER)
                if "ffn" in name:
                    info.ffn_norm = LayerInfo(full_cell_name, cell, LayerType.NORM_LAYER)
                continue
            if isinstance(cell, LLamaAttention):
                info.attention = LayerInfo(full_cell_name, cell, LayerType.UNKNOWN)
                MFLlama2Helper._attention_analysis(info)
                continue
            if isinstance(cell, LlamaFeedForward):
                info.ffn = LayerInfo(full_cell_name, cell, LayerType.UNKNOWN)
                MFLlama2Helper._ffn_analysis(info)
        return info

    def analysis_decoder_groups(self, network):
        """
        Analyze decoder groups information of network.

        Args:
            network (Cell): network to analyze decoder groups information.
        """

        class Llama2Analyzer(Processor):
            """A network iterator for applying algorithm on network."""
            def __init__(self, process_fn):
                self._fn = process_fn
                self.infos: OrderedDict[str, nn.Cell] = OrderedDict()

            def process_cell(self, cell_name: str, cell: nn.Cell):
                if not isinstance(cell, nn.Cell):
                    return cell, True
                if isinstance(cell, LLamaDecodeLayer):
                    self.infos[cell_name] = self._fn(cell_name, cell)
                    return cell, True
                return cell, False
        value_check('network', network, LlamaForCausalLM)
        self._decoder_infos.clear()
        analyzer = Llama2Analyzer(MFLlama2Helper._decoder_analysis)
        analyzer.process(network)
        self._decoder_infos = analyzer.infos

    @staticmethod
    def _get_pre_layer_for_attn(linear_name, decoder_info):
        """_get_pre_layer_for_attn"""
        # attn.qkv
        if decoder_info.qkv_concat:
            if decoder_info.qkv_mm.name == linear_name:
                return decoder_info.attention_norm
        # attn.q attn.k attn.v
        if (decoder_info.q_mm and linear_name == decoder_info.q_mm.name) or \
           (decoder_info.k_mm and linear_name == decoder_info.k_mm.name) or \
           (decoder_info.v_mm and linear_name == decoder_info.v_mm.name):
            return None
        # attn.o
        if linear_name == decoder_info.o_mm.name:
            return decoder_info.qkv_mm if decoder_info.qkv_concat else decoder_info.v_mm
        return None

    @staticmethod
    def _get_pre_layer_for_ffn(linear_name, decoder_info):
        """_get_pre_layer_for_ffn"""
        # ffn.gate_hidden
        if decoder_info.ffn_concat:
            if decoder_info.gate_hidden_mm.name == linear_name:
                return decoder_info.ffn_norm
        # ffn.gate ffn.hidden
        if (decoder_info.gate_mm and linear_name == decoder_info.gate_mm.name) or \
           (decoder_info.hidden_mm and linear_name == decoder_info.hidden_mm.name):
            return None
        # ffn.w2
        if linear_name == decoder_info.w2_mm.name:
            return decoder_info.gate_hidden_mm if decoder_info.ffn_concat else decoder_info.gate_mm
        return None

    def get_pre_layer(self, linear_name: str):
        """
        Get pre layer information from current linear_name.

        Args:
            linear_name (str): linear layer name.

        Returns:
            A dict of pre layer information which include pre layer name, layer and type.
        """
        value_check('linear_name', linear_name, str)
        splits = linear_name.split('.')
        decoder_info: DecoderGroupInfo = self._decoder_infos.get(f'root.model.layers.{splits[3]}')
        if not decoder_info:
            raise RuntimeError(f"Can not find decoder layer for Linear {linear_name}.")
        pre_layer = MFLlama2Helper._get_pre_layer_for_attn(linear_name, decoder_info)
        if pre_layer:
            return pre_layer
        return MFLlama2Helper._get_pre_layer_for_ffn(linear_name, decoder_info)


class MFParallelLlama2Helper(MFLlama2Helper):
    """
    Derived from 'NetworkHelper', a utility class for the MindFormers framework ParrallelLlamaForCasualLM network.

    Args:
        config (MindFormerConfig): A MindFormerConfig object indicates the network configuration.

    Raises:
        TypeError: If input `config` is not an instance of `MindFormerConfig`.

    Examples:
        >>> from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFParallelLlama2Helper
        >>> from mindformers.tools.register.config import MindFormerConfig
        >>> mf_yaml_config_file = "/path/to/mf_yaml_config_file"
        >>> mfconfig = MindFormerConfig(mf_yaml_config_file)
        >>> helper = MFParallelLlama2Helper(mfconfig)
        >>> network = helper.create_network()
        >>> decoder_layers = helper.get_decoder_layers(network)
        >>> linears = helper.get_linears(decoder_layers[0][1])
        >>> page_attention_mgrs = helper.get_page_attention_mgr(decoder_layers[0][1])
        >>> helper.analysis_decoder_groups(network)
    """
    def create_network(self):
        """
        Create network of type ParallelLlamaForCasualLM.

        Returns:
            Network of type ParallelLlamaForCasualLM.
        """
        ms.set_context(mode=self.mf_config.context.mode, device_target=self.mf_config.context.device_target,
                       jit_config={"jit_level": "O0", "infer_boost": "on"})
        init()
        initialize_model_parallel(self.mf_config.parallel_config.model_parallel, order='tp')
        network = AutoModel.from_config(self.mf_config, download_checkpoint=False)
        network.set_train(False)
        network.phase = 'predict'
        ckpt_path = self.mf_config.load_checkpoint
        if ckpt_path:
            transform_and_load_checkpoint(self.mf_config, None, network, None)
        return network

    def get_decoder_layers(self, network: ParallelLlamaForCausalLM):
        """
        Get decoder layers from network.

        Args:
            network (ParallelLlamaForCausalLM): Network to get decoder layers.

        Returns:
            A list of tuples (cell_name, `Cell`) as decoder layers and names.
        """
        value_check('network', network, ParallelLlamaForCausalLM)
        model: ParallelTransformer = network.model
        layers = []
        for i, layer in enumerate(model.layers):
            layers.append((f"root.model.layers.{i}", layer))
        return layers

    def get_linears(self, decoder_layer: ParallelTransformerLayer):
        """
        Get linear layers from decoder layer.

        Args:
            decoder_layer (ParallelTransformerLayer): Decoder layer to get linear layers.

        Returns:
            A list of `Cell` as linear layers of decoder layer.
        """
        value_check('decoder_layer', decoder_layer, ParallelTransformerLayer)
        attention: ParallelAttention = decoder_layer.attention
        if not isinstance(attention, ParallelAttention):
            raise RuntimeError(f"Only support ParallelAttention as attention but got {attention}")
        qkv_concat = (attention.attn_type == "self_attn")
        ffn: ParallelMLP = decoder_layer.feed_forward
        if not isinstance(ffn, ParallelMLP):
            raise RuntimeError(f"Only support ParallelMLP as FFN but got {ffn}")
        ffn_concat = True
        linears = []
        if qkv_concat:
            linears.extend([attention.w_qkv, attention.wo])
        else:
            linears.extend([attention.wq, attention.wk, attention.wv, attention.wo])

        if ffn_concat:
            linears.extend([ffn.w_gate_hidden, ffn.w2])
        else:
            linears.extend([ffn.w1, ffn.w3, ffn.w2])
        return qkv_concat, ffn_concat, linears

    def get_page_attention_mgr(self, decoder_layer: ParallelTransformerLayer):
        """
        Get PageAttentionMgr layers from decoder layer.

        Args:
            decoder_layer (ParallelTransformerLayer): Decoder layer to get PageAttentionMgr layers.

        Returns:
            A list of `Cell` as PageAttentionMgr layers of decoder layer.
        """
        value_check('decoder_layer', decoder_layer, ParallelTransformerLayer)
        if not self.mf_config.model.model_config.use_past:
            raise ValueError("use_path need be True when doing kv cache quantizer.")
        attention: ParallelAttention = decoder_layer.attention
        if not isinstance(attention, ParallelAttention):
            raise RuntimeError(f"Only support ParallelAttention as attention but got {attention}")
        return attention.paged_attention_mgr

    @staticmethod
    def _ffn_analysis(decoder_info: DecoderGroupInfo):
        """_ffn_analysis"""
        ffn_info: LayerInfo = decoder_info.ffn
        ffn: ParallelMLP = ffn_info.layer
        decoder_info.ffn_concat = True
        for name, cell in ffn.name_cells().items():
            full_cell_name = f"{ffn_info.name}.{name}"
            if decoder_info.ffn_concat:
                if isinstance(cell, (ColumnParallelLinear, RowParallelLinear)):
                    if "gate_hidden" in name:
                        decoder_info.gate_hidden_mm = LayerInfo(full_cell_name, cell, LayerType.CONCAT_LINEAR_LAYER)
                        continue
                    if "w2" in name:
                        decoder_info.w2_mm = LayerInfo(full_cell_name, cell, LayerType.LINEAR_LAYER)
                        continue
            else:
                if isinstance(cell, (ColumnParallelLinear, RowParallelLinear)):
                    if "w1" in name:
                        decoder_info.hidden_mm = LayerInfo(full_cell_name, cell, LayerType.LINEAR_LAYER)
                        continue
                    if "w3" in name:
                        decoder_info.gate_mm = LayerInfo(full_cell_name, cell, LayerType.LINEAR_LAYER)
                        continue
                    if "w2" in name:
                        decoder_info.w2_mm = LayerInfo(full_cell_name, cell, LayerType.LINEAR_LAYER)
                        continue

    @staticmethod
    def _attention_analysis(decoder_info: DecoderGroupInfo):
        """_attention_analysis"""
        attention_info: LayerInfo = decoder_info.attention
        attention: ParallelAttention = attention_info.layer
        decoder_info.qkv_concat = (attention.attn_type == "self_attn")
        for name, cell in attention.name_cells().items():
            full_cell_name = f"{attention_info.name}.{name}"
            if decoder_info.qkv_concat:
                if isinstance(cell, (ColumnParallelLinear, RowParallelLinear)):
                    if "qkv" in name:
                        decoder_info.qkv_mm = LayerInfo(full_cell_name, cell, LayerType.CONCAT_LINEAR_LAYER)
                        continue
                    if "wo" in name:
                        decoder_info.o_mm = LayerInfo(full_cell_name, cell, LayerType.LINEAR_LAYER)
                        continue
            else:
                if isinstance(cell, (ColumnParallelLinear, RowParallelLinear)):
                    if "wq" in name:
                        decoder_info.q_mm = LayerInfo(full_cell_name, cell, LayerType.LINEAR_LAYER)
                        continue
                    if "wk" in name:
                        decoder_info.k_mm = LayerInfo(full_cell_name, cell, LayerType.LINEAR_LAYER)
                        continue
                    if "wv" in name:
                        decoder_info.v_mm = LayerInfo(full_cell_name, cell, LayerType.LINEAR_LAYER)
                        continue
                    if "wo" in name:
                        decoder_info.o_mm = LayerInfo(full_cell_name, cell, LayerType.LINEAR_LAYER)
                        continue

    @staticmethod
    def _decoder_analysis(decoder_name: str, decoder: ParallelTransformerLayer) -> DecoderGroupInfo:
        """_decoder_analysis"""
        info: DecoderGroupInfo = DecoderGroupInfo(decoder_name, decoder)
        for name, cell in decoder.name_cells().items():
            full_cell_name = f"{decoder_name}.{name}"
            if isinstance(cell, RMSNorm):
                if "attention" in name:
                    info.attention_norm = LayerInfo(full_cell_name, cell, LayerType.NORM_LAYER)
                if "ffn" in name:
                    info.ffn_norm = LayerInfo(full_cell_name, cell, LayerType.NORM_LAYER)
                continue
            if isinstance(cell, ParallelAttention):
                info.attention = LayerInfo(full_cell_name, cell, LayerType.UNKNOWN)
                MFParallelLlama2Helper._attention_analysis(info)
                continue
            if isinstance(cell, ParallelMLP):
                info.ffn = LayerInfo(full_cell_name, cell, LayerType.UNKNOWN)
                MFParallelLlama2Helper._ffn_analysis(info)
        return info

    def analysis_decoder_groups(self, network):
        """
        Analyze decoder groups information of network.

        Args:
            network (ParallelLlamaForCausalLM): network to analyze decoder groups information.
        """
        class Llama2Analyzer(Processor):
            """A network iterator for applying algorithm on network."""
            def __init__(self, process_fn):
                self._fn = process_fn
                self.infos: OrderedDict[str, nn.Cell] = OrderedDict()

            def process_cell(self, cell_name: str, cell: nn.Cell):
                if not isinstance(cell, nn.Cell):
                    return cell, True
                if isinstance(cell, ParallelTransformerLayer):
                    self.infos[cell_name] = self._fn(cell_name, cell)
                    return cell, True
                return cell, False

        value_check('network', network, ParallelLlamaForCausalLM)
        self._decoder_infos.clear()
        analyzer = Llama2Analyzer(MFParallelLlama2Helper._decoder_analysis)
        analyzer.process(network)
        self._decoder_infos = analyzer.infos
