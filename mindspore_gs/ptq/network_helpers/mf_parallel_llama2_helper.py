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

from collections import OrderedDict
from mindspore import nn
from mindspore import set_context
from mindspore.communication.management import init
from mindformers import AutoModel
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.experimental.llama_demo import (ParallelLlamaForCausalLM, ParallelLlamaModel,
                                                 ParallelLlamaTransformerLayer, ParallelLlamaAttention,
                                                 ParallelLlamaMLPWithGate)
from mindformers.experimental.llama_demo.llama import LlamaRMSNorm as ParallelLlamaRMSNorm
from mindformers.experimental.distri_cores.tensor_parallel.layers import (
    ColumnParallelLinear, RowParallelLinear
)
from mindformers.experimental.distri_cores.create_comm import initialize_model_parallel
from mindspore_gs.common.utils import value_check
from mindspore_gs.ptq.processor import Processor
from .network_helper import DecoderGroupInfo, LayerInfo, LayerType
from .mf_net_helpers import MFLlama2Helper


class MFParallelLlama2Helper(MFLlama2Helper):
    """
    Derived from 'NetworkHelper', a utility class for the MindFormers framework ParrallelLlamaForCasualLM network.

    Args:
        config (MindFormerConfig): A MindFormerConfig object indicates the network configuration.

    Raises:
        TypeError: If input `config` is not an instance of `MindFormerConfig`.

    Examples:
        >>> from mindspore_gs.ptq.network_helpers.mf_parallel_llama2_helper import MFParallelLlama2Helper
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
        set_context(mode=self.mf_config.context.mode,
                    device_target=self.mf_config.context.device_target,
                    jit_config={"jit_level": "O0", "infer_boost": "on"})
        init()
        initialize_model_parallel(tp_size=self.mf_config.parallel_config.model_parallel, order='tp')
        network = AutoModel.from_config(self.mf_config, download_checkpoint=False)
        network.set_train(False)
        network.phase = 'predict'
        transform_and_load_checkpoint(self.mf_config, None, network, None)
        return network

    def get_decoder_layers(self, network: ParallelLlamaForCausalLM):
        """
        Get decoder layers from network.

        Args:
            network (ParallelLlamaForCausalLM): Network to get decoder layers.

        Returns:
            A list of tuple of (cell_name, `Cell`) as decoder layers and names.
        """
        value_check('network', network, ParallelLlamaForCausalLM)
        model: ParallelLlamaModel = network.model
        layers = []
        for i, layer in enumerate(model.layers):
            layers.append((f"root.model.layers.{i}", layer))
        return layers

    def get_linears(self, decoder_layer: ParallelLlamaTransformerLayer):
        """
        Get linear layers from decoder layer.

        Args:
            decoder_layer (ParallelLlamaTransformerLayer): Decoder layer to get linear layers.

        Returns:
            A list of `Cell` as linear layers of decoder layer.
        """
        value_check('decoder_layer', decoder_layer, ParallelLlamaTransformerLayer)
        attention: ParallelLlamaAttention = decoder_layer.attention
        if not isinstance(attention, ParallelLlamaAttention):
            raise RuntimeError(f"Only support ParallelLlamaAttention as attention but got {attention}")
        qkv_concat = (attention.attn_type == "self_attn")
        ffn: ParallelLlamaMLPWithGate = decoder_layer.feed_forward
        if not isinstance(ffn, ParallelLlamaMLPWithGate):
            raise RuntimeError(f"Only support ParallelLlamaMLPWithGate as FFN but got {ffn}")
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

    def get_page_attention_mgr(self, decoder_layer: ParallelLlamaTransformerLayer):
        """
        Get PageAttentionMgr layers from decoder layer.

        Args:
            decoder_layer (ParallelLlamaTransformerLayer): Decoder layer to get PageAttentionMgr layers.

        Returns:
            A list of `Cell` as PageAttentionMgr layers of decoder layer.
        """
        value_check('decoder_layer', decoder_layer, ParallelLlamaTransformerLayer)
        if not self.mf_config.model.model_config.use_past:
            raise ValueError("use_path need be True when doing kv cache quantizer.")
        attention: ParallelLlamaAttention = decoder_layer.attention
        if not isinstance(attention, ParallelLlamaAttention):
            raise RuntimeError(f"Only support ParallelLlamaAttention as attention but got {attention}")
        return attention.paged_attention_mgr

    @staticmethod
    def _ffn_analysis(decoder_info: DecoderGroupInfo):
        """_ffn_analysis"""
        ffn_info: LayerInfo = decoder_info.ffn
        ffn: ParallelLlamaMLPWithGate = ffn_info.layer
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
        attention: ParallelLlamaAttention = attention_info.layer
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
    def _decoder_analysis(decoder_name: str, decoder: ParallelLlamaTransformerLayer) -> DecoderGroupInfo:
        """_decoder_analysis"""
        info: DecoderGroupInfo = DecoderGroupInfo(decoder_name, decoder)
        for name, cell in decoder.name_cells().items():
            full_cell_name = f"{decoder_name}.{name}"
            if isinstance(cell, ParallelLlamaRMSNorm):
                if "attention" in name:
                    info.attention_norm = LayerInfo(full_cell_name, cell, LayerType.NORM_LAYER)
                if "ffn" in name:
                    info.ffn_norm = LayerInfo(full_cell_name, cell, LayerType.NORM_LAYER)
                continue
            if isinstance(cell, ParallelLlamaAttention):
                info.attention = LayerInfo(full_cell_name, cell, LayerType.UNKNOWN)
                MFParallelLlama2Helper._attention_analysis(info)
                continue
            if isinstance(cell, ParallelLlamaMLPWithGate):
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
                if isinstance(cell, ParallelLlamaTransformerLayer):
                    self.infos[cell_name] = self._fn(cell_name, cell)
                    return cell, True
                return cell, False

        value_check('network', network, ParallelLlamaForCausalLM)
        self._decoder_infos.clear()
        analyzer = Llama2Analyzer(MFParallelLlama2Helper._decoder_analysis)
        analyzer.process(network)
        self._decoder_infos = analyzer.infos
