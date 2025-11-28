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

"""
Qwen3 Quantized Model Implementation
"""

import mindspore as ms
from mindone.transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from mindspore_gs.ptq.models.mindone_models.mindone_model import MindOneModel, SmoothLayerInfo


@MindOneModel.reg_model('qwen3')
class Qwen3(MindOneModel):
    """Qwen3 Quantized Model Implementation"""
    def __init__(self, model_path):
        self.network = Qwen3ForCausalLM.from_pretrained(
            model_path,
            mindspore_dtype=ms.bfloat16,
            _attn_implementation="flash_attention_2",
            use_cache=False,
            )
        self._original_sf_path = model_path
        self.num_attention_heads, self.num_key_value_heads = self._get_gqa_info(model_path)
        self.is_gqa = self.num_key_value_heads != self.num_attention_heads

    def get_layers_for_smooth(self, decoder_layer):
        """Get layers for search.
        This method returns a list of layers that should be used for search.
        
        Args:
            layer (Cell): The layer to get layers for search.
        
        Returns:
            list[dict]. List of layers for search. Each layer is a dictionary with the following keys:
                - prev_layer (Cell): The layer before the current layer.
                - curr_layer (Cell): The current layer.
        """
        layers_info = []
        # attention
        layers_info.append(
        SmoothLayerInfo(
            prev_layer=decoder_layer.input_layernorm,
            curr_layer=[decoder_layer.self_attn.q_proj,
                        decoder_layer.self_attn.k_proj,
                        decoder_layer.self_attn.v_proj],
            )
        )

        layers_info.append(
            SmoothLayerInfo(
                prev_layer=decoder_layer.self_attn.v_proj,
                curr_layer=[decoder_layer.self_attn.o_proj],
            )
        )
        # mlp
        layers_info.append(
            SmoothLayerInfo(
                prev_layer=decoder_layer.post_attention_layernorm,
                curr_layer=[decoder_layer.mlp.gate_proj,
                            decoder_layer.mlp.up_proj],
            )
        )

        layers_info.append(
            SmoothLayerInfo(
                prev_layer=decoder_layer.mlp.up_proj,
                curr_layer=[decoder_layer.mlp.down_proj],
            )
        )
        return layers_info

    # pylint: disable=W0237
    def forward(self, inputs, max_new_tokens=1):
        """Perform forward pass through the model.

        This method delegates to the underlying Mindone network's
        generate method for inference.

        Args:
            inputs (Dict): Inputs for the model.
            max_new_tokens (int, optional): Maximum number of tokens to generate.
                Defaults to ``1``.

        Returns:
            Generated output from the model.
        """
        return self.network.generate(**inputs,
                                     max_new_tokens=max_new_tokens,
                                     use_cache=False,
                                     do_sample=False)

    def _network(self):
        """Get the underlying network instance.

        Returns:
            The underlying MindOne network instance.
        """
        return self.network

    def _transformer_layers(self) -> tuple[type]:
        """Get the transformer layer types for quantization.

        This method returns the transformer layer types that should
        be targeted for quantization in MindOne models.

        Returns:
            tuple[type]. Tuple containing TransformerLayer type.
        """
        from mindone.transformers.models.qwen3.modeling_qwen3  import Qwen3DecoderLayer
        return [Qwen3DecoderLayer]
