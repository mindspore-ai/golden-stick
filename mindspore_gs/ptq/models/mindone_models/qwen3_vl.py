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
Qwen3VL Quantized Model Implementation
"""

import mindspore as ms

from mindspore_gs.ptq.models.mindone_models.mindone_model import MindOneModel, SmoothLayerInfo
from transformers.generation.configuration_utils import GenerationConfig


@MindOneModel.reg_model('qwen3_vl')
class Qwen3VL(MindOneModel):
    """Qwen3VL Quantized Model Implementation
    """
    def __init__(self, model_path):
        # pylint: disable=C0415
        from mindone.transformers import Qwen3VLForConditionalGeneration
        self.network = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            mindspore_dtype=ms.bfloat16,
            _attn_implementation="flash_attention_2",
            )
        self._original_sf_path = model_path
        self.num_attention_heads, self.num_key_value_heads = self._get_gqa_info(model_path)
        self.is_gqa = self.num_key_value_heads != self.num_attention_heads
        print("self.network:", self.network, flush=True)
        print("self.is_gqa", self.is_gqa, flush=True)

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
        # pylint: disable=C0415
        from mindone.transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLVisionBlock
        layers_info = []
        if isinstance(decoder_layer, Qwen3VLVisionBlock):
            # attention
            layers_info.append(
            SmoothLayerInfo(
                prev_layer=decoder_layer.norm1,
                curr_layer=[decoder_layer.attn.qkv],
                )
            )

            layers_info.append(
                SmoothLayerInfo(
                    prev_layer=decoder_layer.attn.qkv,
                    curr_layer=[decoder_layer.attn.proj],
                )
            )
            # mlp
            layers_info.append(
                SmoothLayerInfo(
                    prev_layer=decoder_layer.norm2,
                    curr_layer=[decoder_layer.mlp.linear_fc1],
                )
            )

            layers_info.append(
                SmoothLayerInfo(
                    prev_layer=decoder_layer.mlp.linear_fc1,
                    curr_layer=[decoder_layer.mlp.linear_fc2],
                )
            )
        elif isinstance(decoder_layer, Qwen3VLTextDecoderLayer):
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

        This method delegates to the underlying MindFormers network's
        generate method for inference.

        Args:
            inputs (Dict): Inputs for the model.
            max_new_tokens (int, optional): Maximum number of tokens to generate.
                Defaults to ``1``.

        Returns:
            Generated output from the model.
        """
        generation_config = GenerationConfig(use_cache=False)
        return self.network.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                     generation_config=generation_config)

    def _transformer_layers(self) -> tuple[type]:
        """Get the transformer layer types for quantization.

        This method returns the transformer layer types that should
        be targeted for quantization in MindFormers models.

        Returns:
            tuple[type]. Tuple containing TransformerLayer type.
        """
        # pylint: disable=C0415
        from mindone.transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLVisionBlock
        return [Qwen3VLVisionBlock, Qwen3VLTextDecoderLayer]
