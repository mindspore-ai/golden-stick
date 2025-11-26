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
GLM4v Quantized Model Implementation
"""

import os
import json

import mindspore as ms
from mindspore.nn.cell import Cell
from mindone.transformers import Glm4vForConditionalGeneration

from mindspore_gs.ptq.models.mindone_models.mindone_model import MindOneModel, SmoothLayerInfo
from mindspore_gs.ptq.utils import QuantType


@MindOneModel.reg_model('glm4v')
class GLM4v(MindOneModel):
    """GLM4v Quantized Model Implementation
    """
    def __init__(self, model_path):
        self.network = Glm4vForConditionalGeneration.from_pretrained(
            model_path,
            mindspore_dtype=ms.bfloat16,
            _attn_implementation="flash_attention_2",
            )
        self._original_sf_path = model_path
        self.num_attention_heads, self.num_key_value_heads = self._get_gqa_info(model_path)
        self.is_gqa = self.num_key_value_heads != self.num_attention_heads

    def _get_gqa_info(self, model_path):
        """Get GQA information from config file."""
        config_path = os.path.join(model_path, 'config.json')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            text_config = config.get('text_config', None)
            if text_config is None:
                text_config = config
            num_attention_heads = text_config.get('num_attention_heads', None)
            num_key_value_heads = text_config.get('num_key_value_heads', None)
            if num_attention_heads is None or num_key_value_heads is None:
                raise ValueError(f"num_attention_heads or num_key_value_heads is not found in {config_path}.")
            return num_attention_heads, num_key_value_heads
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Config file not found at {config_path}.") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON from {config_path}.") from e

    def get_layers_for_smooth(self, decoder_layer):
        """Get layers for search.
        This method returns a list of layers that should be used for search.
        
        Args:
            layer (Cell): The layer to get layers for search.
        
        Returns:
            list[SmoothLayerInfo]. List of layers for search. Each layer is a SmoothLayerInfo with the following keys:
                - prev_layer (Cell): The layer before the current layer.
                - curr_layer (List[Cell]): The current layer.
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
                curr_layer=[decoder_layer.mlp.gate_up_proj],
            )
        )

        layers_info.append(
            SmoothLayerInfo(
                prev_layer=decoder_layer.mlp.gate_up_proj,
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
        return self.network.generate(**inputs,
                                     max_new_tokens=max_new_tokens,
                                     do_sample=False,
                                     use_cache=False)

    def _network(self):
        """Get the underlying network instance.

        Returns:
            The underlying MindFormers network instance.
        """
        return self.network

    def _transformer_layers(self) -> tuple[type]:
        """Get the transformer layer types for quantization.

        This method returns the transformer layer types that should
        be targeted for quantization in MindFormers models.

        Returns:
            tuple[type]. Tuple containing TransformerLayer type.
        """
        from mindone.transformers.models.glm4v.modeling_glm4v import Glm4vTextDecoderLayer
        return [Glm4vTextDecoderLayer]

    def _get_quant_type(self):
        """Get quantization type information for network parameters.

        This method analyzes the network to determine the quantization
        type for each parameter, such as W8A8 or W4A8_DYNAMIC.

        Args:
            network (Cell): The network to analyze for quantization types.

        Returns:
            dict. Dictionary mapping parameter names to their quantization types.

        Raises:
            TypeError: If the input network is not a Cell instance.
        """
        if not isinstance(self.network, Cell):
            raise TypeError(f"Input network should be a Cell, but got: {type(Cell)}.")
        results = {}
        def process(root: Cell, name_prefix):
            """Iterate the whole network and call callback function `process_cell`."""
            if root is None:
                return
            for name, cell in root.name_cells().items():
                full_cell_name = f"{name_prefix}.{name}"
                if not hasattr(cell, "quant_type_dict"):
                    process(cell, full_cell_name)
                    continue
                info = cell.quant_type_dict()
                results.update(info)
        process(self.network, 'network')
        return results

    # pylint: disable=W0221
    def get_description_file(self):
        """Obtain the description of quantization type for network parameters.

        This method generates a comprehensive description of the
        quantization type for each parameter in each layer of the network.
        The description includes information such as W8A8 or W4A8_DYNAMIC
        for each parameter.

        Args:
            network (Cell): The network to analyze for quantization descriptions.

        Returns:
            dict. Dictionary mapping parameter names to their quantization
                type descriptions.
        """
        results = self._get_quant_type()
        param_dict = self.network.parameters_dict()

        desc_info = {}
        for key in param_dict:
            if key in results:
                desc_info[key] = results[key]
            else:
                desc_info[key] = QuantType.FLOAT.value
        return desc_info
