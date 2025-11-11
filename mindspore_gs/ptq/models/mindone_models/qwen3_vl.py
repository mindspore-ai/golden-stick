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
Qwen3_vl Quantized Model Implementation
"""

import mindspore as ms
from mindspore.nn.cell import Cell

from mindspore_gs.ptq.models.mindone_models.mindone_model import MindOneModel
from mindspore_gs.ptq.utils import QuantType
from transformers.generation.configuration_utils import GenerationConfig


@MindOneModel.reg_model('qwen3_vl')
class Qwen3_vl(MindOneModel):
    """Qwen3_vl Quantized Model Implementation
    """
    def __init__(self, model_path):
        from mindone.transformers import Qwen3VLForConditionalGeneration
        self.network = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            mindspore_dtype=ms.bfloat16,
            _attn_implementation="flash_attention_2",
            )
        self._original_sf_path = model_path

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
        generation_config = GenerationConfig(max_new_tokens=max_new_tokens, use_cache=False)
        return self.network.generate(**inputs, do_sample=False, generation_config=generation_config)

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
        from mindone.transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer
        return [Qwen3VLTextDecoderLayer]

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
