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

This module provides the quantized implementation for Qwen3 models
using the MindFormers framework. It extends the MFModel base class
to provide specific functionality for Qwen3 model quantization.

Qwen3 is a series of large language models developed by Tongyi Lab,
featuring strong performance in various natural language processing tasks.
This quantized implementation enables efficient deployment of Qwen3
models while maintaining high accuracy.

The implementation supports various quantization techniques including:
- Weight quantization (INT8, INT4)
- Activation quantization
- KVCache quantization
- Mixed precision quantization

Example:
    >>> from mindspore_gs.ptq.models import AutoQuantForCausalLM
    >>>
    >>> # Automatically select and load Qwen3 model
    >>> model = AutoQuantForCausalLM.from_pretrained("/path/to/qwen3_config.yaml")
    >>>
    >>> # Calibrate and quantize the model
    >>> model.calibrate(ptq_config, layers_policy, calibration_dataset)
    >>>
    >>> # Save the quantized model
    >>> model.save_quantized("/path/to/save/qwen3_quant")
"""

from mindspore.nn.cell import Cell
from mindspore_gs.ptq.models.mindformers_models.mf_model import MFModel, MFModelEnableSafeTensors
from mindspore_gs.ptq.models.mindformers_models.param_processor import (QKVParamProcessor,
                                                                        FFNParamProcessor)
from mindspore_gs.ptq.utils import QuantType


@MFModel.reg_model('qwen3')
class QWen3(MFModelEnableSafeTensors):
    """Qwen3 Quantized Model Implementation

    This class provides the quantized implementation for Qwen3 models
    using the MindFormers framework. It extends MFModelEnableSafeTensors
    to provide specific functionality for Qwen3 model quantization.

    Key features of this implementation include:
    - Support for Qwen3-specific model architectures
    - Integration with MindFormers' distributed computing capabilities
    - Efficient parameter management for large-scale Qwen3 models
    - SafeTensors format support for model persistence
    - Quantization type description generation

    The class is automatically registered with the 'qwen3' alias, making
    it discoverable by the AutoQuantForCausalLM interface when a Qwen3
    model configuration is detected.

    Attributes:
        Inherits all attributes from MFModelEnableSafeTensors.

    Examples:
        >>> from mindspore_gs.ptq.models import AutoQuantForCausalLM
        >>>
        >>> # Automatically detect and load Qwen3 model
        >>> model = AutoQuantForCausalLM.from_pretrained("/path/to/qwen3_config.yaml")
        >>>
        >>> # The model will be an instance of QWen3 class
        >>> assert isinstance(model, QWen3)
    """

    def _process_params_dict_before_save(self, param_dict) -> tuple[dict, dict]:
        """Process parameter dictionary before saving.

        This method processes the parameter dictionary to handle
        Qwen3-specific parameter management requirements before
        saving the model.

        Args:
            param_dict (dict): Dictionary of model parameters.

        Returns:
            tuple[dict, dict]: Tuple containing the processed parameter
                dictionary and parameter name trace.
        """
        param_dict, param_name_trace = super()._process_params_dict_before_save(param_dict)

        # Apply QKV split
        qkv_processor = QKVParamProcessor(self.network)
        param_dict, qkv_trace = qkv_processor.split_param(param_dict)
        param_name_trace.update(qkv_trace)

        # Apply FFN split
        ffn_processor = FFNParamProcessor(self.network)
        param_dict, ffn_trace = ffn_processor.split_param(param_dict)
        param_name_trace.update(ffn_trace)

        return param_dict, param_name_trace

    def _get_quant_type(self, network):
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
        if not isinstance(network, Cell):
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
        process(network, 'network')
        return results

    def _get_description_file(self, network):
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
        results = self._get_quant_type(network)

        # Apply parameter name splitting to quantization type info to match split parameters
        qkv_processor = QKVParamProcessor(self.network)
        results = qkv_processor.split_name(results)

        ffn_processor = FFNParamProcessor(self.network)
        results = ffn_processor.split_name(results)

        hf_results = {}
        for key, value in results.items():
            hf_results[self._convert_param_names_to_hf(key)] = value

        desc_info = {}
        param_dict = self.parameters_dict()
        for key, _ in param_dict.items():
            if key in hf_results:
                desc_info[key] = hf_results[key]
            else:
                desc_info[key] = QuantType.FLOAT.value
        return desc_info

    @classmethod
    def _convert_param_names_to_hf(cls, param_name):
        """Convert mcore name to huggingface name.
        One parameter may correspond to multiple parameters in huggingface,
        so return a list of names."""
        rules = {
            "model.": "",
            "decoder.layers.": "model.layers.",
            ".self_attention.": ".self_attn.",
            "embedding.word_embeddings.": "model.embed_tokens.",
            "decoder.final_layernorm.": "model.norm.",
            "output_layer.": "lm_head.",
            ".pre_mlp_layernorm.": ".post_attention_layernorm.",
            ".q_layernorm.": ".q_norm.",
            ".k_layernorm.": ".k_norm.",
            ".linear_proj.": ".o_proj.",
            ".linear_q.": ".q_proj.",
            ".linear_k.": ".k_proj.",
            ".linear_v.": ".v_proj.",
            ".gating.": ".gate_proj.",
            ".hidden.": ".up_proj.",
            ".linear_fc2.": ".down_proj."
        }

        for old, new in rules.items():
            param_name = param_name.replace(old, new)

        return param_name
