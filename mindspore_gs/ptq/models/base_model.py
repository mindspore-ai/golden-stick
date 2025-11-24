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
Base Class for Causal Language Model Quantization

This module defines the base class and interface for quantizing causal language models.
It establishes a standardized API that all quantized model implementations must follow,
ensuring consistency and interoperability across different model frameworks and quantization
algorithms.

The base class implements a registry mechanism that allows different model frameworks
to register their specific implementations, enabling automatic model detection and selection
through the AutoQuantForCausalLM interface.

Key features of this base class include:
- Standardized quantization workflow methods
- Registry mechanism for model framework integration
- Abstract interface enforcement for consistent implementation

Examples:
    >>> from mindspore_gs.ptq.models.base_model import BaseQuantForCausalLM
    >>>
    >>> # A custom model implementation
    >>> class CustomQuantModel(BaseQuantForCausalLM):
    >>>     @classmethod
    >>>     def from_pretrained(cls, **kwargs):
    >>>         # Custom model loading logic
    >>>         pass
    >>>
    >>>     def forward(self, input_ids, max_new_tokens=1):
    >>>         # Custom forward pass logic
    >>>         pass
"""


from mindspore_gs.common import BackendTarget


class BaseQuantForCausalLM:
    """Base Class for Causal Language Model Quantization

    This is the base class that defines the standard interface for all
    quantized causal language model implementations. It provides the
    fundamental structure and required methods that must be implemented
    by all derived classes.

    The class implements a registry mechanism that allows different model
    frameworks to register their implementations. This enables the automatic
    model detection and selection functionality provided by AutoQuantForCausalLM.

    Examples:
        >>> from mindspore_gs.ptq.models.base_model import BaseQuantForCausalLM
        >>>
        >>> # A custom model implementation
        >>> class MyCustomQuantModel(BaseQuantForCausalLM):
        >>>     pass
    """
    @classmethod
    def from_pretrained(cls, **kwargs):
        """Create a model instance from pretrained weights.

        This is an abstract method that must be implemented by derived classes.
        It should handle loading pretrained model weights and configuration.

        Args:
            **kwargs (dict): Arbitrary keyword arguments for model creation.

        Returns:
            BaseQuantForCausalLM. An instance of the quantized model.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def forward(self, input_ids, max_new_tokens=1):
        """Perform forward pass through the model.

        This is an abstract method that must be implemented by derived classes.
        It should handle the forward pass logic for model inference.

        Args:
            input_ids (Tensor): Input token IDs for the model.
            max_new_tokens (int, optional): Maximum number of tokens to generate.
                Defaults to ``1``.

        Returns:
            Forward pass results.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def calibrate(self, ptq_config, layers_policy, datasets, **kwargs):
        """Calibrate and quantize the model.

        This is an abstract method that must be implemented by derived classes.
        It should handle the model calibration process using calibration datasets
        and apply quantization according to the provided configuration.

        Args:
            ptq_config (PTQConfig): Configuration for post-training quantization.
            layers_policy (dict): Policy for different layer quantization strategies.
            datasets (Dataset): Calibration dataset for quantization.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def save_quantized(self, save_path, backend=BackendTarget.ASCEND):
        """Save the quantized model to disk.

        This is an abstract method that must be implemented by derived classes.
        It should handle saving the quantized model weights and configuration.

        Args:
            save_path (str): Path where the quantized model should be saved.
            backend (BackendTarget, optional): Target backend for the saved model.
                Defaults to ``BackendTarget.ASCEND``.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def fake_quant(self, ptq_config, layers_policy, quant_safetensors_path: str = ""):
        """Apply fake quantization to the model.

        This method applies fake quantization to the model, which is useful
        for validating quantization effects without actually converting to
        integer operations.

        Args:
            ptq_config (PTQConfig): Configuration for post-training quantization.
            layers_policy (dict): Policy for different layer quantization strategies.
            quant_safetensors_path (str, optional): Path to quantized safetensors.
                Defaults to ``""``.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError
