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

"""Plugin system for model hub integration.

This module provides the base plugin interface for integrating different model hubs
into the PTQ (Post-Training Quantization) framework.
"""

from mindspore_gs.ptq.models.base_model import BaseQuantForCausalLM


class ModelHubPlugin:
    """Base class for model hub plugins.
    
    This class defines the interface that all model hub plugins must implement
    to integrate with the PTQ framework. Subclasses should provide specific
    implementations for loading models and quantization cells from different
    model hubs.
    """
    @classmethod
    def load(cls):
        """Create and initialize a plugin instance.
        
        This class method creates a new plugin instance and initializes it
        by calling the internal model and quant cell loading methods.
        
        Returns:
            ModelHubPlugin: An initialized plugin instance ready for use.
        """
        plugin = cls()
        plugin._load_models()
        plugin._load_quant_cells()
        plugin._load_algo_modules()
        return plugin

    def create_model(self, pretrained) -> BaseQuantForCausalLM:
        """Create a quantized model instance from pretrained configuration.
        
        This method should be implemented by subclasses to create a specific
        quantized model instance based on the provided pretrained configuration.
        
        Args:
            pretrained: Pretrained model configuration or path. The exact type
                       depends on the specific plugin implementation.
        
        Returns:
            BaseQuantForCausalLM: A quantized model instance.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("create_model method not implemented")

    def _load_models(self):
        """Load model implementations for the plugin.
        
        This internal method should be implemented by subclasses to import
        and register the specific model implementations supported by this plugin.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("_load_models method not implemented")

    def _load_quant_cells(self):
        """Load quantization cell implementations for the plugin.
        
        This internal method should be implemented by subclasses to import
        and register the specific quantization cell implementations supported
        by this plugin.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("_load_quant_cells method not implemented")

    def _load_algo_modules(self):
        """Load algorithm modules for the plugin.
        
        This internal method should be implemented by subclasses to import
        and register the specific algorithm modules supported by this plugin.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("_load_algo_modules method not implemented")
