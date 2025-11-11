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

"""Plugin System for Multi-Framework Model Hub Integration.

This module implements the plugin architecture that enables the Golden Stick PTQ 
framework to support multiple model ecosystems through a standardized interface. 
The plugin system is a core component of the Golden Stick architecture, providing 
extensibility and framework interoperability.

Architecture Overview:
    - Plugin Interface: Standardized contract for model hub integration
    - Framework-Specific Plugins: Concrete implementations for different ecosystems
    - Automatic Plugin Discovery: Runtime detection and loading of appropriate plugins
    - Unified API: Consistent interface across different model frameworks

Supported Model Ecosystems:
    - MindOne Plugin: Integration with MindOne ecosystem models
    - MindFormers Plugin: Integration with MindFormers ecosystem models
    - Extensible: Additional plugins can be developed for other frameworks

Plugin Responsibilities:
    1. Model Registration
    2. QuantCell Registration
"""

import os
from .plugin import ModelHubPlugin
from .mindformers_plugin import MFModelHubPlugin
from .mindone_plugin import MindOneModelHubPlugin

def load_plugin(pretrained):
    """Load the appropriate plugin based on the pretrained configuration.

    This function determines which model hub plugin to load based on the
    pretrained configuration and returns an initialized plugin instance.

    The function uses a simple heuristic to determine the appropriate plugin:
    - If the pretrained configuration is a string ending with '.yaml', it assumes
      a MindFormers configuration file and loads the MFModelHubPlugin
    - Otherwise, it assumes a MindOne configuration and loads the MindOneModelHubPlugin

    Args:
        pretrained(str): The pretrained model configuration or path. The function
                         will inspect this to determine which plugin to load.

    Returns:
        ModelHubPlugin: An initialized plugin instance for the appropriate model hub.

    Raises:
        ValueError: If the pretrained configuration format is not recognized or
                    if the appropriate plugin cannot be loaded.
    
    Example:
        >>> # Load MindFormers plugin for YAML configuration
        >>> plugin = load_plugin("model_config.yaml")
        >>> 
        >>> # Load MindOne plugin for other configurations
        >>> plugin = load_plugin("mindone_model_path")
    """
    if isinstance(pretrained, str) and pretrained.endswith('.yaml'):
        return MFModelHubPlugin.load()
    if isinstance(pretrained, str) and os.path.isdir(pretrained):
        return MindOneModelHubPlugin.load()
    raise ValueError(f"Unsupported model type: {pretrained}")
