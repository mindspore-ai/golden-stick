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
"""MindFormers model hub plugin for PTQ framework.

This module provides integration with MindFormers model hub for loading
and quantizing models from the MindFormers ecosystem.
"""


from mindspore_gs.ptq.models.base_model import BaseQuantForCausalLM
from .plugin import ModelHubPlugin


class MFModelHubPlugin(ModelHubPlugin):
    """Plugin for MindFormers models

    This class provides integration with MindFormers model hub for
    loading and quantizing models from the MindFormers ecosystem.
    """
    def __init__(self):
        """Initialize the MindFormers plugin.
        
        This constructor checks if mindformers is installed and raises
        an ImportError if it's not available.
        
        Raises:
            ImportError: If mindformers package is not installed.
        """
        super().__init__()
        try:
            # pylint: disable=unused-import
            import mindformers
        except ImportError as exc:
            raise ImportError("mindformers is not installed, please install it first.") from exc

    def _load_models(self):
        """Load MindFormers model implementations.
        
        This method imports and registers all supported MindFormers model
        implementations for quantization.
        
        The imports are marked as unused to prevent linting warnings since
        the import itself registers the models with the framework.
        """
        # pylint: disable=unused-import
        from mindspore_gs.ptq.models.mindformers_models.qwen3 import QWen3
        # pylint: disable=unused-import
        from mindspore_gs.ptq.models.mindformers_models.qwen3_moe import QWen3MoE
        # pylint: disable=unused-import
        from mindspore_gs.ptq.models.mindformers_models.deepseekv3 import DeepSeekV3
        # pylint: disable=unused-import
        from mindspore_gs.ptq.models.mindformers_models.telechat2 import Telechat2

    def _load_quant_cells(self):
        """Load MindFormers quantization cell implementations.
        
        This method imports and registers all supported quantization cell
        implementations for MindFormers models.
        
        Each cell is registered with the framework using the reg_self() method.
        """
        from mindspore_gs.ptq.quant_cells.mindformers.linear_smooth_wrappers import (
            SmoothQuantLinearCell,
            AWQSmoothLinearCell,
            OutlierSuppressionPlusLinearCell,
            OutlierSuppressionPlusSmoothLinearCell,
            SearchOutlierSuppressionLiteLinearCell)
        from mindspore_gs.ptq.quant_cells.mindformers.linear_weight_quant_wrappers import WeightQuantLinearCell
        from mindspore_gs.ptq.quant_cells.mindformers.linear_gptq_quant_wrappers import GptqWeightQuantLinearCell
        from mindspore_gs.ptq.quant_cells.mindformers.linear_clip_wrappers import ClipLinearCell
        from mindspore_gs.ptq.quant_cells.mindformers.linear_all_quant_wrappers import AllQuantLinearCell
        from mindspore_gs.ptq.quant_cells.mindformers.linear_dynamic_quant_wrappers import DynamicQuantLinearCell
        from mindspore_gs.ptq.quant_cells.mindformers.linear_gptq_dynamic_quant_wrappers import (
            GptqDynamicQuantLinearCell)
        from mindspore_gs.ptq.quant_cells.mindformers.kvcache_quant_wrappers import (QuantPageAttentionMgrCell,
                                                                                     DynamicQuantPageAttentionMgrCell)

        from mindspore_gs.ptq.quant_cells.mindformers.fake_quant_linear import (FakeQuantW8A8Wrapper,
                                                                                 FakeQuantW4A8DynamicWrapper,
                                                                                 FakeQuantW8A8DynamicWrapper)
        from mindspore_gs.ptq.quant_cells.mindformers.fake_quant_gmm import (FakeQuantW8A8DynamicGroupWrapper,
                                                                              FakeQuantW4A8DynamicGroupWrapper)

        SearchOutlierSuppressionLiteLinearCell.reg_self()
        SmoothQuantLinearCell.reg_self()
        AWQSmoothLinearCell.reg_self()
        OutlierSuppressionPlusLinearCell.reg_self()
        OutlierSuppressionPlusSmoothLinearCell.reg_self()
        WeightQuantLinearCell.reg_self()
        GptqWeightQuantLinearCell.reg_self()
        ClipLinearCell.reg_self()
        AllQuantLinearCell.reg_self()
        DynamicQuantLinearCell.reg_self()
        GptqDynamicQuantLinearCell.reg_self()
        QuantPageAttentionMgrCell.reg_self()
        DynamicQuantPageAttentionMgrCell.reg_self()

        FakeQuantW8A8Wrapper.reg_self()
        FakeQuantW4A8DynamicWrapper.reg_self()
        FakeQuantW8A8DynamicWrapper.reg_self()
        FakeQuantW8A8DynamicGroupWrapper.reg_self()
        FakeQuantW4A8DynamicGroupWrapper.reg_self()

    def _load_algo_modules(self):
        """Load MindFormers algorithm modules.
        
        This method imports and registers all supported algorithm modules
        for MindFormers models.
        """
        from mindspore_gs.ptq.algo_modules.mindformers.anti_outliers import LinearSmoothQuant, LinearAutoSmoother
        from mindspore_gs.ptq.algo_modules.mindformers.clipper import LinearClipper
        from mindspore_gs.ptq.algo_modules.quantizer import Quantizer

        LinearSmoothQuant.reg_self()
        LinearAutoSmoother.reg_self()
        LinearClipper.reg_self()
        Quantizer.reg_self()

    # pylint: disable=arguments-renamed
    def create_model(self, yaml_path) -> BaseQuantForCausalLM:
        """Create a model instance from a pretrained configuration.

        This method creates a model instance by loading the MindFormers
        configuration and selecting the appropriate specific model
        implementation based on the configuration.

        Args:
            yaml_path (str): Path to the MindFormers model configuration YAML file.

        Returns:
            BaseQuantForCausalLM: An instance of the appropriate model implementation.

        Raises:
            ValueError: If the model name in the configuration is not supported.
        """
        from mindspore_gs.ptq.models.mindformers_models.mf_model import MFModel
        return MFModel.from_pretrained(yaml_path)
