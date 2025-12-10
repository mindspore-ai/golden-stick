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
"""MindOne model hub plugin for PTQ framework.

This module provides integration with MindOne model hub for loading
and quantizing models from the MindOne ecosystem.
"""


from mindspore_gs.common import logger
from mindspore_gs.ptq.models.base_model import BaseQuantForCausalLM
from .plugin import ModelHubPlugin


class MindOneModelHubPlugin(ModelHubPlugin):
    """Plugin for MindOne models
    
    This class provides integration with MindOne model hub for
    loading and quantizing models from the MindOne ecosystem.
    """
    def __init__(self):
        """Initialize the MindOne plugin.
        
        This constructor checks if mindone is installed and raises
        an ImportError if it's not available.
        
        Raises:
            ImportError: If mindone package is not installed.
        """
        super().__init__()
        try:
            # pylint: disable=unused-import
            import mindone
        except ImportError as exc:
            raise ImportError("mindone is not installed, please install it first.") from exc

    def _load_models(self):
        """Load MindOne model implementations.
        
        This method imports and registers all supported MindOne model
        implementations for quantization.
        
        The imports are marked as unused to prevent linting warnings since
        the import itself registers the models with the framework.
        """
        # pylint: disable=unused-import
        from mindspore_gs.ptq.models.mindone_models.glm4v import GLM4v
        from mindspore_gs.ptq.models.mindone_models.qwen3 import Qwen3
        from mindspore_gs.ptq.models.mindone_models.qwen3_vl import Qwen3VL

    def _load_quant_cells(self):
        """Load MindOne quantization cell implementations.
        
        This method is currently not implemented for MindOne models as
        they use the same quantization cells as MindFormers models.
        
        No quantization cells need to be specifically loaded for MindOne
        at this time.
        """
        from mindspore_gs.ptq.quant_cells.mindone.linear_weight_quant_wrappers import WeightQuantLinearCell
        from mindspore_gs.ptq.quant_cells.mindone.linear_all_quant_wrappers import AllQuantLinearCell
        from mindspore_gs.ptq.quant_cells.mindone.linear_dynamic_quant_wrappers import DynamicQuantLinearCell
        from mindspore_gs.ptq.quant_cells.mindone.linear_gptq_quant_wrappers import GptqWeightQuantLinearCell

        from mindspore_gs.ptq.quant_cells.mindone.linear_smooth_wrappers import SmoothQuantLinearCell
        from mindspore_gs.ptq.quant_cells.mindone.linear_smooth_wrappers import OSLSmoothQuantLinearCell
        from mindspore_gs.ptq.quant_cells.mindone.linear_smooth_wrappers import AWQSmoothQuantLinearCell

        from mindspore_gs.ptq.quant_cells.mindone.linear_clip_wrappers import ClipLinearCell

        from mindspore_gs.ptq.quant_cells.mindone.fake_quant_linear import FakeQuantA16WxWrapper
        from mindspore_gs.ptq.quant_cells.mindone.fake_quant_linear import FakeQuantW8A8Wrapper
        from mindspore_gs.ptq.quant_cells.mindone.fake_quant_linear import FakeQuantW8A8DynamicWrapper

        WeightQuantLinearCell.reg_self()
        AllQuantLinearCell.reg_self()
        DynamicQuantLinearCell.reg_self()
        GptqWeightQuantLinearCell.reg_self()

        SmoothQuantLinearCell.reg_self()
        OSLSmoothQuantLinearCell.reg_self()
        AWQSmoothQuantLinearCell.reg_self()

        ClipLinearCell.reg_self()

        FakeQuantA16WxWrapper.reg_self()
        FakeQuantW8A8Wrapper.reg_self()
        FakeQuantW8A8DynamicWrapper.reg_self()

    def _load_algo_modules(self):
        """Load MindOne algorithm modules.
        
        This method is currently not implemented for MindOne models as
        they use the same algorithm modules as MindFormers models.
        """
        from mindspore_gs.ptq.algo_modules.mindone.anti_outliers import (SmoothQuantSmoother,
                                                                         OSLSmoother,
                                                                         AWQSmoother)
        from mindspore_gs.ptq.algo_modules.mindone.clipper import LinearClipper
        from mindspore_gs.ptq.algo_modules.quantizer import Quantizer

        SmoothQuantSmoother.reg_self()
        OSLSmoother.reg_self()
        AWQSmoother.reg_self()
        LinearClipper.reg_self()
        Quantizer.reg_self()

    def create_model(self, pretrained) -> BaseQuantForCausalLM:
        """Create a model instance from a pretrained configuration.

        This method creates a model instance by loading the MindOne
        configuration and selecting the appropriate specific model
        implementation based on the configuration.

        Args:
            pretrained: Pretrained model configuration or path for MindOne models.

        Returns:
            BaseQuantForCausalLM: An instance of the appropriate model implementation.

        Raises:
            ValueError: If the model name in the configuration is not supported.
        """
        from mindspore_gs.ptq.models.mindone_models.mindone_model import MindOneModel
        logger.info('Creating mindone model...', flush=True)
        return MindOneModel.from_pretrained(pretrained)
