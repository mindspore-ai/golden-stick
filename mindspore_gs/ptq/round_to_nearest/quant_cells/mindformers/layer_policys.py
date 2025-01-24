# Copyright 2023 Huawei Technologies Co., Ltd
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
"""RTNLayerPolicy for mindformers layers."""

from mindspore.nn import Cell
from mindspore_gs.ptq.ptq_config import QuantGranularity
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq import PTQMode
from mindspore_gs.common import logger
from mindspore_gs.ptq.round_to_nearest.rtn_layer_policy import RTNLayerPolicy
from .quant_cells import LinearQuant, LinearDeploy, PagedAttentionDeploy, PagedAttentionQuant, \
    PagedAttentionDeployFusion, DynamicQuantPagedAttentionDeploy


class LinearLayerPolicy(RTNLayerPolicy):
    """
    Derived class of SimulatedLayerPolicy. LayerPolicy used for nn.Dense.
    """
    def __init__(self, weight_names: [], act_names: [], config: InnerPTQConfig = InnerPTQConfig()):
        super().__init__(weight_names, act_names, config)
        self.set_input_number(1)
        if config.act_quant_dtype is None:
            self.set_input_not_insert_fq()
            self.set_output_not_insert_fq()
        self.is_deploy = config.mode == PTQMode.DEPLOY

    def wrap_cell(self, handler) -> Cell:
        if self._config.weight_quant_dtype is None:
            return None
        if self.is_deploy:
            return LinearDeploy(handler, self)
        return LinearQuant(handler, self)


class PagedAttentionMgrPolicy(RTNLayerPolicy):
    """
    Derived class of SimulatedLayerPolicy. LayerPolicy used for nn.Dense.
    """
    def __init__(self, weight_names: [], act_names: [], config: InnerPTQConfig = InnerPTQConfig()):
        super().__init__(weight_names, act_names, config)
        self.set_input_number(3)
        self.is_deploy = config.mode == PTQMode.DEPLOY
        self.enable_deploy_fusion = config.enable_deploy_fusion
        self.kvcache_dynamic_quant = config.kvcache_quant_granularity == QuantGranularity.PER_TOKEN
        logger.info(f"PagedAttentionMgr Quant conifg: {config}.")

    def wrap_cell(self, handler) -> Cell:
        if self._config.kvcache_quant_dtype is None:
            return None
        if self.is_deploy:
            if self.kvcache_dynamic_quant:
                return DynamicQuantPagedAttentionDeploy(handler, self)
            if self.enable_deploy_fusion:
                return PagedAttentionDeployFusion(handler, self)
            return PagedAttentionDeploy(handler, self)
        return PagedAttentionQuant(handler, self)
