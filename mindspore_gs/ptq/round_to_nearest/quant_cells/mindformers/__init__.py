# Copyright 2024 Huawei Technologies Co., Ltd
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
LayerPolicy for Mindformers.
"""
from functools import partial

from mindformers.modules.layers import Linear
from mindformers.modules.paged_attention_mgr import PagedAttentionMgr
from research.telechat.telechat_layer import TelechatLinear
from research.telechat2.telechat_layer import TelechatLinear as TelechatLinear2
from mindspore_gs.ptq.ptq_policy import PTQNetPolicy
from mindspore_gs.ptq.round_to_nearest.rtn_net_policy import RTNNetPolicy
from .layer_policys import LinearLayerPolicy, PagedAttentionMgrPolicy, TeleLinearLayerPolicy, TeleLinearLayerPolicy2

PTQNetPolicy.register_policy(RTNNetPolicy, Linear, partial(LinearLayerPolicy, [], []))
PTQNetPolicy.register_policy(RTNNetPolicy, PagedAttentionMgr, partial(PagedAttentionMgrPolicy, [], []))
PTQNetPolicy.register_policy(RTNNetPolicy, TelechatLinear, partial(TeleLinearLayerPolicy, [], []))
PTQNetPolicy.register_policy(RTNNetPolicy, TelechatLinear2, partial(TeleLinearLayerPolicy2, [], []))

