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
"""NetworkQConfig for PTQ."""
import abc

from mindspore_gs.quantization.net_policy import NetPolicy
from mindspore_gs.quantization.layer_policy import LayerPolicy, PerChannelArgs
from .fake_quantizer import FakeQuantizer


class PTQLayerPolicy(LayerPolicy):
    @abc.abstractmethod
    def get_kvcache_quantizer(self, weight_name="", perchannel_args: PerChannelArgs = PerChannelArgs(),
                              **kwargs) -> FakeQuantizer:
        """get_kvcache_quantizer"""
        raise NotImplementedError


class PTQNetPolicy(NetPolicy):
    """PTQNetPolicy"""

    register_policy_map = {}

    @staticmethod
    def register_policy(algorithm, layer_type, policy_partial):
        PTQNetPolicy.register_policy_map[(algorithm, layer_type)] = policy_partial
