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
"""Algorithm base class."""

from mindspore_gs.ptq.network_helpers import NetworkHelper


class Algorithm:
    """Algorithm"""
    def process(self, decoder_layer_name: str, decoder_layer, args_list, kwargs_list, network_helper: NetworkHelper):
        """process"""
        raise NotImplementedError
