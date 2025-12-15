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
"""Module for testing."""

import mindspore as ms  # pylint: disable=unused-import
from mindspore import nn


# Test case for verifying unused import handling
class NetWithUnusedImport(nn.Cell):
    """NetWithUnusedImport for testing."""
    def __init__(self):
        """Method implementation."""
        super().__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        """Method implementation."""
        x = self.relu(x)
        return x
