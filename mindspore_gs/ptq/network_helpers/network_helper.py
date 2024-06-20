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
"""network base class"""


import numpy as np
from mindspore.nn import Cell


class NetworkHelper:
    def get_spec(self, name: str):
        raise NotImplementedError

    def create_tokenizer(self, **kwargs):
        """create_tokenizer."""
        raise NotImplementedError

    def generate(self, network: Cell, input_ids: np.ndarray, max_new_tokens=1, **kwargs):
        raise NotImplementedError

    def assemble_inputs(self, input_ids: np.ndarray, **kwargs):
        """Prepare inputs for network.predict."""
        raise NotImplementedError
