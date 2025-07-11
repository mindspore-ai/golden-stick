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
dump functions for golden-stick
"""

import os
import numpy as np
from mindspore.communication import get_rank


class Dumper:
    """Dumper for GoldenStick."""
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(Dumper, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.path = ""
        self.layer_name = ""
        self.nsamples = 0

    def set_dump_path(self, path):
        """set_dump_path"""
        if path != "":
            self.path = os.path.join(path, f"rank_{get_rank()}/")
            os.makedirs(self.path, exist_ok=True)

    def _get_layer_name(self, layer_name: str):
        """obtain the name of the provided layer"""
        if layer_name != "":
            if self.nsamples < 10:
                self.layer_name = (f"0000{self.nsamples}|layer" + "_" + layer_name.split('.')[3] + "|" +
                                   layer_name.split('.')[-1])
            elif self.nsamples >= 10 and self.nsamples < 100:
                self.layer_name = (f"000{self.nsamples}|layer" + "_" + layer_name.split('.')[3] + "|" +
                                   layer_name.split('.')[-1])
            elif self.nsamples >= 100 and self.nsamples < 1000:
                self.layer_name = (f"00{self.nsamples}|layer" + "_" + layer_name.split('.')[3] + "|" +
                                   layer_name.split('.')[-1])
            elif self.nsamples >= 1000 and self.nsamples < 10000:
                self.layer_name = (f"0{self.nsamples}|layer" + "_" + layer_name.split('.')[3] + "|" +
                                   layer_name.split('.')[-1])
            else:
                self.layer_name = (f"{self.nsamples}|layer" + "_" + layer_name.split('.')[3] + "|" +
                                   layer_name.split('.')[-1])
        else:
            self.layer_name = ""

    def dump_data(self, layer_name: str, name: str, params):
        """save data"""
        if self.path != "":
            self._get_layer_name(layer_name)
            path = os.path.join(self.path, self.layer_name + name)
            np.save(path, params.asnumpy())
            self.nsamples += 1
