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

""" configs for golden-stick """

from dataclasses import dataclass, field
from typing import List

import yaml

from .utils import value_check


@dataclass
class GSBaseConfig:
    """ base config for golden-stick """
    save_model_type: str = field(default='ckpt',
                                 metadata={'choices': ['ckpt', 'mindir']})
    device_target: str = field(default='ascend',
                               metadata={'choices': ['cpu', 'ascend', 'gpu']})
    dev_id: List[int] = field(default_factory=lambda: [0])

    def __post_init__(self):
        value_check('dev_id', self.dev_id, int)

    def dump(self, file_path: str):
        """dump config to yaml file"""
        parsed_dict = self._parse_dict()
        with open(file_path, 'w') as fi:
            yaml.safe_dump(parsed_dict, fi, allow_unicode=True)

    def load(self, yaml_file):
        """init config from yaml_file"""
        with open(yaml_file, 'r', encoding='utf8') as fi:
            load_data = yaml.safe_load(fi)
            self._unparse_dict(load_data)

    def _parse_dict(self):
        """ parse data class to readable dicts"""
        return self.__dict__

    def _unparse_dict(self, data_dict):
        """ convert readable dicts to data config"""
        self.__dict__.update(data_dict)
