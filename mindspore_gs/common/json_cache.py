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
"""
cache key-value with json file.
"""

import json
import os
from typing import Optional
import numpy as np

from .logger import logger


class JSONCache:
    """JSONCache"""

    _instance = None
    _filepath = None
    _empty_path_warned = False

    def __new__(cls, filepath=''):
        if cls._instance is not None:
            if filepath != cls._filepath:
                raise ValueError(
                    f"Singleton already initialized with path: {cls._filepath!r}, "
                    f"attempted to reuse with: {filepath!r}"
                )
            return cls._instance

        instance = super().__new__(cls)

        if filepath == "":
            if not cls._empty_path_warned:
                logger.warning("Initialized with empty filepath - data will not persist")
                cls._empty_path_warned = True

        cls._filepath = filepath
        cls._instance = instance
        return instance

    def __init__(self, filepath=''):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._data = {}
            self._load_data()

    def _should_skip_io(self) -> bool:
        """_should_skip_io"""
        return self.__class__._filepath == ''

    def _ensure_directory_exists(self):
        if self._should_skip_io():
            return

        dir_path = os.path.dirname(self.__class__._filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    def _load_data(self):
        """_load_data"""
        if self._should_skip_io():
            self._data = {}
            return

        try:
            self._ensure_directory_exists()
            with open(self.__class__._filepath, 'r') as f:
                raw_data = json.load(f)
                for k, v in raw_data.items():
                    if isinstance(v, str):
                        self._data[str(k)] = np.load(v)
                    else:
                        self._data[str(k)] = float(v)
        except (FileNotFoundError, json.JSONDecodeError):
            self._data = {}

    def _save_data(self):
        """_save_data"""
        if self._should_skip_io():
            return

        self._ensure_directory_exists()
        with open(self.__class__._filepath, 'w') as f:
            json.dump(self._data, f, indent=2)

    def get(self, key: str) -> Optional[float]:
        return self._data.get(key, None)

    def put(self, key: str, value: float) -> None:
        """push key and value to cache and save data"""
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        try:
            if isinstance(value, np.ndarray):
                cache_path = os.path.abspath(self._filepath)
                cache_path = cache_path.split('.')[0]
                os.makedirs(cache_path, exist_ok=True)
                np.save(os.path.join(cache_path, key), value)
                num = os.path.join(cache_path, f"{key}.npy")
            else:
                num = float(value)
        except ValueError:
            raise TypeError("Value must be a number")
        self._data[key] = num
        self._save_data()

    @classmethod
    def get_filepath(cls) -> str:
        return cls._filepath

    def __contains__(self, key: str) -> bool:
        return key in self._data

    @property
    def size(self) -> int:
        return len(self._data)
