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
cache key-value.
"""

from typing import Union, List, Optional
import numpy as np

from .logger import logger


class KVCache:
    """store by key."""
    _instance = None

    def __new__(cls):
        """Create a new instance only if one doesn't exist yet"""
        if cls._instance is None:
            cls._instance = super(KVCache, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.cache = {}

    def put(self, key: Union[str, List[str]], value):
        """Store a value in the cache

        Args:
            key: Key which can be a string (single level) or list (multi-level)
            value: Object to be stored
        """
        if isinstance(key, str):
            self.cache[key] = value
            return
        if isinstance(key, list):
            current = self.cache
            for k in key[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            if key[-1] in current:
                logger.info(f'Overwrite {key} in cache.')
            current[key[-1]] = value
            return
        raise TypeError("Key must be a string or a list of strings")

    def get(self, key: Union[str, List[str]]) -> Optional:
        """Retrieve a value from the cache

        Args:
            key: Key which can be a string (single level) or list (multi-level)

        Returns:
            Corresponding object or None if key does not exist
        """
        if isinstance(key, str):
            return self.cache.get(key)
        if isinstance(key, list):
            current = self.cache
            for k in key:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return None
            return current
        raise TypeError("Key must be a string or a list of strings")

    def __contains__(self, key: Union[str, List[str]]) -> bool:
        """Check if a key exists in the cache

        Args:
            key: Key which can be a string (single level) or list (multi-level)

        Returns:
            Boolean indicating whether the key exists
        """
        if isinstance(key, str):
            return key in self.cache
        if isinstance(key, list):
            current = self.cache
            for k in key:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return False
            return isinstance(current, np.ndarray)
        raise TypeError("Key must be a string or a list of strings")

    def __str__(self):
        """Return string representation of the cache"""
        return str(self.cache)
