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
"""test KVCache."""


import unittest
import numpy as np
from mindspore_gs.common.kvcache import KVCache


class TestKVCache(unittest.TestCase):
    """TestKVCache"""
    # pylint: disable=protected-access
    def setUp(self):
        KVCache._instance = None
        self.cache = KVCache()

    def test_singleton_pattern(self):
        """
        Feature: test_singleton_pattern.
        Description: Verify that KVCache correctly implements the singleton pattern.
        Expectation: Two instances of KVCache should be the same object.
        """
        cache1 = KVCache()
        cache2 = KVCache()
        self.assertIs(cache1, cache2)

    def test_put_and_get_single_level(self):
        """
        Feature: test_put_and_get_single_level.
        Description: Test storing and retrieving a value with a single-level key.
        Expectation: The retrieved value should match the stored value.
        """
        key = "test_key"
        value = np.array([1, 2, 3])
        self.cache.put(key, value)
        retrieved = self.cache.get(key)
        np.testing.assert_array_equal(retrieved, value)

    def test_put_and_get_multi_level(self):
        """
        Feature: test_put_and_get_multi_level.
        Description: Test storing and retrieving a value with a multi-level key.
        Expectation: The retrieved value should match the stored value.
        """
        key = ["network.model.layers.0.attention.wq", "weight", "max"]
        value = np.array([4, 5, 6])
        self.cache.put(key, value)
        retrieved = self.cache.get(key)
        np.testing.assert_array_equal(retrieved, value)

    def test_overwrite_single_level(self):
        """
        Feature: test_overwrite_single_level.
        Description: Test overwriting a value with a single-level key.
        Expectation: The new value should replace the old value.
        """
        key = "test_key"
        value1 = np.array([1, 2, 3])
        value2 = np.array([4, 5, 6])
        self.cache.put(key, value1)
        self.cache.put(key, value2)
        retrieved = self.cache.get(key)
        np.testing.assert_array_equal(retrieved, value2)

    def test_overwrite_multi_level(self):
        """
        Feature: test_overwrite_multi_level.
        Description: Test overwriting a value with a multi-level key.
        Expectation: The new value should replace the old value.
        """
        key = ["level1", "level2", "test_key"]
        value1 = np.array([1, 2, 3])
        value2 = np.array([4, 5, 6])
        self.cache.put(key, value1)
        self.cache.put(key, value2)
        retrieved = self.cache.get(key)
        np.testing.assert_array_equal(retrieved, value2)

    def test_invalid_key_type(self):
        """
        Feature: test_invalid_key_type.
        Description: Test passing an invalid key type to put method.
        Expectation: A TypeError should be raised.
        """
        invalid_key = 123
        value = np.array([1, 2, 3])
        with self.assertRaises(TypeError):
            self.cache.put(invalid_key, value)

    def test_invalid_key_type_for_get(self):
        """
        Feature: test_invalid_key_type_for_get.
        Description: Test passing an invalid key type to get method.
        Expectation: A TypeError should be raised.
        """
        invalid_key = 123
        with self.assertRaises(TypeError):
            self.cache.get(invalid_key)

    def test_key_not_found(self):
        """
        Feature: test_key_not_found.
        Description: Test retrieving a non-existent single-level key.
        Expectation: Should return None.
        """
        key = "nonexistent_key"
        self.assertIsNone(self.cache.get(key))

    def test_multi_level_key_not_found(self):
        """
        Feature: test_multi_level_key_not_found.
        Description: Test retrieving a non-existent multi-level key.
        Expectation: Should return None.
        """
        key = ["level1", "nonexistent_key"]
        self.assertIsNone(self.cache.get(key))

    def test_contains_single_level(self):
        """
        Feature: test_contains_single_level.
        Description: Test checking existence of a single-level key.
        Expectation: Should return True if the key exists.
        """
        key = "test_key"
        value = np.array([1, 2, 3])
        self.cache.put(key, value)
        self.assertTrue(key in self.cache)

    def test_contains_multi_level(self):
        """
        Feature: test_contains_multi_level.
        Description: Test checking existence of a multi-level key.
        Expectation: Should return True if the key exists.
        """
        key = ["network.model.layers.0.attention.wq", "weight", "max"]
        value = np.array([1, 2, 3])
        self.cache.put(key, value)
        self.assertTrue(key in self.cache)

    def test_contains_invalid_key_type(self):
        """
        Feature: test_contains_invalid_key_type.
        Description: Test checking existence with an invalid key type.
        Expectation: A TypeError should be raised.
        """
        invalid_key = 123
        with self.assertRaises(TypeError):
            _ = invalid_key in self.cache

    def test_str_representation(self):
        """
        Feature: test_str_representation.
        Description: Test the string representation of the cache.
        Expectation: The string should match the expected format.
        """
        key = "test_key"
        value = np.array([1, 2, 3])
        self.cache.put(key, value)
        expected_str = str({key: value})
        self.assertEqual(str(self.cache), expected_str)

    def test_empty_cache(self):
        """
        Feature: test_empty_cache.
        Description: Test behavior of an empty cache.
        Expectation: The string representation should be an empty dict, and get should return None.
        """
        self.assertEqual(str(self.cache), str({}))
        self.assertIsNone(self.cache.get("nonexistent_key"))

    def test_mixed_key_types(self):
        """
        Feature: test_mixed_key_types.
        Description: Test using both single-level and multi-level keys.
        Expectation: Both types of keys should be handled correctly.
        """
        single_key = "single_key"
        multi_key = ["multi", "level", "key"]
        single_value = np.array([1, 2, 3])
        multi_value = np.array([4, 5, 6])

        self.cache.put(single_key, single_value)
        self.cache.put(multi_key, multi_value)

        np.testing.assert_array_equal(self.cache.get(single_key), single_value)
        np.testing.assert_array_equal(self.cache.get(multi_key), multi_value)

    def test_partial_multi_level_key(self):
        """
        Feature: test_partial_multi_level_key.
        Description: Test retrieving a partial multi-level key.
        Expectation: Should return None.
        """
        key = ["level1", "level2", "test_key"]
        value = np.array([1, 2, 3])
        self.cache.put(key, value)

        partial_key = ["level1", "level2"]
        self.assertIsNone(self.cache.get(partial_key))

    def test_deeply_nested_keys(self):
        """
        Feature: test_deeply_nested_keys.
        Description: Test using deeply nested multi-level keys.
        Expectation: The value should be stored and retrieved correctly.
        """
        key = ["level1", "level2", "level3", "level4", "test_key"]
        value = np.array([1, 2, 3])
        self.cache.put(key, value)
        retrieved = self.cache.get(key)
        np.testing.assert_array_equal(retrieved, value)


if __name__ == "__main__":
    unittest.main()
