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
"""test JSONCache."""

import os
import tempfile
import unittest
from unittest.mock import patch
import pytest


from mindspore_gs.common.logger import logger
from mindspore_gs.common.json_cache import JSONCache


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
class TestJSONCache(unittest.TestCase):
    """TestJSONCache"""
    def setUp(self):
        """setUp"""
        JSONCache._instance = None
        JSONCache._filepath = None
        JSONCache._empty_path_warned = False
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file = os.path.join(self.temp_dir.name, 'test_cache.json')

    def tearDown(self):
        """tearDown"""
        self.temp_dir.cleanup()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_singleton_same_path(self):
        """
        Feature: test_singleton_same_path.
        Description: test_singleton_same_path.
        Expectation: test_singleton_same_path.
        """
        cache1 = JSONCache(self.test_file)
        cache2 = JSONCache(self.test_file)
        self.assertIs(cache1, cache2)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @patch.object(logger, 'warning')
    def test_empty_filepath_warning(self, mock_warning):
        """
        Feature: test_empty_filepath_warning.
        Description: test_empty_filepath_warning.
        Expectation: test_empty_filepath_warning.
        """
        JSONCache('')
        mock_warning.assert_called_once_with(
            "Initialized with empty filepath - data will not persist"
        )
        JSONCache._instance = None
        JSONCache('')
        self.assertEqual(mock_warning.call_count, 1)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_put_and_get(self):
        """
        Feature: test_put_and_get.
        Description: test_put_and_get.
        Expectation: test_put_and_get.
        """
        cache = JSONCache(self.test_file)
        cache.put("temperature", 25.5)
        self.assertEqual(cache.get("temperature"), 25.5)
        self.assertIn("temperature", cache)
        self.assertEqual(cache.size, 1)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_overwrite_existing_key(self):
        """
        Feature: test_overwrite_existing_key.
        Description: test_overwrite_existing_key.
        Expectation: test_overwrite_existing_key.
        """
        cache = JSONCache(self.test_file)
        cache.put("pressure", 1013.25)
        cache.put("pressure", 1020.0)
        self.assertEqual(cache.get("pressure"), 1020.0)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_nonexistent_key(self):
        """
        Feature: test_get_nonexistent_key.
        Description: test_get_nonexistent_key.
        Expectation: test_get_nonexistent_key.
        """
        cache = JSONCache(self.test_file)
        self.assertIsNone(cache.get("humidity"))
        self.assertNotIn("humidity", cache)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_persistence(self):
        """
        Feature: test_persistence.
        Description: test_persistence.
        Expectation: test_persistence.
        """
        cache = JSONCache(self.test_file)
        cache.put("key1", 1.23)
        del cache

        new_cache = JSONCache(self.test_file)
        self.assertEqual(new_cache.get("key1"), 1.23)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_corrupted_file_handling(self):
        """
        Feature: test_corrupted_file_handling.
        Description: test_corrupted_file_handling.
        Expectation: test_corrupted_file_handling.
        """
        with open(self.test_file, 'w') as f:
            f.write("{invalid_json")

        cache = JSONCache(self.test_file)
        self.assertEqual(cache.size, 0)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_invalid_key_type(self):
        """
        Feature: test_invalid_key_type.
        Description: test_invalid_key_type.
        Expectation: test_invalid_key_type.
        """
        cache = JSONCache(self.test_file)
        with self.assertRaises(TypeError):
            cache.put(123, 45.6)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_invalid_value_type(self):
        """
        Feature: test_invalid_value_type.
        Description: test_invalid_value_type.
        Expectation: test_invalid_value_type.
        """
        cache = JSONCache(self.test_file)
        with self.assertRaises(TypeError):
            cache.put("valid_key", "not_a_number")

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_directory_creation(self):
        """
        Feature: test_directory_creation.
        Description: test_directory_creation.
        Expectation: test_directory_creation.
        """
        nested_path = os.path.join(self.temp_dir.name, "nested", "cache.json")
        cache = JSONCache(nested_path)
        cache.put("test", 1.0)
        self.assertTrue(os.path.exists(os.path.dirname(nested_path)))

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_empty_filepath_no_persistence(self):
        """
        Feature: test_empty_filepath_no_persistence.
        Description: test_empty_filepath_no_persistence.
        Expectation: test_empty_filepath_no_persistence.
        """
        cache = JSONCache('')
        cache.put("ephemeral", 3.14)
        self.assertEqual(cache.get("ephemeral"), 3.14)

        self.assertFalse(os.path.exists(""))

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_size_property(self):
        """
        Feature: test_size_property.
        Description: test_size_property.
        Expectation: test_size_property.
        """
        cache = JSONCache(self.test_file)
        self.assertEqual(cache.size, 0)
        cache.put("k1", 1.0)
        cache.put("k2", 2.0)
        self.assertEqual(cache.size, 2)
        cache.put("k1", 3.0)
        self.assertEqual(cache.size, 2)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_file_not_exists(self):
        """
        Feature: test_file_not_exists.
        Description: test_file_not_exists.
        Expectation: test_file_not_exists.
        """
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        cache = JSONCache(self.test_file)
        self.assertEqual(cache.size, 0)


if __name__ == '__main__':
    unittest.main()
