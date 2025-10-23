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
"""Test SafeTensorsMgr."""

import os
import tempfile
import shutil
import json
import pytest

from mindspore_gs.ptq.models.safetensors_mgr import SafeTensorsMgr


class TestSafeTensorsMgr:
    """Test cases for SafeTensorsMgr class."""

    def setup_method(self):
        """Setup method to create temporary directories and test files."""
        # Create temporary work directory
        self.work_dir = tempfile.mkdtemp()
        self.original_path = os.path.join(self.work_dir, "original")
        self.save_path = os.path.join(self.work_dir, "save")

        # Create directories
        os.makedirs(self.original_path)
        os.makedirs(self.save_path)

        # Create test files in original directory
        self._create_test_files()

    def teardown_method(self):
        """Teardown method to clean up temporary directories."""
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)

    def _create_test_files(self):
        """Create various test files in the original directory."""
        # Create a regular JSON file (should be copied)
        config_data = {"model_type": "test", "hidden_size": 768}
        with open(os.path.join(self.original_path, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(config_data, f)

        # Create a tokenizer JSON file (should be copied)
        tokenizer_data = {"vocab_size": 50000}
        with open(os.path.join(self.original_path, "tokenizer.json"), 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f)

        # Create an index.json file (should be filtered out)
        index_data = {"weight_map": {"layer.0.weight": "model-00001-of-00002.safetensors"}}
        with open(os.path.join(self.original_path, "model.safetensors.index.json"), 'w', encoding='utf-8') as f:
            json.dump(index_data, f)

        # Create a safetensors file (should be filtered out)
        with open(os.path.join(self.original_path, "model.safetensors"), 'wb') as f:
            f.write(b"fake safetensors content")

        # Create a text file (should be copied)
        with open(os.path.join(self.original_path, "README.txt"), 'w', encoding='utf-8') as f:
            f.write("This is a test README file.")

        # Create a Python file (should be copied)
        with open(os.path.join(self.original_path, "modeling.py"), 'w', encoding='utf-8') as f:
            f.write("# Test Python file\nprint('Hello World')")

        # Create another safetensors file with different name (should be filtered out)
        with open(os.path.join(self.original_path, "pytorch_model.safetensors"), 'wb') as f:
            f.write(b"another fake safetensors content")

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_copy_original_files_basic(self):
        """
        Feature: SafeTensorsMgr _copy_original_files method.
        Description: Test basic file copying with blacklist filtering.
        Expectation: Only non-blacklisted files are copied.
        """
        # Create SafeTensorsMgr instance
        mgr = SafeTensorsMgr(self.original_path)

        # Call the private method directly for testing
        mgr._copy_original_files(self.original_path, self.save_path)

        # Check that expected files are copied
        expected_files = ["config.json", "tokenizer.json", "README.txt", "modeling.py"]
        for expected_file in expected_files:
            copied_file_path = os.path.join(self.save_path, expected_file)
            assert os.path.exists(copied_file_path), f"Expected file {expected_file} was not copied"

        # Check that blacklisted files are not copied
        blacklisted_files = ["model.safetensors.index.json", "model.safetensors", "pytorch_model.safetensors"]
        for blacklisted_file in blacklisted_files:
            copied_file_path = os.path.join(self.save_path, blacklisted_file)
            assert not os.path.exists(copied_file_path), f"Blacklisted file {blacklisted_file} should not be copied"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_copy_original_files_content_integrity(self):
        """
        Feature: SafeTensorsMgr _copy_original_files method.
        Description: Test that copied files maintain content integrity.
        Expectation: Copied files have identical content to originals.
        """
        # Create SafeTensorsMgr instance
        mgr = SafeTensorsMgr(self.original_path)

        # Call the private method
        mgr._copy_original_files(self.original_path, self.save_path)

        # Verify content integrity for JSON files
        original_config = os.path.join(self.original_path, "config.json")
        copied_config = os.path.join(self.save_path, "config.json")

        with open(original_config, 'r', encoding='utf-8') as f:
            original_content = json.load(f)
        with open(copied_config, 'r', encoding='utf-8') as f:
            copied_content = json.load(f)

        assert original_content == copied_content, "Config file content should be identical"

        # Verify content integrity for text files
        original_readme = os.path.join(self.original_path, "README.txt")
        copied_readme = os.path.join(self.save_path, "README.txt")

        with open(original_readme, 'r', encoding='utf-8') as f:
            original_text = f.read()
        with open(copied_readme, 'r', encoding='utf-8') as f:
            copied_text = f.read()

        assert original_text == copied_text, "README file content should be identical"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_copy_original_files_empty_directory(self):
        """
        Feature: SafeTensorsMgr _copy_original_files method.
        Description: Test behavior with empty source directory.
        Expectation: No files are copied, no errors occur.
        """
        # Create empty source directory
        empty_source = os.path.join(self.work_dir, "empty_source")
        os.makedirs(empty_source)

        # Create SafeTensorsMgr instance
        mgr = SafeTensorsMgr(empty_source)

        # Call the private method
        mgr._copy_original_files(empty_source, self.save_path)

        # Check that save directory is still empty (except for any pre-existing files)
        copied_files = [f for f in os.listdir(self.save_path) if os.path.isfile(os.path.join(self.save_path, f))]
        assert not copied_files, "No files should be copied from empty directory"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_copy_original_files_subdirectories(self):
        """
        Feature: SafeTensorsMgr _copy_original_files method.
        Description: Test that subdirectories are not processed (only files in root).
        Expectation: Only files in the root directory are considered.
        """
        # Create subdirectory with files
        subdir = os.path.join(self.original_path, "subdir")
        os.makedirs(subdir)

        with open(os.path.join(subdir, "sub_config.json"), 'w', encoding='utf-8') as f:
            json.dump({"sub": "config"}, f)

        # Create SafeTensorsMgr instance
        mgr = SafeTensorsMgr(self.original_path)

        # Call the private method
        mgr._copy_original_files(self.original_path, self.save_path)

        # Check that subdirectory file is not copied
        sub_config_path = os.path.join(self.save_path, "sub_config.json")
        assert not os.path.exists(sub_config_path), "Files in subdirectories should not be copied"

        # Check that subdirectory itself is not copied
        subdir_path = os.path.join(self.save_path, "subdir")
        assert not os.path.exists(subdir_path), "Subdirectories should not be copied"

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_copy_original_files_various_extensions(self):
        """
        Feature: SafeTensorsMgr _copy_original_files method.
        Description: Test copying files with various extensions.
        Expectation: All non-blacklisted files are copied regardless of extension.
        """
        # Create files with various extensions
        test_files = {
            "model.bin": b"binary model data",
            "vocab.txt": "vocabulary content",
            "special_tokens_map.json": '{"unk_token": "[UNK]"}',
            "tokenizer_config.json": '{"do_lower_case": true}',
            "generation_config.json": '{"max_length": 512}',
            "config.yaml": "model_type: test\nhidden_size: 768",
            "requirements.txt": "torch>=1.9.0\ntransformers>=4.0.0"
        }

        for filename, content in test_files.items():
            file_path = os.path.join(self.original_path, filename)
            mode = 'wb' if isinstance(content, bytes) else 'w'
            with open(file_path, mode) as f:
                f.write(content)

        # Create SafeTensorsMgr instance
        mgr = SafeTensorsMgr(self.original_path)

        # Call the private method
        mgr._copy_original_files(self.original_path, self.save_path)

        # Check that all test files are copied
        for filename in test_files:
            copied_file_path = os.path.join(self.save_path, filename)
            assert os.path.exists(copied_file_path), f"File {filename} should be copied"
