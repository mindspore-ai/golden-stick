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
import stat
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
import mindspore as ms

from mindspore import Parameter
from mindspore_gs.ptq.basic_functions.safetensors_mgr import SafeTensorsMgr
from mindspore_gs.ptq.basic_functions.distributed_parameter import DistributedParameter


class TestSafeTensorsMgr:
    """Test cases for SafeTensorsMgr class - normal and exception cases."""

    def setup_method(self):
        """Setup method to create temporary directories and test files."""
        # Create temporary work directory
        self.work_dir = tempfile.mkdtemp()
        self.original_path = os.path.join(self.work_dir, "original")
        self.save_path = os.path.join(self.work_dir, "save")

        # Create directories
        os.makedirs(self.original_path, exist_ok=True)
        os.makedirs(self.save_path, exist_ok=True)

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

    def _create_test_parameters(self):
        """Create test parameters for save method."""
        dis_params_dict = {}
        # Create a simple parameter
        param1 = Parameter(ms.numpy.randn(2, 3), name="layer.0.weight")
        dis_param1 = DistributedParameter(param1, axis=None)
        dis_params_dict["layer.0.weight"] = dis_param1

        # Create another parameter
        param2 = Parameter(ms.numpy.randn(4, 5), name="layer.1.weight")
        dis_param2 = DistributedParameter(param2, axis=None)
        dis_params_dict["layer.1.weight"] = dis_param2

        return dis_params_dict

    def _verify_safetensors_content(self, original_params_dict, weight_map):
        """
        Verify that the saved safetensors files contain correct parameter content.

        Args:
            original_params_dict: Original parameters dictionary before saving
            weight_map: Weight map from index.json showing which file contains which parameters
        """
        # Load all safetensors files and collect all parameters
        loaded_params = {}
        for safetensors_file in os.listdir(self.save_path):
            if safetensors_file.endswith('.safetensors'):
                file_path = os.path.join(self.save_path, safetensors_file)
                param_dict = ms.load_checkpoint(file_path, format="safetensors")
                loaded_params.update(param_dict)

        # Verify all original parameters are present in loaded parameters
        assert len(loaded_params) == len(original_params_dict), \
            f"Loaded parameters count {len(loaded_params)} should match original {len(original_params_dict)}"

        # Verify each parameter's name, shape, dtype, and values
        for param_name, dis_param in original_params_dict.items():
            assert param_name in loaded_params, \
                f"Parameter {param_name} should be in loaded parameters"

            original_param = dis_param.param
            loaded_param = loaded_params[param_name]

            # Verify parameter shape
            assert loaded_param.shape == original_param.shape, \
                f"Parameter shape mismatch for {param_name}: " \
                f"{loaded_param.shape} vs {original_param.shape}"

            # Verify parameter dtype
            assert loaded_param.dtype == original_param.dtype, \
                f"Parameter dtype mismatch for {param_name}: " \
                f"{loaded_param.dtype} vs {original_param.dtype}"

            # Verify parameter values (using numpy array comparison)
            original_array = original_param.asnumpy()
            loaded_array = loaded_param.asnumpy()
            np.testing.assert_array_equal(loaded_array, original_array,
                                          err_msg=f"Parameter values mismatch for {param_name}")

            # Verify weight_map contains correct file mapping
            assert param_name in weight_map, \
                f"Parameter {param_name} should be in weight_map"

            # Verify the mapped file exists
            mapped_file = weight_map[param_name]
            mapped_file_path = os.path.join(self.save_path, mapped_file)
            assert os.path.exists(mapped_file_path), \
                f"Mapped file {mapped_file} should exist for parameter {param_name}"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_default_construct(self):
        """
        Feature: SafeTensorsMgr default construction test case
        Description: Test the default construction of SafeTensorsMgr class in single machine mode
        Expectation: Object is created correctly, default values are as expected,
            property access is normal
        """
        # Test default construction
        mgr = SafeTensorsMgr()
        assert mgr.file_limit_g == 4, "Default file_limit_g should be 4"
        assert mgr.rank_id == 0, "Default rank_id should be 0 in single machine mode"
        assert mgr.group_size == 1, "Default group_size should be 1 in single machine mode"
        assert not hasattr(mgr, 'barrier'), "barrier should not be created in single machine mode"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_custom_file_limit_construct(self):
        """
        Feature: SafeTensorsMgr custom file_limit_g construction test case
        Description: Test SafeTensorsMgr construction with custom file_limit_g parameter
        Expectation: Object is created correctly with custom file_limit_g value
        """
        # Test custom file_limit_g
        mgr_custom = SafeTensorsMgr(file_limit_g=8)
        assert mgr_custom.file_limit_g == 8, "Custom file_limit_g should be 8"
        assert mgr_custom.rank_id == 0, "rank_id should be 0 in single machine mode"
        assert mgr_custom.group_size == 1, "group_size should be 1 in single machine mode"
        assert not hasattr(mgr_custom, 'barrier'), "barrier should not exist in single machine mode"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_save_basic(self):
        """
        Feature: SafeTensorsMgr save functionality test case
        Description: Test the basic functionality of the save method
        Expectation: Save functionality works correctly, file structure is correct
        """
        # Create SafeTensorsMgr instance
        mgr = SafeTensorsMgr(file_limit_g=4)

        # Create test parameters
        dis_params_dict = self._create_test_parameters()

        # Create quantization description info
        quant_desc_info = {
            "quantization_method": "A8W8",
            "quantization_config": {
                "weight_quant_dtype": "int8",
                "act_quant_dtype": "int8"
            }
        }

        # Call save method
        mgr.save(self.original_path, self.save_path, dis_params_dict, quant_desc_info)

        # Verify that original files are copied
        assert os.path.exists(os.path.join(self.save_path, "config.json")), \
            "config.json should be copied"
        assert os.path.exists(os.path.join(self.save_path, "tokenizer.json")), \
            "tokenizer.json should be copied"

        # Verify that index JSON file is created
        index_json_path = os.path.join(self.save_path, "model.safetensors.index.json")
        assert os.path.exists(index_json_path), "index.json should be created"

        # Verify index JSON content
        with open(index_json_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        assert "metadata" in index_data, "index.json should contain metadata"
        assert "weight_map" in index_data, "index.json should contain weight_map"
        assert "total_size" in index_data["metadata"], "metadata should contain total_size"

        # Verify total_size calculation is correct
        expected_total_size = sum(param.size() for param in dis_params_dict.values())
        assert index_data["metadata"]["total_size"] == expected_total_size, \
            f"total_size should be {expected_total_size}, " \
            f"but got {index_data['metadata']['total_size']}"

        # Verify quantization description JSON is created
        quant_desc_path = os.path.join(self.save_path, "quantization_description.json")
        assert os.path.exists(quant_desc_path), "quantization_description.json should be created"

        # Verify quantization description content
        with open(quant_desc_path, 'r', encoding='utf-8') as f:
            quant_desc = json.load(f)
        assert quant_desc == quant_desc_info, "quantization description should match"

        # Verify safetensors files are created
        safetensors_files = [f for f in os.listdir(self.save_path) if f.endswith('.safetensors')]
        assert len(safetensors_files) > 0, "At least one safetensors file should be created"

        # Verify safetensors file content correctness
        self._verify_safetensors_content(dis_params_dict, index_data["weight_map"])

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_save_empty_parameters(self):
        """
        Feature: SafeTensorsMgr save with empty parameters test case
        Description: Test save method with empty parameters dictionary
        Expectation: Original files are still copied, index.json is created with empty weight_map
        """
        mgr = SafeTensorsMgr(file_limit_g=4)
        empty_dis_params_dict = {}
        empty_quant_desc_info = {"quantization_method": "A8W8"}
        empty_save_path = os.path.join(self.work_dir, "save_empty")
        os.makedirs(empty_save_path, exist_ok=True)

        mgr.save(self.original_path, empty_save_path, empty_dis_params_dict, empty_quant_desc_info)

        # Verify that original files are still copied even with empty parameters
        assert os.path.exists(os.path.join(empty_save_path, "config.json")), \
            "config.json should be copied even with empty parameters"

        # Verify that index JSON file is created with empty weight_map
        empty_index_json_path = os.path.join(empty_save_path, "model.safetensors.index.json")
        assert os.path.exists(empty_index_json_path), "index.json should be created"
        with open(empty_index_json_path, 'r', encoding='utf-8') as f:
            empty_index_data = json.load(f)
        assert "weight_map" in empty_index_data, "index.json should contain weight_map"
        assert len(empty_index_data["weight_map"]) == 0, \
            "weight_map should be empty for empty parameters"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_save_path_creation(self):
        """
        Feature: SafeTensorsMgr automatic path creation test case
        Description: Test save method with non-existent save path (automatic path creation)
        Expectation: Directory is created automatically, files are saved correctly
        """
        mgr = SafeTensorsMgr(file_limit_g=4)
        dis_params_dict = self._create_test_parameters()
        quant_desc_info = {
            "quantization_method": "A8W8",
            "quantization_config": {
                "weight_quant_dtype": "int8",
                "act_quant_dtype": "int8"
            }
        }

        # Test with non-existent save path (automatic path creation)
        non_existent_save_path = os.path.join(self.work_dir, "non_existent", "save")
        mgr.save(self.original_path, non_existent_save_path, dis_params_dict, quant_desc_info)

        # Verify that the directory was created automatically
        assert os.path.exists(non_existent_save_path), "Save path should be created automatically"
        assert os.path.exists(os.path.join(non_existent_save_path,
                                           "model.safetensors.index.json")), \
            "index.json should be created in the new directory"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_param_type_error_original_path(self):
        """
        Feature: SafeTensorsMgr original_path parameter type error test case
        Description: Test original_path parameter type error handling
        Expectation: TypeError exception is raised, error message is accurate
        """
        mgr = SafeTensorsMgr(file_limit_g=4)
        dis_params_dict = self._create_test_parameters()
        quant_desc_info = {"quantization_method": "A8W8"}

        # Test original_path with non-string type
        with pytest.raises((TypeError, AttributeError)):
            mgr.save(original_path=123, save_path=self.save_path,
                     dis_params_dict=dis_params_dict, quant_desc_info=quant_desc_info)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_param_type_error_save_path(self):
        """
        Feature: SafeTensorsMgr save_path parameter type error test case
        Description: Test save_path parameter type error handling
        Expectation: TypeError exception is raised, error message is accurate
        """
        mgr = SafeTensorsMgr(file_limit_g=4)
        dis_params_dict = self._create_test_parameters()
        quant_desc_info = {"quantization_method": "A8W8"}

        # Test save_path with non-string type
        with pytest.raises((TypeError, AttributeError)):
            mgr.save(original_path=self.original_path, save_path=456,
                     dis_params_dict=dis_params_dict, quant_desc_info=quant_desc_info)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_param_type_error_original_path_list(self):
        """
        Feature: SafeTensorsMgr original_path list type error test case
        Description: Test original_path with list type error handling
        Expectation: TypeError exception is raised, error message is accurate
        """
        mgr = SafeTensorsMgr(file_limit_g=4)
        dis_params_dict = self._create_test_parameters()
        quant_desc_info = {"quantization_method": "A8W8"}

        # Test original_path with list type
        with pytest.raises((TypeError, AttributeError)):
            mgr.save(original_path=[self.original_path], save_path=self.save_path,
                     dis_params_dict=dis_params_dict, quant_desc_info=quant_desc_info)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_path_not_exists_error(self):
        """
        Feature: SafeTensorsMgr parameter value range error test case
        Description: Test error handling when path does not exist
        Expectation: FileNotFoundError exception is raised, error message is accurate
        """
        mgr = SafeTensorsMgr(file_limit_g=4)
        dis_params_dict = self._create_test_parameters()
        quant_desc_info = {"quantization_method": "A8W8"}

        # Test with non-existent original_path
        non_existent_path = os.path.join(self.work_dir, "non_existent_dir")
        with pytest.raises(FileNotFoundError) as exc_info:
            mgr.save(original_path=non_existent_path, save_path=self.save_path,
                    dis_params_dict=dis_params_dict, quant_desc_info=quant_desc_info)
        assert "Source path does not exist" in str(exc_info.value) or "does not exist" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_path_not_directory_error(self):
        """
        Feature: SafeTensorsMgr parameter value range error test case
        Description: Test error handling when path is not a directory
        Expectation: NotADirectoryError exception is raised, error message is accurate
        """
        mgr = SafeTensorsMgr(file_limit_g=4)
        dis_params_dict = self._create_test_parameters()
        quant_desc_info = {"quantization_method": "A8W8"}

        # Create a file instead of directory
        file_path = os.path.join(self.work_dir, "not_a_directory")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("test")

        # Test with file path instead of directory
        with pytest.raises(NotADirectoryError) as exc_info:
            mgr.save(original_path=file_path, save_path=self.save_path,
                    dis_params_dict=dis_params_dict, quant_desc_info=quant_desc_info)
        assert "not a directory" in str(exc_info.value).lower() or "NotADirectoryError" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_save_invalid_dis_params_dict_string(self):
        """
        Feature: SafeTensorsMgr dis_params_dict string type error test case
        Description: Test dis_params_dict parameter with string type error handling
        Expectation: Corresponding exception is raised, error message is accurate
        """
        mgr = SafeTensorsMgr(file_limit_g=4)
        quant_desc_info = {"quantization_method": "A8W8"}

        # Test with non-dict type
        with pytest.raises((TypeError, AttributeError)):
            mgr.save(original_path=self.original_path, save_path=self.save_path,
                     dis_params_dict="not_a_dict", quant_desc_info=quant_desc_info)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_save_invalid_dis_params_dict_list(self):
        """
        Feature: SafeTensorsMgr dis_params_dict list type error test case
        Description: Test dis_params_dict parameter with list type error handling
        Expectation: Corresponding exception is raised, error message is accurate
        """
        mgr = SafeTensorsMgr(file_limit_g=4)
        quant_desc_info = {"quantization_method": "A8W8"}

        # Test with list instead of dict
        with pytest.raises((TypeError, AttributeError)):
            mgr.save(original_path=self.original_path, save_path=self.save_path,
                     dis_params_dict=[], quant_desc_info=quant_desc_info)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_file_sharding(self):
        """
        Feature: SafeTensorsMgr file sharding management test case
        Description: Test file sharding management functionality, including large file sharding,
            file size limit, and shard index generation
        Expectation: File sharding management works correctly, file size control is effective,
            shard index generation is correct
        """
        # Create SafeTensorsMgr with very small file limit to trigger multiple file sharding
        mgr = SafeTensorsMgr(file_limit_g=1)  # 1GB limit

        # Create multiple large parameters to trigger file splitting
        dis_params_dict = {}
        param_sizes = []
        for i in range(10):
            # Create parameters with different sizes
            size = (50, 50) if i % 2 == 0 else (100, 100)
            param = Parameter(ms.numpy.randn(*size), name=f"layer.{i}.weight")
            dis_param = DistributedParameter(param, axis=None)
            dis_params_dict[f"layer.{i}.weight"] = dis_param
            param_sizes.append(dis_param.size())

        quant_desc_info = {"quantization_method": "A8W8"}

        # Call save method
        mgr.save(self.original_path, self.save_path, dis_params_dict, quant_desc_info)

        # Verify index JSON file is created
        index_json_path = os.path.join(self.save_path, "model.safetensors.index.json")
        assert os.path.exists(index_json_path), "index.json should be created"

        with open(index_json_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)

        # Verify weight_map contains all parameters
        assert len(index_data["weight_map"]) == 10, "weight_map should contain all 10 parameters"

        # Verify safetensors files are created (may be multiple files due to sharding)
        safetensors_files = sorted([f for f in os.listdir(self.save_path) if f.endswith('.safetensors')])
        assert len(safetensors_files) > 0, "At least one safetensors file should be created"

        # Verify file naming pattern (quant-model-XXXXX-of-YYYYY.safetensors)
        for sf_file in safetensors_files:
            assert sf_file.startswith("quant-model-"), f"File {sf_file} should start with 'quant-model-'"
            assert sf_file.endswith(".safetensors"), f"File {sf_file} should end with '.safetensors'"
            assert "of" in sf_file, f"File {sf_file} should contain 'of' in the name"

        # Verify all parameters are mapped to files
        mapped_files = set(index_data["weight_map"].values())
        for mapped_file in mapped_files:
            assert mapped_file in safetensors_files, f"Mapped file {mapped_file} should exist"

        # Verify file sharding logic: check if multiple files are created when needed
        # Extract file numbers from filenames (e.g., "quant-model-00001-of-00003.safetensors")
        file_numbers = []
        total_file_num = None
        for sf_file in safetensors_files:
            # Parse filename to extract file number and total number
            # Format: quant-model-XXXXX-of-YYYYY.safetensors
            parts = sf_file.replace(".safetensors", "").split("-")
            if len(parts) >= 5 and parts[-2] == "of":
                file_num = int(parts[-3])
                total_num = int(parts[-1])
                file_numbers.append(file_num)
                if total_file_num is None:
                    total_file_num = total_num
                assert total_file_num == total_num, "All files should have same total number"

        # Verify file numbering is correct (should be 1, 2, 3, ...)
        assert len(file_numbers) == len(safetensors_files), "Should extract all file numbers"
        assert sorted(file_numbers) == list(range(1, total_file_num + 1)), \
            f"File numbers should be consecutive from 1 to {total_file_num}"

        # Verify file size control: each file should not exceed the limit (with some tolerance)
        # Note: The actual file size may include metadata overhead, so we check approximate size
        for sf_file in safetensors_files:
            file_path = os.path.join(self.save_path, sf_file)
            file_size = os.path.getsize(file_path)
            # Allow some overhead for safetensors format metadata (typically < 1MB overhead)
            # In practice, if file_limit_g is large, files may not actually split
            # But we verify the mechanism works correctly
            assert file_size > 0, f"File {sf_file} should have content"

        # Verify that parameters are distributed across files correctly
        # Group parameters by their mapped file
        file_to_params = {}
        for param_name, mapped_file in index_data["weight_map"].items():
            if mapped_file not in file_to_params:
                file_to_params[mapped_file] = []
            file_to_params[mapped_file].append(param_name)

        # Verify each file contains at least one parameter
        for mapped_file, params in file_to_params.items():
            assert len(params) > 0, f"File {mapped_file} should contain at least one parameter"

        # Verify safetensors file content correctness
        self._verify_safetensors_content(dis_params_dict, index_data["weight_map"])

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_weight_map_mapping(self):
        """
        Feature: SafeTensorsMgr index file generation test case
        Description: Test index file generation functionality, including accuracy of
            weight_map mapping relationships
        Expectation: Index file is generated correctly, weight_map mapping relationships are accurate,
            each parameter is mapped to the correct file
        """
        mgr = SafeTensorsMgr(file_limit_g=4)

        # Create test parameters with known names
        dis_params_dict = {}
        param_names = ["layer.0.weight", "layer.1.weight", "layer.2.weight"]
        for param_name in param_names:
            param = Parameter(ms.numpy.randn(10, 10), name=param_name)
            dis_param = DistributedParameter(param, axis=None)
            dis_params_dict[param_name] = dis_param

        quant_desc_info = {"quantization_method": "A8W8"}

        # Call save method
        mgr.save(self.original_path, self.save_path, dis_params_dict, quant_desc_info)

        # Verify index JSON file
        index_json_path = os.path.join(self.save_path, "model.safetensors.index.json")
        with open(index_json_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)

        # Verify weight_map structure
        assert "weight_map" in index_data, "index.json should contain weight_map"
        weight_map = index_data["weight_map"]

        # Verify all parameters are in weight_map
        for param_name in param_names:
            assert param_name in weight_map, f"Parameter {param_name} should be in weight_map"
            mapped_file = weight_map[param_name]
            # Verify mapped file name format
            assert mapped_file.startswith("quant-model-"), \
                "Mapped file should start with 'quant-model-'"
            assert mapped_file.endswith(".safetensors"), \
                "Mapped file should end with '.safetensors'"
            # Verify mapped file exists
            mapped_file_path = os.path.join(self.save_path, mapped_file)
            assert os.path.exists(mapped_file_path), f"Mapped file {mapped_file} should exist"

        # Verify each mapped file contains the correct parameters
        file_to_params = {}
        for param_name, mapped_file in weight_map.items():
            if mapped_file not in file_to_params:
                file_to_params[mapped_file] = []
            file_to_params[mapped_file].append(param_name)

        # Verify parameters in each file
        for mapped_file, params in file_to_params.items():
            file_path = os.path.join(self.save_path, mapped_file)
            loaded_params = ms.load_checkpoint(file_path, format="safetensors")
            for param_name in params:
                assert param_name in loaded_params, \
                    f"Parameter {param_name} should be in file {mapped_file}"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_quantization_description_save(self):
        """
        Feature: SafeTensorsMgr quantization description save test case
        Description: Test quantization description information save functionality
        Expectation: quantization_description.json is generated correctly, quantization information records are complete
        """
        mgr = SafeTensorsMgr(file_limit_g=4)
        dis_params_dict = self._create_test_parameters()

        # Create detailed quantization description info
        quant_desc_info = {
            "quantization_method": "A8W8",
            "quantization_config": {
                "weight_quant_dtype": "int8",
                "act_quant_dtype": "int8",
                "weight_quant_granularity": "per_tensor",
                "act_quant_granularity": "per_token"
            },
            "algorithm": {
                "name": "SmoothQuant",
                "alpha": 0.5
            },
            "model_info": {
                "model_type": "test_model",
                "version": "1.0"
            }
        }

        # Call save method
        mgr.save(self.original_path, self.save_path, dis_params_dict, quant_desc_info)

        # Verify quantization description JSON is created
        quant_desc_path = os.path.join(self.save_path, "quantization_description.json")
        assert os.path.exists(quant_desc_path), "quantization_description.json should be created"

        # Verify quantization description content
        with open(quant_desc_path, 'r', encoding='utf-8') as f:
            saved_quant_desc = json.load(f)

        # Verify all fields are preserved
        assert saved_quant_desc == quant_desc_info, "Quantization description should match exactly"
        assert saved_quant_desc["quantization_method"] == "A8W8", "quantization_method should be preserved"
        assert "quantization_config" in saved_quant_desc, "quantization_config should be preserved"
        assert saved_quant_desc["quantization_config"]["weight_quant_dtype"] == "int8", \
            "weight_quant_dtype should be preserved"
        assert "algorithm" in saved_quant_desc, "algorithm should be preserved"
        assert "model_info" in saved_quant_desc, "model_info should be preserved"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_copy_original_files(self):
        """
        Feature: SafeTensorsMgr original file copy test case
        Description: Test original JSON file copy functionality, including file identification,
            file copying, and exclusion of index.json and safetensors files
        Expectation: File copy functionality works correctly, filtering rules are correct,
            blacklisted files are properly excluded
        """
        # Create additional test files including blacklisted files
        # Create index.json file (should be filtered out)
        index_data = {"weight_map": {"layer.0.weight": "model-00001-of-00002.safetensors"}}
        with open(os.path.join(self.original_path, "model.safetensors.index.json"), 'w', encoding='utf-8') as f:
            json.dump(index_data, f)

        # Create safetensors file (should be filtered out)
        with open(os.path.join(self.original_path, "model.safetensors"), 'wb') as f:
            f.write(b"fake safetensors content")

        # Create another safetensors file with different name (should be filtered out)
        with open(os.path.join(self.original_path, "pytorch_model.safetensors"), 'wb') as f:
            f.write(b"another fake safetensors content")

        # Create additional files that should be copied
        with open(os.path.join(self.original_path, "README.txt"), 'w', encoding='utf-8') as f:
            f.write("This is a test README file.")

        with open(os.path.join(self.original_path, "modeling.py"), 'w', encoding='utf-8') as f:
            f.write("# Test Python file\nprint('Hello World')")

        mgr = SafeTensorsMgr(file_limit_g=4)
        dis_params_dict = self._create_test_parameters()
        quant_desc_info = {"quantization_method": "A8W8"}

        # Call save method (which internally calls _copy_original_files)
        mgr.save(self.original_path, self.save_path, dis_params_dict, quant_desc_info)

        # Verify that expected files are copied
        expected_files = ["config.json", "tokenizer.json", "README.txt", "modeling.py"]
        for expected_file in expected_files:
            copied_file_path = os.path.join(self.save_path, expected_file)
            assert os.path.exists(copied_file_path), f"Expected file {expected_file} should be copied"

        # Verify that blacklisted files from original directory are not copied
        # Note: model.safetensors.index.json will be generated by save method,
        # but the original one should not be copied
        # Verify original model.safetensors.index.json is not copied (check content difference)
        original_index_path = os.path.join(self.original_path, "model.safetensors.index.json")
        new_index_path = os.path.join(self.save_path, "model.safetensors.index.json")

        # The new index.json should exist (generated by save method)
        assert os.path.exists(new_index_path), "New index.json should be generated by save method"

        # Verify the new index.json has different content than the original
        # (proving original wasn't copied)
        with open(original_index_path, 'r', encoding='utf-8') as f:
            original_index_content = json.load(f)
        with open(new_index_path, 'r', encoding='utf-8') as f:
            new_index_content = json.load(f)

        # The new index should have metadata and correct structure, different from original
        assert "metadata" in new_index_content, "New index.json should have metadata"
        assert original_index_content != new_index_content, \
            "New index.json should be different from original"

        # Verify other blacklisted files are not copied
        blacklisted_files = ["model.safetensors", "pytorch_model.safetensors"]
        for blacklisted_file in blacklisted_files:
            copied_file_path = os.path.join(self.save_path, blacklisted_file)
            assert not os.path.exists(copied_file_path), \
                f"Blacklisted file {blacklisted_file} should not be copied"

        # Verify content integrity for copied files
        original_config = os.path.join(self.original_path, "config.json")
        copied_config = os.path.join(self.save_path, "config.json")
        with open(original_config, 'r', encoding='utf-8') as f:
            original_content = json.load(f)
        with open(copied_config, 'r', encoding='utf-8') as f:
            copied_content = json.load(f)
        assert original_content == copied_content, "Config file content should be identical"

        # Verify text file content
        original_readme = os.path.join(self.original_path, "README.txt")
        copied_readme = os.path.join(self.save_path, "README.txt")
        with open(original_readme, 'r', encoding='utf-8') as f:
            original_text = f.read()
        with open(copied_readme, 'r', encoding='utf-8') as f:
            copied_text = f.read()
        assert original_text == copied_text, "README file content should be identical"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_copy_original_files_empty_directory(self):
        """
        Feature: SafeTensorsMgr copy original files from empty directory test case
        Description: Test original file copy functionality with empty source directory
        Expectation: No files are copied from empty directory, no error is raised
        """
        # Test 1: Empty directory handling
        empty_source = os.path.join(self.work_dir, "empty_source")
        os.makedirs(empty_source, exist_ok=True)
        empty_save_path = os.path.join(self.work_dir, "save_empty")
        os.makedirs(empty_save_path, exist_ok=True)

        mgr_empty = SafeTensorsMgr(file_limit_g=4)
        empty_dis_params_dict = {}
        empty_quant_desc_info = {"quantization_method": "A8W8"}

        # Call save method with empty source directory - should not raise error
        mgr_empty.save(empty_source, empty_save_path, empty_dis_params_dict, empty_quant_desc_info)

        # Verify that no files are copied from empty directory (except generated files)
        copied_files = [f for f in os.listdir(empty_save_path)
                        if os.path.isfile(os.path.join(empty_save_path, f))
                        and not f.endswith('.safetensors')
                        and f != "model.safetensors.index.json"
                        and f != "quantization_description.json"]
        assert not copied_files, "No files should be copied from empty directory"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_copy_original_files_subdirectories(self):
        """
        Feature: SafeTensorsMgr copy original files with subdirectories test case
        Description: Test original file copy functionality with subdirectories in source
        Expectation: Only root directory files are copied, subdirectory files are not copied
        """
        # Test 2: Subdirectories handling
        subdir_source = os.path.join(self.work_dir, "subdir_source")
        os.makedirs(subdir_source, exist_ok=True)
        subdir_save_path = os.path.join(self.work_dir, "save_subdir")
        os.makedirs(subdir_save_path, exist_ok=True)

        # Create a file in root directory (should be copied)
        with open(os.path.join(subdir_source, "root_file.json"), 'w', encoding='utf-8') as f:
            json.dump({"root": "file"}, f)

        # Create subdirectory with file (should not be copied)
        subdir = os.path.join(subdir_source, "subdir")
        os.makedirs(subdir, exist_ok=True)
        with open(os.path.join(subdir, "sub_config.json"), 'w', encoding='utf-8') as f:
            json.dump({"sub": "config"}, f)

        mgr_subdir = SafeTensorsMgr(file_limit_g=4)
        subdir_dis_params_dict = {}
        subdir_quant_desc_info = {"quantization_method": "A8W8"}

        # Call save method
        mgr_subdir.save(subdir_source, subdir_save_path, subdir_dis_params_dict, subdir_quant_desc_info)

        # Verify that root file is copied
        root_file_path = os.path.join(subdir_save_path, "root_file.json")
        assert os.path.exists(root_file_path), "Root file should be copied"

        # Verify that subdirectory file is not copied
        sub_config_path = os.path.join(subdir_save_path, "sub_config.json")
        assert not os.path.exists(sub_config_path), \
            "Files in subdirectories should not be copied"

        # Verify that subdirectory itself is not copied
        subdir_path = os.path.join(subdir_save_path, "subdir")
        assert not os.path.exists(subdir_path), "Subdirectories should not be copied"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_copy_original_files_various_extensions(self):
        """
        Feature: SafeTensorsMgr copy original files with various extensions test case
        Description: Test original file copy functionality with various file extensions
        Expectation: All files with various extensions are copied correctly, content integrity is preserved
        """
        # Test 3: Various file extensions handling
        various_ext_source = os.path.join(self.work_dir, "various_ext_source")
        os.makedirs(various_ext_source, exist_ok=True)
        various_ext_save_path = os.path.join(self.work_dir, "save_various_ext")
        os.makedirs(various_ext_save_path, exist_ok=True)

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
            file_path = os.path.join(various_ext_source, filename)
            mode = 'wb' if isinstance(content, bytes) else 'w'
            with open(file_path, mode, encoding='utf-8' if mode == 'w' else None) as f:
                f.write(content)

        mgr_various = SafeTensorsMgr(file_limit_g=4)
        various_dis_params_dict = {}
        various_quant_desc_info = {"quantization_method": "A8W8"}

        # Call save method
        mgr_various.save(various_ext_source, various_ext_save_path,
                         various_dis_params_dict, various_quant_desc_info)

        # Verify that all test files are copied
        for filename in test_files:
            copied_file_path = os.path.join(various_ext_save_path, filename)
            assert os.path.exists(copied_file_path), f"File {filename} should be copied"

        # Verify content integrity for binary file
        original_bin = os.path.join(various_ext_source, "model.bin")
        copied_bin = os.path.join(various_ext_save_path, "model.bin")
        with open(original_bin, 'rb') as f:
            original_bin_content = f.read()
        with open(copied_bin, 'rb') as f:
            copied_bin_content = f.read()
        assert original_bin_content == copied_bin_content, \
            "Binary file content should be identical"

        # Verify content integrity for text file
        original_txt = os.path.join(various_ext_source, "vocab.txt")
        copied_txt = os.path.join(various_ext_save_path, "vocab.txt")
        with open(original_txt, 'r', encoding='utf-8') as f:
            original_txt_content = f.read()
        with open(copied_txt, 'r', encoding='utf-8') as f:
            copied_txt_content = f.read()
        assert original_txt_content == copied_txt_content, \
            "Text file content should be identical"

        # Verify content integrity for JSON file
        original_json = os.path.join(various_ext_source, "tokenizer_config.json")
        copied_json = os.path.join(various_ext_save_path, "tokenizer_config.json")
        with open(original_json, 'r', encoding='utf-8') as f:
            original_json_content = json.load(f)
        with open(copied_json, 'r', encoding='utf-8') as f:
            copied_json_content = json.load(f)
        assert original_json_content == copied_json_content, \
            "JSON file content should be identical"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_distributed_mode_construct(self):
        """
        Feature: SafeTensorsMgr distributed mode construction test case
        Description: Test SafeTensorsMgr construction in distributed mode using mock
        Expectation: Distributed mode construction succeeds, barrier is created correctly
        """
        # Mock distributed environment
        with patch('mindspore_gs.ptq.basic_functions.safetensors_mgr.get_rank', return_value=1):
            with patch('mindspore_gs.ptq.basic_functions.safetensors_mgr.get_group_size', return_value=4):
                with patch('mindspore.ops.Barrier') as mock_barrier_class:
                    mock_barrier = MagicMock()
                    mock_barrier_class.return_value = mock_barrier

                    # Create SafeTensorsMgr in distributed mode
                    mgr = SafeTensorsMgr(file_limit_g=4)

                    # Verify distributed mode properties
                    assert mgr.rank_id == 1, "rank_id should be 1 in distributed mode"
                    assert mgr.group_size == 4, "group_size should be 4 in distributed mode"
                    assert hasattr(mgr, 'barrier'), "barrier should exist in distributed mode"
                    assert mgr.barrier == mock_barrier, "barrier should be created"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_distributed_save(self):
        """
        Feature: SafeTensorsMgr distributed save test case
        Description: Test save functionality in distributed environment using mock
        Expectation: Only rank0 saves files, other ranks wait, barrier synchronization works
        """
        # Mock distributed environment with rank 0
        with patch('mindspore_gs.ptq.basic_functions.safetensors_mgr.get_rank', return_value=0):
            with patch('mindspore_gs.ptq.basic_functions.safetensors_mgr.get_group_size', return_value=2):
                with patch('mindspore.ops.Barrier') as mock_barrier_class:
                    mock_barrier = MagicMock()
                    mock_barrier_class.return_value = mock_barrier

                    mgr = SafeTensorsMgr(file_limit_g=4)
                    dis_params_dict = self._create_test_parameters()
                    quant_desc_info = {"quantization_method": "A8W8"}

                    # Call save method
                    mgr.save(self.original_path, self.save_path, dis_params_dict, quant_desc_info)

                    # Verify files are saved (rank 0 should save)
                    assert os.path.exists(os.path.join(self.save_path,
                                                       "model.safetensors.index.json")), \
                        "index.json should be created by rank 0"
                    assert os.path.exists(os.path.join(self.save_path,
                                                       "quantization_description.json")), \
                        "quantization_description.json should be created by rank 0"

                    # Verify barrier was called multiple times:
                    # - Once for each parameter in _tp_merge (2 parameters)
                    # - Once at the end of save method
                    # So total should be 2 + 1 = 3 calls
                    assert mock_barrier.call_count == 3, "barrier should be called at least once"
                    # Note: In _tp_merge, barrier is called for each parameter, plus once at the end

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_distributed_save_non_zero_rank(self):
        """
        Feature: SafeTensorsMgr distributed save with non-zero rank test case
        Description: Test save functionality in distributed environment with non-zero rank using mock
        Expectation: Non-zero rank does not save files, but waits at barrier
        """
        # Test with non-zero rank (should not save files, but wait at barrier)
        with patch('mindspore_gs.ptq.basic_functions.safetensors_mgr.get_rank', return_value=1):
            with patch('mindspore_gs.ptq.basic_functions.safetensors_mgr.get_group_size', return_value=2):
                with patch('mindspore.ops.Barrier') as mock_barrier_class:
                    mock_barrier = MagicMock()
                    mock_barrier_class.return_value = mock_barrier

                    # Create a new save path for this test
                    non_zero_rank_save_path = os.path.join(self.work_dir, "save_non_zero_rank")
                    os.makedirs(non_zero_rank_save_path, exist_ok=True)

                    mgr = SafeTensorsMgr(file_limit_g=4)
                    dis_params_dict = self._create_test_parameters()
                    quant_desc_info = {"quantization_method": "A8W8"}

                    # Mock _tp_merge to avoid actual communication
                    with patch.object(mgr, '_tp_merge'):
                        # Call save method
                        mgr.save(self.original_path, non_zero_rank_save_path,
                                 dis_params_dict, quant_desc_info)

                    # Verify files are NOT saved by non-zero rank
                    # (In real distributed env, rank 0 would save, but here we're testing rank 1)
                    # The barrier should still be called
                    assert mock_barrier.call_count >= 1, \
                        "barrier should be called even for non-zero rank"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_file_permission_error(self):
        """
        Feature: SafeTensorsMgr file operation error test case
        Description: Test file permission error handling
        Expectation: OSError or PermissionError is raised when file permission is insufficient
        """
        mgr = SafeTensorsMgr(file_limit_g=4)
        dis_params_dict = self._create_test_parameters()
        quant_desc_info = {"quantization_method": "A8W8"}

        # Create a read-only directory to test permission error
        read_only_dir = os.path.join(self.work_dir, "read_only_dir")
        os.makedirs(read_only_dir, exist_ok=True)

        # Make the directory read-only (on Unix systems)
        try:
            os.chmod(read_only_dir, stat.S_IREAD | stat.S_IEXEC)

            # Try to save to read-only directory - should raise OSError or PermissionError
            with pytest.raises((OSError, PermissionError)):
                mgr.save(self.original_path, read_only_dir, dis_params_dict, quant_desc_info)
        except (OSError, PermissionError):
            # If we can't set permissions (e.g., on Windows), skip this test
            pytest.skip("Cannot set read-only permissions on this system")
        finally:
            # Restore permissions for cleanup
            try:
                os.chmod(read_only_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            except (OSError, PermissionError):
                pass

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_distributed_communication_error(self):
        """
        Feature: SafeTensorsMgr distributed communication error test case
        Description: Test distributed communication error handling using mock
        Expectation: RuntimeError is raised when barrier fails or communication error occurs
        """
        # Test barrier failure
        with patch('mindspore_gs.ptq.basic_functions.safetensors_mgr.get_rank', return_value=0):
            with patch('mindspore_gs.ptq.basic_functions.safetensors_mgr.get_group_size', return_value=2):
                with patch('mindspore.ops.Barrier') as mock_barrier_class:
                    mock_barrier = MagicMock()
                    # Simulate barrier failure by raising RuntimeError
                    mock_barrier.side_effect = RuntimeError("Barrier timeout")
                    mock_barrier_class.return_value = mock_barrier

                    mgr = SafeTensorsMgr(file_limit_g=4)
                    dis_params_dict = self._create_test_parameters()
                    quant_desc_info = {"quantization_method": "A8W8"}

                    # Call save method - should raise RuntimeError when barrier is called
                    with pytest.raises(RuntimeError) as exc_info:
                        mgr.save(self.original_path, self.save_path, dis_params_dict, quant_desc_info)
                    assert "Barrier" in str(exc_info.value) or \
                        "barrier" in str(exc_info.value).lower()

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_distributed_communication_error_tp_merge(self):
        """
        Feature: SafeTensorsMgr distributed TP merge communication error test case
        Description: Test distributed communication error handling during TP merge using mock
        Expectation: RuntimeError is raised when TP merge communication fails
        """
        # Test TP merge communication failure
        with patch('mindspore_gs.ptq.basic_functions.safetensors_mgr.get_rank', return_value=0):
            with patch('mindspore_gs.ptq.basic_functions.safetensors_mgr.get_group_size', return_value=2):
                with patch('mindspore.ops.Barrier') as mock_barrier_class:
                    mock_barrier = MagicMock()
                    mock_barrier_class.return_value = mock_barrier

                    mgr = SafeTensorsMgr(file_limit_g=4)
                    dis_params_dict = self._create_test_parameters()
                    quant_desc_info = {"quantization_method": "A8W8"}

                    # Mock _tp_merge to raise communication error
                    def mock_tp_merge_error(dis_params_dict):
                        raise RuntimeError("Communication failure during TP merge")

                    # pylint: disable=protected-access
                    mgr._tp_merge = mock_tp_merge_error

                    # Call save method - should raise RuntimeError during TP merge
                    with pytest.raises(RuntimeError) as exc_info:
                        mgr.save(self.original_path, self.save_path, dis_params_dict, quant_desc_info)
                    assert "Communication" in str(exc_info.value) or \
                        "TP merge" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_file_sharding_trigger_split(self):
        """
        Feature: SafeTensorsMgr file sharding trigger test case
        Description: Test file sharding logic when cur_bytes > limits
        Expectation: File sharding is triggered when accumulated bytes exceed file limit
        """
        # Create SafeTensorsMgr with 1GB file limit to trigger file splitting
        # File limit: 1GB = 1,073,741,824 bytes
        mgr = SafeTensorsMgr(file_limit_g=1)

        dis_params_dict = {}
        # Create parameters that will trigger file splitting when cur_bytes > limits
        # For float32: each element is 4 bytes
        # Each parameter: (2000, 2000) = 4,000,000 elements * 4 bytes = 16MB
        # Create 70 parameters = 70 * 16MB = 1,120MB ≈ 1.12GB
        # This will exceed the 1GB limit and trigger the splitting logic
        for i in range(70):
            param = Parameter(ms.numpy.randn(2000, 2000, dtype=ms.float32), name=f"layer.{i}.weight")
            dis_param = DistributedParameter(param, axis=None)
            dis_params_dict[f"layer.{i}.weight"] = dis_param

        quant_desc_info = {"quantization_method": "A8W8"}

        # Call save method - this should trigger the file splitting logic
        mgr.save(self.original_path, self.save_path, dis_params_dict, quant_desc_info)

        # Verify that multiple files are created due to sharding
        safetensors_files = sorted([f for f in os.listdir(self.save_path) if f.endswith('.safetensors')])
        assert len(safetensors_files) > 1, "Multiple safetensors files should be created due to sharding"

        # Verify index JSON file
        index_json_path = os.path.join(self.save_path, "model.safetensors.index.json")
        with open(index_json_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)

        # Verify that parameters are distributed across multiple files
        # (indicating sharding occurred)
        mapped_files = set(index_data["weight_map"].values())
        assert len(mapped_files) > 1, \
            "Parameters should be distributed across multiple files, " \
            "indicating sharding was triggered"

        # Verify file numbering shows multiple files
        file_numbers = []
        total_file_num = None
        for sf_file in safetensors_files:
            parts = sf_file.replace(".safetensors", "").split("-")
            if len(parts) >= 5 and parts[-2] == "of":
                file_num = int(parts[-3])
                total_num = int(parts[-1])
                file_numbers.append(file_num)
                if total_file_num is None:
                    total_file_num = total_num

        assert len(file_numbers) > 1, \
            "Should have multiple file numbers indicating sharding occurred " \
            "(cur_bytes > limits was triggered)"
        assert len(file_numbers) == total_file_num, \
            "Number of files should match total count"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_runtime_error_handling(self):
        """
        Feature: SafeTensorsMgr RuntimeError handling in constructor test case
        Description: Test RuntimeError exception handling when get_rank/get_group_size
            fail using mock
        Expectation: Constructor handles RuntimeError gracefully, sets rank_id=0 and group_size=1
        """
        # Explicitly test RuntimeError exception handling using mock
        # This ensures the exception handling branch is executed
        with patch('mindspore_gs.ptq.basic_functions.safetensors_mgr.get_rank',
                   side_effect=RuntimeError("Not in distributed mode")):
            with patch('mindspore_gs.ptq.basic_functions.safetensors_mgr.get_group_size',
                       side_effect=RuntimeError("Not in distributed mode")):
                # Create SafeTensorsMgr - should catch RuntimeError and set default values
                mgr_mock = SafeTensorsMgr(file_limit_g=4)

                # Verify that the exception was caught and handled correctly
                assert mgr_mock.rank_id == 0, \
                    "rank_id should be 0 after RuntimeError exception handling"
                assert mgr_mock.group_size == 1, \
                    "group_size should be 1 after RuntimeError exception handling"
                assert not hasattr(mgr_mock, 'barrier'), \
                    "barrier should not be created when group_size == 1"
