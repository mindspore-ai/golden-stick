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
"""Test suite for PTQ models."""

import os
import sys
import pytest

# Add the path to import mindspore_gs modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_ptq_models_suite():
    """
    Feature: PTQ Models Test Suite.
    Description: Run all PTQ models tests to verify implementation correctness.
    Expectation: All tests pass successfully.
    """
    # Run all test files we created
    test_files = [
        "test_auto_model.py",
        "test_base_model.py",
        "test_mf_model.py",
        "test_qwen3.py",
        "test_qwen3_moe.py",
        "test_deepseekv3.py"
    ]

    # Get the directory of this file
    test_dir = os.path.dirname(os.path.abspath(__file__))

    # Run each test file
    for test_file in test_files:
        file_path = os.path.join(test_dir, test_file)
        if os.path.exists(file_path):
            print(f"Running tests in {test_file}...")
            # Run the test file
            exit_code = pytest.main(["-v", file_path])
            if exit_code != 0:
                raise RuntimeError(f"Tests in {test_file} failed with exit code {exit_code}")
            print(f"Finished running tests in {test_file}.\n")

    print("All PTQ model tests passed!")


if __name__ == "__main__":
    test_ptq_models_suite()
