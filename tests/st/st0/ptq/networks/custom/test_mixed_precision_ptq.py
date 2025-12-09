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
"""36-layer mixed precision network PTQ test case.
Standard Linear: 24 layers, GroupedLinear: 12 layers.
"""
from pathlib import Path
import os
import subprocess
import pytest

os.environ['GSLOG'] = "1"


def build_command_list(run_script_path):
    """Build the command list for running the test script"""
    cmd_list = ["python", str(run_script_path)]
    return cmd_list


class TestMixedPrecisionPTQ:
    """Test class for mixed precision network PTQ with different configurations"""

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_mindformers_script_path = self.sh_path / "custom_mindformers/run_mixed_precision_ptq.py"
        self.run_mindone_script_path = self.sh_path / "custom_mindone/run_mixed_precision_ptq.py"

    def infer(self, run_script_path, show_logs=True):
        """Run inference with the specified parameters and check for output file"""
        cmd_list = build_command_list(
            run_script_path=run_script_path,
        )

        # Preserve PYTHONPATH environment variable
        env = os.environ.copy()

        if show_logs:
            # Run with real-time output
            result = subprocess.run(
                cmd_list, shell=False, check=False, env=env)
        else:
            # Run with captured output for pytest
            result = subprocess.run(
                cmd_list, shell=False, capture_output=True, text=True, check=False, env=env)

        assert result.returncode == 0, (
            f"Test script failed with non-zero exit code: {result.returncode}."
            if show_logs else
            f"Test script failed with non-zero exit code: "
            f"{result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
        )

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_mf_single_card_mixed_precision_ptq(self):  # pylint: disable=redefined-outer-name
        """
        Feature: Quantize and evaluate 36-layer parallel mixed precision mindformers network with PTQ algorithm.
        Description: Test PTQ fake quantization inference for mixed precision network on single card.
        Expectation: Precision check should pass for all layers.
        """
        self.infer(self.run_mindformers_script_path, show_logs=False)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_mo_single_card_mixed_precision_ptq(self):  # pylint: disable=redefined-outer-name
        """
        Feature: Quantize and evaluate 54-layer parallel mixed precision mindone network with PTQ algorithm.
        Description: Test PTQ fake quantization inference for mixed precision network on single card.
        Expectation: Precision check should pass for all layers.
        """
        self.infer(self.run_mindone_script_path, show_logs=False)


if __name__ == "__main__":
    # Run test directly with real-time logs
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    test_instance = TestMixedPrecisionPTQ()
    test_instance.setup_method()

    # Test mindformers
    print("Running mindformers test...")
    test_instance.infer(
        run_script_path=test_instance.run_mindformers_script_path,
        show_logs=True
    )
    print("Mindformers test completed successfully!")

    # Test mindone
    print("Running mindone test...")
    test_instance.infer(
        run_script_path=test_instance.run_mindone_script_path,
        show_logs=True
    )
    print("Mindone test completed successfully!")

    print("All tests completed successfully!")
