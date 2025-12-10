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
import tempfile
import pytest
import numpy as np

from tests.st.precision_utils import PrecisionChecker

def build_command_list(run_script_path, output_path_param):
    """Build the command list for running the test script"""
    cmd_list = ["python", str(run_script_path), f"--output_path={output_path_param}"]
    return cmd_list


class TestMixedPrecisionPTQ:
    """Test class for mixed precision network PTQ with different configurations"""

    def setup_method(self):
        """Setup method to prepare test environment"""
        self.sh_path = Path(__file__).parent.resolve()
        self.run_script_path = self.sh_path / "run_mixed_precision_ptq.py"

    def infer(self, output_file_path, show_logs=True):
        """Run inference with the specified parameters and check for output file"""
        cmd_list = build_command_list(
            run_script_path=self.run_script_path,
            output_path_param=output_file_path,
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
        assert output_file_path.exists(), (
            f"Output file {output_file_path} was not created."
        )

    def run_test(self, tmp_path, show_logs=True):  # pylint: disable=redefined-outer-name
        """Helper function to run test and check results"""
        output_file_path = tmp_path / 'output.npz'
        self.infer(output_file_path=output_file_path, show_logs=show_logs)
        output = np.load(output_file_path)

        checker = PrecisionChecker(cos_sim_thd=0.99, l1_norm_thd=0.01, kl_dvg_thd=0.005)
        succeed = True
        for key in output:
            if key.endswith('_quant'):
                fp_key = key.replace('_quant', '_fp')
                if fp_key not in output:
                    raise ValueError(f"Key {fp_key} not found in output for quant key {key}")

                try:
                    checker.check_precision(output[fp_key], output[key])
                    print(f"Check precision for {key} succeed", flush=True)
                except AssertionError as e:
                    print(f"Check precision for {key} failed: {e}", flush=True)
                    succeed = False
        assert succeed, "Some precision check failed"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_single_card_mixed_precision_ptq(self, tmp_path):  # pylint: disable=redefined-outer-name
        """
        Feature: Quantize and evaluate 36-layer parallel mixed precision network with PTQ algorithm.
        Description: Test PTQ fake quantization inference for mixed precision network on single card.
        Expectation: Precision check should pass for all layers.
        """
        self.run_test(tmp_path, show_logs=False)


if __name__ == "__main__":
    # Run test directly with real-time logs
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_instance = TestMixedPrecisionPTQ()
        test_instance.setup_method()
        test_instance.run_test(tmp_path, show_logs=True)
        print("Test completed successfully!")
