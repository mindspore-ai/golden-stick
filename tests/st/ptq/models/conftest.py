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
"""Pytest configuration for models tests."""

import os
import subprocess
import sys


# pylint: disable=unused-argument
def pytest_configure(config):
    """
    Install test requirements before running tests.
    This hook is called after command line options have been parsed
    and all plugins and initial conftest files been loaded.
    """
    # Get the path to tests/requirements.txt
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate from tests/st/ptq/models to tests/
    tests_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
    requirements_path = os.path.join(tests_dir, "requirements.txt")

    if os.path.exists(requirements_path):
        print(f"Installing requirements from {requirements_path}")
        try:
            # Install requirements
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", requirements_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("Requirements installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install requirements: {e}")
            # Don't fail the test run if installation fails, just warn
    else:
        print(f"Warning: Requirements file not found at {requirements_path}")
