#!/usr/bin/env python
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
"""Test combined models using the test scheduler."""

import os
import time
import pytest
from tests.st.ptq.task_scheduler import run_combined_tests
from tests.st.test_utils import get_available_port


@pytest.mark.level3
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_mcore_mix_quant_accuracy():
    """
    Feature: test Qwen3 and Qwen3-MoE network mix-quant accuracy in parallel.
    Description: apply mix quant policy on Qwen3 and Qwen3-MoE and check scores.
    Expectation: scores are good.
    """
    # Define test configurations
    current_dir = os.path.dirname(os.path.abspath(__file__))
    qwen3_log_path = "./test_ptq_predict_qwen3_2p_logs"
    qwen3_moe_log_path = "./test_ptq_predict_qwen3_moe_2p_logs"
    telechat2_log_path = "./test_ptq_predict_telechat2_2p_logs"
    test_configs = [
        {
            'name': 'Qwen3-0.6B',
            'script': os.path.join(current_dir, f"qwen3_accuracy_runner.py --log_path={qwen3_log_path}"),
            'num_cards': 2,
            'log_dir': qwen3_log_path
        },
        {
            'name': 'Qwen3_MoE-30B-A3B',
            'script': os.path.join(current_dir, f"qwen3_moe_accuracy_runner.py --log_path={qwen3_moe_log_path}"),
            'num_cards': 2,
            'log_dir': qwen3_moe_log_path
        },
        {
            'name': 'Telechat2-7B',
            'script': os.path.join(current_dir, f"telechat2_accuracy_runner.py --log_path={telechat2_log_path}"),
            'num_cards': 2,
            'log_dir': telechat2_log_path
        }
    ]
    # Run the tests
    failures = run_combined_tests(test_configs)
    for failure_name in failures:
        for test_config in test_configs:
            if test_config['name'] == failure_name:
                print(f"Test {failure_name} failed in {test_config['name']}")
                os.system(f"cat {os.path.join(test_config['log_dir'], 'worker_0.log')}")
    assert not failures, "Combined mcore tests failed"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_qwen3_mix_quant_accuracy():
    """
    Feature: test Qwen3 network mix-quant accuracy in parallel.
    Description: apply mix quant policy on Qwen3 and check scores.
    Expectation: scores are good.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    qwen3_log_path = os.path.join(current_dir, "./test_ptq_predict_qwen3_2p_logs")
    os.environ['HCCL_CONNECT_TIMEOUT'] = "1800"
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_accuracy_runner.py")
    port = get_available_port()
    os.system(f"kill -9 $(lsof -i:{port} | " + "awk '{print $2}')")
    time.sleep(1.0)
    return_code = os.system(
        f"msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 "
        f"--master_port={port} --join=True --log_dir={qwen3_log_path} "
        f"python {run_file} --log_path={qwen3_log_path}"
    )
    time.sleep(1.0)
    assert return_code == 0


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_qwen3_moe_mix_quant_accuracy():
    """
    Feature: test Qwen3-MoE network mix-quant accuracy in parallel.
    Description: apply mix quant policy on Qwen3-MoE and check scores.
    Expectation: scores are good.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    qwen3_moe_log_path = os.path.join(current_dir, "./test_ptq_predict_qwen3_moe_2p_logs")
    os.environ['HCCL_CONNECT_TIMEOUT'] = "1800"
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_moe_accuracy_runner.py")
    port = get_available_port()
    os.system(f"kill -9 $(lsof -i:{port} | " + "awk '{print $2}')")
    time.sleep(1.0)
    return_code = os.system(
        f"msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 "
        f"--master_port={port} --join=True --log_dir={qwen3_moe_log_path} "
        f"python {run_file} --log_path={qwen3_moe_log_path}"
    )
    time.sleep(1.0)
    assert return_code == 0


@pytest.mark.level3
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_telechat2_mix_quant_accuracy():
    """
    Feature: test Telechat2 network mix-quant accuracy in parallel.
    Description: apply mix quant policy on Telechat2 and check scores.
    Expectation: scores are good.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    telechat2_log_path = os.path.join(current_dir, "./test_ptq_predict_telechat2_2p_logs")
    os.environ['HCCL_CONNECT_TIMEOUT'] = "1800"
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telechat2_accuracy_runner.py")
    port = get_available_port()
    os.system(f"kill -9 $(lsof -i:{port} | " + "awk '{print $2}')")
    time.sleep(1.0)
    return_code = os.system(
        f"msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 "
        f"--master_port={port} --join=True --log_dir={telechat2_log_path} "
        f"python {run_file} --log_path={telechat2_log_path}"
    )
    time.sleep(1.0)
    assert return_code == 0
