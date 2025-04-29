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
"""test interfaces of ptq."""
import os
import time
import pytest
from tests.st.test_utils import get_available_port


def ptq_predict_2stage_2p_run(quant_algo):
    """
    Feature: test dynamic quant adjust parameter in two stages with two cards.
    Description: apply ptq on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    os.environ['quant_algo'] = f"{quant_algo}"
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptq_llama_runner.py")
    port = get_available_port()
    os.system(f"kill -9 $(lsof -i:{port} | " + "awk '{print $2}')")
    time.sleep(1.0)
    return_code = os.system(
        f"msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 "
        f"--master_port={port} --join=True --log_dir=./test_ptq_{quant_algo}_predict_llama2_2p_logs "
        f"python {run_file} -m 2 -a {quant_algo}"
    )
    if return_code != 0:
        log_file = open(f"./test_ptq_{quant_algo}_predict_llama2_2p_logs/worker_0.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()
    os.system("ps -u | grep 'ptq_network_runner' | grep -v grep | awk -F ' ' '{print$2}' | xargs kill -9")
    os.system(f"kill -9 $(lsof -i:{port} | " + "awk '{print $2}')")
    time.sleep(1.0)
    assert return_code == 0


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_ptq_llama2_predict_2stage_2p_run_a8w8c8():
    """
    Feature: test omni quant adjust parameter in two stages with two cards.
    Description: apply A8W8C8 on llama2 and check score.
    Expectation: score is good.
    """
    ptq_predict_2stage_2p_run("A8W8C8")

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_ptq_llama2_predict_2stage_2p_run_a16w8c8():
    """
    Feature: test omni quant adjust parameter in two stages with two cards.
    Description: apply A16W8C8 on llama2 and check score.
    Expectation: score is good.
    """
    ptq_predict_2stage_2p_run("A16W8C8")

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_ptq_llama2_predict_2stage_2p_run_c8():
    """
    Feature: test omni quant adjust parameter in two stages with two cards.
    Description: apply C8 on llama2 and check score.
    Expectation: score is good.
    """
    ptq_predict_2stage_2p_run("C8")

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_ptq_llama2_predict_2stage_2p_run_a8w8():
    """
    Feature: test omni quant adjust parameter in two stages with two cards.
    Description: apply A8W8 on llama2 and check score.
    Expectation: score is good.
    """
    ptq_predict_2stage_2p_run("A8W8")

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_ptq_llama2_predict_2stage_2p_run_a16w8():
    """
    Feature: test omni quant adjust parameter in two stages with two cards.
    Description: apply A16W8 on llama2 and check score.
    Expectation: score is good.
    """
    ptq_predict_2stage_2p_run("A16W8")

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_ptq_llama2_predict_2stage_2p_run_a8w8_dynamic():
    """
    Feature: test omni quant adjust parameter in two stages with two cards.
    Description: apply A8W8_Dynamic on llama2 and check score.
    Expectation: score is good.
    """
    ptq_predict_2stage_2p_run("A8W8_Dynamic")

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_ptq_llama2_predict_2stage_2p_run_c8_dynamic():
    """
    Feature: test omni quant adjust parameter in two stages with two cards.
    Description: apply C8_Dynamic on llama2 and check score.
    Expectation: score is good.
    """
    ptq_predict_2stage_2p_run("C8_Dynamic")
