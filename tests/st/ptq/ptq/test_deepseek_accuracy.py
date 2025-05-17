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

def ptq_predict_deepseek_4p_run(quant_algo):
    """
    Feature: test dynamic quant adjust parameter in two stages with two cards.
    Description: apply ptq on deepseek and check accuracy.
    Expectation: accuracy is good.
    """
    os.environ['quant_algo'] = f"{quant_algo}"
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptq_deepseek_runner.py")
    port = get_available_port()
    os.system(f"kill -9 $(lsof -i:{port} | " + "awk '{print $2}')")
    time.sleep(1.0)
    return_code = os.system(
        f"msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 "
        f"--master_port={port} --join=True --log_dir=./test_ptq_{quant_algo}_predict_deepseek_4p_logs "
        f"python {run_file} -m 4 -a {quant_algo}"
    )
    if return_code != 0:
        log_file = open(f"./test_ptq_{quant_algo}_predict_deepseek_4p_logs/worker_0.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()
    os.system("ps -u | grep 'ptq_deepseek_runner' | grep -v grep | awk -F ' ' '{print$2}' | xargs kill -9")
    os.system(f"kill -9 $(lsof -i:{port} | " + "awk '{print $2}')")
    time.sleep(1.0)
    assert return_code == 0


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_ptq_deepseek_smoothquant_accuracy():
    """
    Feature: test smoothquant adjust parameter in two stages with two cards.
    Description: apply smoothquant on deepseek and check accuracy.
    Expectation: accuracy is good.
    """
    ptq_predict_deepseek_4p_run("smoothquant")


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_ptq_deepseek_a16w8_accuracy():
    """
    Feature: test a16w8 adjust parameter in two stages with two cards.
    Description: apply a16w8 on deepseek and check accuracy.
    Expectation: accuracy is good.
    """
    ptq_predict_deepseek_4p_run("a16w8")
