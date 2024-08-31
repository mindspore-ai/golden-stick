# Copyright 2024 Huawei Technologies Co., Ltd
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
"""test interfaces of smooth quant."""
import os
import pytest


# @pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_algo", ['A8W8', 'A16W8'])
def test_ptq_llama2_predict_2stage_1p_run(quant_algo):
    """
    Feature: test omni quant adjust parameter in two stages with one cards.
    Description: apply OQ on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    os.system("kill -9 $(lsof -i:10926 | awk '{print $2}')")
    return_code = os.system(
        "msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 "
        "--master_port=10926 --join=True --log_dir=./test_ptq_predict_llama2_1p_logs "
        f"python ptq_network_runner.py -m 1 -a {quant_algo}"
    )
    if return_code != 0:
        log_file = open("./test_ptq_predict_llama2_1p_logs/worker_0.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()

    assert return_code == 0


# @pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
@pytest.mark.parametrize("quant_algo", ['A16W8'])
def test_ptq_llama2_predict_2stage_2p_run(quant_algo):
    """
    Feature: test omni quant adjust parameter in two stages with two cards.
    Description: apply OQ on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    os.system("kill -9 $(lsof -i:10926 | awk '{print $2}')")
    os.environ['quant_algo'] = f"{quant_algo}"
    return_code = os.system(
        "msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 "
        "--master_port=10926 --join=True --log_dir=./test_ptq_predict_llama2_2p_logs "
        f"python ptq_network_runner.py -m 2 -a {quant_algo}"
    )
    if return_code != 0:
        log_file = open("./test_ptq_predict_llama2_2p_logs/worker_0.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()

    assert return_code == 0
