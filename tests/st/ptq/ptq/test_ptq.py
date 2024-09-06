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
import time

import pytest
import numpy as np

from mindspore import set_context, context, nn, Tensor, dtype, GRAPH_MODE
from mindformers.modules import Linear
from mindspore_gs.ptq.ptq import PTQ
from mindspore_gs.ptq import PTQConfig, PTQMode, OutliersSuppressionType
from mindspore_gs.ptq.network_helpers.network_helper import LayerInfo


class SimpleNet(nn.Cell):
    """
    Network with single linear to be quant
    """
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = Linear(in_channels=5, out_channels=6, weight_init="ones")

    def construct(self, x):
        return self.linear(x)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", ["Ascend", "CPU"])
def test_input_catcher(device):
    """
    Feature: InputCatcher.
    Description: Apply InputCatcher on SimpleNet and check if inputs being caught correctly.
    Expectation: Inputs being caught correctly.
    """
    from mindspore_gs.ptq.ptq.quant import InputCatcher
    context.set_context(mode=context.PYNATIVE_MODE, device_target=device, max_device_memory="8GB")

    net = SimpleNet()
    foo_input = Tensor(np.ones((1, 3), dtype=np.float16))
    catcher = InputCatcher(net.linear)
    net.linear = catcher

    try:
        net(foo_input)
    except GeneratorExit:
        pass
    try:
        net(foo_input)
    except GeneratorExit:
        pass
    assert len(catcher.args) == 2
    assert len(catcher.kwargs) == 2

    for i in range(2):
        assert isinstance(catcher.args[i], list)
        assert len(catcher.args[i]) == 1
        assert isinstance(catcher.args[i][0], Tensor)
        assert catcher.args[i][0].shape == (1, 3)
        assert catcher.args[i][0].dtype == dtype.float16

        assert isinstance(catcher.kwargs[i], dict)
        assert not catcher.kwargs[i]
    os.system("ps -u | grep 'test_input_catcher' | grep -v grep | awk -F ' ' '{print$2}' | xargs kill -9")
    time.sleep(1.0)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ptq_config_error():
    """
    Feature: simulated PTQ _ptq_config_check function.
    Description: Feed invalid value of PTQConfig to _ptq_config_check function.
    Expectation: Except ValueError.
    """
    set_context(device_target="CPU")
    config = PTQConfig(act_quant_dtype=dtype.int8, weight_quant_dtype=None)
    with pytest.raises(ValueError):
        _ = PTQ(config)

    config = PTQConfig(act_quant_dtype=dtype.int8, weight_quant_dtype=None,
                       kvcache_quant_dtype=dtype.int8)
    with pytest.raises(ValueError):
        _ = PTQ(config)

    config = PTQConfig(act_quant_dtype=dtype.int8, weight_quant_dtype=None,
                       outliers_suppression=OutliersSuppressionType.SMOOTH)
    with pytest.raises(ValueError):
        _ = PTQ(config)

    config = PTQConfig(act_quant_dtype=dtype.int8, weight_quant_dtype=None,
                       kvcache_quant_dtype=dtype.int8,
                       outliers_suppression=OutliersSuppressionType.SMOOTH)
    with pytest.raises(ValueError):
        _ = PTQ(config)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_context_mode_error():
    """
    Feature: set GRAPH mode to QUANTIZE.
    Description:set GRAPH mode to QUANTIZE when using PTQ alogrith to quant network.
    Expectation: Except ValueError.
    """
    from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper
    config = PTQConfig(mode=PTQMode.QUANTIZE, act_quant_dtype=dtype.int8,
                       outliers_suppression=OutliersSuppressionType.SMOOTH)
    ptq = PTQ(config)

    cur_dir, _ = os.path.split(os.path.abspath(__file__))
    config_path_ = os.path.join(cur_dir, "../../../data/test_llama2/predict_llama2_13b_fp16_910b_1p.yaml")
    helper = MFLlama2Helper(config_path_)
    set_context(mode=GRAPH_MODE, device_target='Ascend', jit_config={"jit_level": "O0", "infer_boost": "on"},
                max_device_memory=helper.mf_config.context.max_device_memory)
    network = helper.create_network()
    with pytest.raises(ValueError, match="Quantization phase only support PYNATIVE MODE."):
        ptq.apply(network, helper)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_layer_info_error():
    """
    Feature: test LayerInfo class.
    Description: Feed invalid param to LayerInfo to raise type error.
    Expectation: Except ValueError.
    """
    set_context(device_target="CPU")
    with pytest.raises(TypeError):
        _ = LayerInfo(name=1)

    with pytest.raises(TypeError):
        _ = LayerInfo(layer="1")

    with pytest.raises(TypeError):
        _ = LayerInfo(type_="1")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_algo", ['A8W8'])
def test_ptq_llama2_predict_2stage_1p_run_a8w8(quant_algo):
    """
    Feature: test omni quant adjust parameter in two stages with one cards.
    Description: apply OQ on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptq_network_runner.py")
    return_code = os.system(
        f"msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 "
        "--master_port=10926 --join=True --log_dir=./test_ptq_A8W8_predict_llama2_1p_logs "
        f"python {run_file} -m 1 -a {quant_algo}"
    )
    if return_code != 0:
        log_file = open("./test_ptq_A8W8_predict_llama2_1p_logs/worker_0.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()
    os.system("ps -u | grep 'ptq_network_runner' | grep -v grep | awk -F ' ' '{print$2}' | xargs kill -9")
    os.system("kill -9 $(lsof -i:10926 | awk '{print $2}')")
    time.sleep(1.0)
    assert return_code == 0


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_algo", ['A16W8'])
def test_ptq_llama2_predict_2stage_1p_run_a16w8(quant_algo):
    """
    Feature: test omni quant adjust parameter in two stages with one cards.
    Description: apply OQ on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptq_network_runner.py")
    return_code = os.system(
        f"msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 "
        "--master_port=10926 --join=True --log_dir=./test_ptq_A16W8_predict_llama2_1p_logs "
        f"python {run_file} -m 1 -a {quant_algo}"
    )
    if return_code != 0:
        log_file = open("./test_ptq_A16W8_predict_llama2_1p_logs/worker_0.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()
    os.system("ps -u | grep 'ptq_network_runner' | grep -v grep | awk -F ' ' '{print$2}' | xargs kill -9")
    os.system("kill -9 $(lsof -i:10926 | awk '{print $2}')")
    time.sleep(1.0)
    assert return_code == 0


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_algo", ['A8W8C8'])
def test_ptq_llama2_predict_2stage_1p_run_a8w8c8(quant_algo):
    """
    Feature: test omni quant adjust parameter in two stages with one cards.
    Description: apply OQ on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptq_network_runner.py")
    return_code = os.system(
        f"msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 "
        "--master_port=10926 --join=True --log_dir=./test_ptq_A8W8C8_predict_llama2_1p_logs "
        f"python {run_file} -m 1 -a {quant_algo}"
    )
    if return_code != 0:
        log_file = open("./test_ptq_A8W8C8_predict_llama2_1p_logs/worker_0.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()
    os.system("ps -u | grep 'ptq_network_runner' | grep -v grep | awk -F ' ' '{print$2}' | xargs kill -9")
    os.system("kill -9 $(lsof -i:10926 | awk '{print $2}')")
    time.sleep(1.0)
    assert return_code == 0


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_algo", ['A16W8C8'])
def test_ptq_llama2_predict_2stage_1p_run_a16w8c8(quant_algo):
    """
    Feature: test omni quant adjust parameter in two stages with one cards.
    Description: apply OQ on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptq_network_runner.py")
    return_code = os.system(
        f"msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 "
        "--master_port=10926 --join=True --log_dir=./test_ptq_A16W8C8_predict_llama2_1p_logs "
        f"python {run_file} -m 1 -a {quant_algo}"
    )
    if return_code != 0:
        log_file = open("./test_ptq_A16W8C8_predict_llama2_1p_logs/worker_0.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()
    os.system("ps -u | grep 'ptq_network_runner' | grep -v grep | awk -F ' ' '{print$2}' | xargs kill -9")
    os.system("kill -9 $(lsof -i:10926 | awk '{print $2}')")
    time.sleep(1.0)
    assert return_code == 0


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_algo", ['C8'])
def test_ptq_llama2_predict_2stage_1p_run_c8(quant_algo):
    """
    Feature: test omni quant adjust parameter in two stages with one cards.
    Description: apply OQ on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptq_network_runner.py")
    return_code = os.system(
        f"msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 "
        "--master_port=10926 --join=True --log_dir=./test_ptq_C8_predict_llama2_1p_logs "
        f"python {run_file} -m 1 -a {quant_algo}"
    )
    if return_code != 0:
        log_file = open("./test_ptq_C8_predict_llama2_1p_logs/worker_0.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()
    os.system("ps -u | grep 'ptq_network_runner' | grep -v grep | awk -F ' ' '{print$2}' | xargs kill -9")
    os.system("kill -9 $(lsof -i:10926 | awk '{print $2}')")
    time.sleep(1.0)
    assert return_code == 0


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
@pytest.mark.parametrize("quant_algo", ['A8W8'])
def test_ptq_llama2_predict_2stage_2p_run_a8w8(quant_algo):
    """
    Feature: test omni quant adjust parameter in two stages with two cards.
    Description: apply OQ on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    os.environ['quant_algo'] = f"{quant_algo}"
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptq_network_runner.py")
    return_code = os.system(
        f"msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 "
        "--master_port=10926 --join=True --log_dir=./test_ptq_A8W8_predict_llama2_2p_logs "
        f"python {run_file} -m 2 -a {quant_algo}"
    )
    if return_code != 0:
        log_file = open("./test_ptq_A8W8_predict_llama2_2p_logs/worker_0.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()
    os.system("ps -u | grep 'ptq_network_runner' | grep -v grep | awk -F ' ' '{print$2}' | xargs kill -9")
    os.system("kill -9 $(lsof -i:10926 | awk '{print $2}')")
    time.sleep(1.0)
    assert return_code == 0


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
@pytest.mark.parametrize("quant_algo", ['A16W8'])
def test_ptq_llama2_predict_2stage_2p_run_a16w8(quant_algo):
    """
    Feature: test omni quant adjust parameter in two stages with two cards.
    Description: apply OQ on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    os.environ['quant_algo'] = f"{quant_algo}"
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptq_network_runner.py")
    return_code = os.system(
        f"msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 "
        "--master_port=10926 --join=True --log_dir=./test_ptq_A16W8_predict_llama2_2p_logs "
        f"python {run_file} -m 2 -a {quant_algo}"
    )
    if return_code != 0:
        log_file = open("./test_ptq_A16W8_predict_llama2_2p_logs/worker_0.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()
    os.system("ps -u | grep 'ptq_network_runner' | grep -v grep | awk -F ' ' '{print$2}' | xargs kill -9")
    os.system("kill -9 $(lsof -i:10926 | awk '{print $2}')")
    time.sleep(1.0)
    assert return_code == 0


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
@pytest.mark.parametrize("quant_algo", ['A8W8C8'])
def test_ptq_llama2_predict_2stage_2p_run_a8w8c8(quant_algo):
    """
    Feature: test omni quant adjust parameter in two stages with two cards.
    Description: apply OQ on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    os.environ['quant_algo'] = f"{quant_algo}"
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptq_network_runner.py")
    return_code = os.system(
        f"msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 "
        "--master_port=10926 --join=True --log_dir=./test_ptq_A8W8C8_predict_llama2_2p_logs "
        f"python {run_file} -m 2 -a {quant_algo}"
    )
    if return_code != 0:
        log_file = open("./test_ptq_A8W8C8_predict_llama2_2p_logs/worker_0.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()
    os.system("ps -u | grep 'ptq_network_runner' | grep -v grep | awk -F ' ' '{print$2}' | xargs kill -9")
    os.system("kill -9 $(lsof -i:10926 | awk '{print $2}')")
    time.sleep(1.0)
    assert return_code == 0


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
@pytest.mark.parametrize("quant_algo", ['A16W8C8'])
def test_ptq_llama2_predict_2stage_2p_run_a16w8c8(quant_algo):
    """
    Feature: test omni quant adjust parameter in two stages with two cards.
    Description: apply OQ on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    os.environ['quant_algo'] = f"{quant_algo}"
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptq_network_runner.py")
    return_code = os.system(
        f"msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 "
        "--master_port=10926 --join=True --log_dir=./test_ptq_A16W8C8_predict_llama2_2p_logs "
        f"python {run_file} -m 2 -a {quant_algo}"
    )
    if return_code != 0:
        log_file = open("./test_ptq_A16W8C8_predict_llama2_2p_logs/worker_0.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()
    os.system("ps -u | grep 'ptq_network_runner' | grep -v grep | awk -F ' ' '{print$2}' | xargs kill -9")
    os.system("kill -9 $(lsof -i:10926 | awk '{print $2}')")
    time.sleep(1.0)
    assert return_code == 0


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
@pytest.mark.parametrize("quant_algo", ['C8'])
def test_ptq_llama2_predict_2stage_2p_run_c8(quant_algo):
    """
    Feature: test omni quant adjust parameter in two stages with two cards.
    Description: apply OQ on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    os.environ['quant_algo'] = f"{quant_algo}"
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptq_network_runner.py")
    return_code = os.system(
        f"msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 "
        "--master_port=10926 --join=True --log_dir=./test_ptq_C8_predict_llama2_2p_logs "
        f"python {run_file} -m 2 -a {quant_algo}"
    )
    if return_code != 0:
        log_file = open("./test_ptq_C8_predict_llama2_2p_logs/worker_0.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()
    os.system("ps -u | grep 'ptq_network_runner' | grep -v grep | awk -F ' ' '{print$2}' | xargs kill -9")
    os.system("kill -9 $(lsof -i:10926 | awk '{print $2}')")
    time.sleep(1.0)
    assert return_code == 0


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
@pytest.mark.parametrize("quant_algo", ['A8W8_FallBack'])
def test_ptq_llama2_predict_2stage_2p_run_a8w8_fallback(quant_algo):
    """
    Feature: test omni quant adjust parameter in two stages with two cards.
    Description: apply OQ on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    os.environ['quant_algo'] = f"{quant_algo}"
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptq_network_runner.py")
    return_code = os.system(
        f"msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 "
        "--master_port=10926 --join=True --log_dir=./test_ptq_A8W8_FallBack_predict_llama2_2p_logs "
        f"python {run_file} -m 2 -a {quant_algo}"
    )
    if return_code != 0:
        log_file = open("./test_ptq_A8W8_FallBack_predict_llama2_2p_logs/worker_0.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()
    os.system("ps -u | grep 'ptq_network_runner' | grep -v grep | awk -F ' ' '{print$2}' | xargs kill -9")
    os.system("kill -9 $(lsof -i:10926 | awk '{print $2}')")
    time.sleep(1.0)
    assert return_code == 0
