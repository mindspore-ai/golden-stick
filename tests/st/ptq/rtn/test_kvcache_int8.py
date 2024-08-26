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
"""test Linear Int8 algorithm."""
import os
import sys
from collections import OrderedDict

import pytest
import numpy as np
from mindspore import dtype as msdtype
from mindspore import context, GRAPH_MODE, Tensor, nn, save_checkpoint, load_checkpoint
from mindspore.communication import get_rank
from mindspore_gs.ptq import RoundToNearest as RTN
from mindspore_gs.ptq.convert_utils import QuantCellV2
from mindspore_gs.ptq.round_to_nearest.quant_cells.mindformers.quant_cells import PagedAttentionQuant
from mindspore_gs.ptq.fake_quantizer import MinMaxPerChannel
from mindspore_gs.ptq.ptq_config import PTQConfig, PTQMode
from mindspore_gs.common.gs_enum import BackendTarget

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
# pylint: disable=wrong-import-position
from tests.st.test_utils import relative_tolerance_acceptable, \
    set_config, load_distribut_checkpoint, MFLlama2HelloNetworkHelper, create_hello_ds
from mindformers.models.llama.llama_tokenizer import LlamaTokenizer
from mindformers.modules import PagedAttentionMgr
from mindformers import LlamaForCausalLM


class SimpleNet(nn.Cell):
    """
    Network with single linear to be quant
    """

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.kv_cache = PagedAttentionMgr(2, 12, 2, (256, 1, 2, 12))

    def construct(self, key, value, slot_mapping):
        return self.kv_cache(key, value, slot_mapping)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", ["Ascend"])
@pytest.mark.parametrize("mode", [GRAPH_MODE])
def test_apply_convert(device, mode):
    """
    Feature: RoundToNearestPTQ algorithm set functions.
    Description: Apply RoundToNearestPTQ on SimpleNet.
    Expectation: Apply success and coordinate attributes are same as config.
    """
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    print(f"---------------- Testing params: {device} {mode} ", flush=True)
    context.set_context(device_target=device, mode=mode, jit_config={"jit_level": "O0", "infer_boost": "on"})
    network = SimpleNet()
    ptq = RTN()
    # pylint: disable=W0212
    ptq._config.kvcache_quant_dtype = msdtype.int8
    # apply & calibrate
    new_network = ptq.apply(network)
    fakekey = np.ones((1, 1, 24), dtype=np.float16)
    fakevalue = np.ones((1, 1, 24), dtype=np.float16)
    fakeslot = np.ones((1), dtype=np.int32)
    new_network(Tensor(fakekey), Tensor(fakevalue), Tensor(fakeslot))

    cells: OrderedDict = new_network.name_cells()
    quant_cell = cells.get("kv_cache", None)
    assert isinstance(quant_cell, PagedAttentionQuant)
    assert quant_cell.weight_quantizer() is None
    # pylint: disable=W0212
    key_fake_quant: MinMaxPerChannel = quant_cell._key_input_quantizer
    assert isinstance(key_fake_quant, MinMaxPerChannel)
    assert key_fake_quant.symmetric()
    assert key_fake_quant.quant_dtype() == msdtype.int8
    assert key_fake_quant.is_per_channel()
    assert not key_fake_quant.narrow_range()
    assert key_fake_quant.num_bits() == 8

    # pylint: disable=W0212
    value_fake_quant: MinMaxPerChannel = quant_cell._value_input_quantizer
    assert isinstance(value_fake_quant, MinMaxPerChannel)
    assert value_fake_quant.symmetric()
    assert value_fake_quant.quant_dtype() == msdtype.int8
    assert value_fake_quant.is_per_channel()
    assert not value_fake_quant.narrow_range()
    assert value_fake_quant.num_bits() == 8

    quant_params = key_fake_quant.quant_params()
    min_data = np.array(quant_params.get("min"))
    max_data = np.array(quant_params.get("max"))
    assert min_data.shape == (1, 1, 24)
    assert max_data.shape == (1, 1, 24)
    for idx in range(24):
        assert min_data[0][0][idx] == 1.
        assert max_data[0][0][idx] == 1.

    # convert
    new_network = ptq.convert(new_network)
    cells: OrderedDict = new_network.name_cells()

    quant_cell = cells.get("kv_cache", None)
    assert isinstance(quant_cell, PagedAttentionQuant)

    key_fake_quant = quant_cell._key_input_quantizer
    assert isinstance(key_fake_quant, QuantCellV2)

    value_fake_quant = quant_cell._value_input_quantizer
    assert isinstance(value_fake_quant, QuantCellV2)

    assert quant_cell.weight_quantizer() is None


def kv_predict_llama2_2stage(device, mode, model_parallel, enable_deploy_fusion=False):
    """test_kv_predict_llama2_2stage"""
    os.environ['GRAPH_OP_RUN'] = "1"
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    print(f"---------------- Testing params: {device} {mode} ", flush=True)
    context.set_context(device_target=device, mode=mode, jit_config={"jit_level": "O0", "infer_boost": "on"})
    if model_parallel == 1:
        fp16_config_path = "../../../data/test_llama2/predict_llama2_13b_fp16_910b_1p.yaml"
        w8a16c8_config_path = "../../../data/test_llama2/predict_llama2_13b_fp16_910b_1p.yaml"
        fp16_ckpt_path = "../../../data/test_llama2/llama2-13b-fp16-1decoder.ckpt"
        w8a16c8_ckpt_path = "../../../data/test_llama2/llama2-13b-w8a16c8-1decoder.ckpt"
    else:
        fp16_config_path = "../../../data/test_llama2/predict_llama2_13b_fp16_910b_2p.yaml"
        w8a16c8_config_path = "../../../data/test_llama2/predict_llama2_13b_fp16_910b_2p.yaml"
        fp16_ckpt_path = "../../../data/test_llama2/llama2-13b-fp16"
        w8a16c8_ckpt_path = "../../../data/test_llama2/llama2-13b-w8a16c8"
    cur_dir, _ = os.path.split(os.path.abspath(__file__))
    tokenizer_path = os.path.join(cur_dir, "../../../data/llama2-tokenizer.model")
    fp16_config_path = os.path.join(cur_dir, fp16_config_path)
    w8a16c8_config_path = os.path.join(cur_dir, w8a16c8_config_path)
    fp16_ckpt_path = os.path.join(cur_dir, fp16_ckpt_path)
    w8a16c8_ckpt_path = os.path.join(cur_dir, w8a16c8_ckpt_path)

    def quant(ckpt_path, config_path):
        config = set_config(config_path)
        config.model.model_config.use_past = True
        config.model.model_config.use_flash_attention = True
        config.model.model_config.is_dynamic = True
        network = LlamaForCausalLM(config.model.model_config)
        tokenizer = LlamaTokenizer(vocab_file=tokenizer_path)
        if model_parallel == 1:
            load_checkpoint(ckpt_path, network)
        else:
            network = load_distribut_checkpoint(config, ckpt_path, network)
        cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, opname_blacklist=["lm_head"],
                        kvcache_quant_dtype=msdtype.int8)
        ptq = RTN(config=cfg)
        net_helper = MFLlama2HelloNetworkHelper(config)
        ds = create_hello_ds(tokenizer, 1)
        network = ptq.apply(network, net_helper, datasets=ds)
        network = ptq.convert(network)
        if model_parallel == 1:
            save_checkpoint(network.parameters_dict(), w8a16c8_ckpt_path, integrated_save=False)
        else:
            rank_id = get_rank() or 0
            save_path = os.path.join(w8a16c8_ckpt_path, f"rank_{rank_id}")
            os.makedirs(save_path, exist_ok=True)
            save_checkpoint(network.parameters_dict(), os.path.join(save_path, "w8a16c8.ckpt"),
                            choice_func=lambda x: "key_cache" not in x and "value_cache" not in x)

    def w8a16c8_infer(input_, ckpt_path, config_path):
        config = set_config(config_path)
        config.model.model_config.use_past = True
        config.model.model_config.use_flash_attention = True
        config.model.model_config.is_dynamic = True
        network = LlamaForCausalLM(config.model.model_config)
        cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, opname_blacklist=["lm_head"],
                        kvcache_quant_dtype=msdtype.int8)
        ptq = RTN(config=cfg)
        # pylint: disable=W0212
        ptq._config.enable_deploy_fusion = enable_deploy_fusion
        network = ptq.apply(network)
        network = ptq.convert(network)

        tokenizer = LlamaTokenizer(vocab_file=tokenizer_path)
        if model_parallel == 1:
            load_checkpoint(ckpt_path, network)
        else:
            network = load_distribut_checkpoint(config, ckpt_path, network)
        seq_len = 100
        input_ids = tokenizer(input_)['input_ids']
        outputs = network.generate(input_ids, do_sample=False, max_length=seq_len, top_p=1, top_k=3)
        answer = tokenizer.decode(outputs, skip_special_tokens=True)
        return outputs, answer

    def fp16_infer(input_, ckpt_path, config_path):
        config = set_config(config_path)
        network = LlamaForCausalLM(config.model.model_config)
        tokenizer = LlamaTokenizer(vocab_file=tokenizer_path)
        if model_parallel == 1:
            load_checkpoint(ckpt_path, network)
        else:
            network = load_distribut_checkpoint(config, ckpt_path, network)
        seq_len = 100
        input_ids = tokenizer(input_)['input_ids']
        outputs = network.generate(input_ids, do_sample=False, max_length=seq_len, top_p=1, top_k=3)
        answer = tokenizer.decode(outputs, skip_special_tokens=True)
        return outputs, answer
    example = "hello"
    quant(fp16_ckpt_path, fp16_config_path)
    foutput, _ = fp16_infer(example, fp16_ckpt_path, fp16_config_path)
    qoutput, _ = w8a16c8_infer(example, w8a16c8_ckpt_path, w8a16c8_config_path)
    npfoutput = np.array(foutput)
    npqoutput = np.array(qoutput)
    print(npfoutput)
    print(npqoutput)
    if not np.allclose(npqoutput[:, :30], npfoutput[:, :30], 0, 0):
        return False
    return relative_tolerance_acceptable(np.array(qoutput), np.array(foutput), 25.3)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", ["Ascend"])
@pytest.mark.parametrize("mode", [GRAPH_MODE])
def test_kv_llama2_predict_2stage_1p(device, mode):
    """
    Feature: test RTQ kvcache int8 quant in two stages with one cards.
    Description: apply RTQ kvcache int8 quant on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    model_parallel = int(os.environ.get("sq_test_model_parallel", 1))
    assert kv_predict_llama2_2stage(device, mode, model_parallel)

@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_kv_llama2_predict_2stage_2p():
    """
    Feature: test RTQ kvcache int8 quant in two stages with two cards.
    Description: apply RTQ kvcache int8 quant on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    model_parallel = 2
    os.environ['sq_test_model_parallel'] = str(model_parallel)
    return_code = os.system(
        "msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 "
        "--master_port=10926 --join=True --log_dir=./test_sq_predict_llama2_13b_logs "
        "pytest -s test_kvcache_int8.py::test_kv_llama2_predict_2stage_1p"
    )
    if return_code != 0:
        log_file = open("./test_sq_predict_llama2_13b_logs/worker_1.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()

    assert return_code == 0

@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", ["Ascend"])
@pytest.mark.parametrize("mode", [GRAPH_MODE])
def test_kv_fusion_ops_llama2_predict_2stage_1p(device, mode):
    """
    Feature: test RTQ kvcache int8 quant use PA int8 ops in two stages with one cards.
    Description: apply RTQ kvcache int8 quant use PA int8 ops on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    model_parallel = int(os.environ.get("sq_test_model_parallel", 1))
    assert kv_predict_llama2_2stage(device, mode, model_parallel, True)

@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_kv_fusion_ops_llama2_predict_2stage_2p():
    """
    Feature: test RTQ kvcache int8 quant use PA int8 ops in two stages with two cards.
    Description: apply RTQ kvcache int8 quant use PA int8 ops on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    model_parallel = 2
    os.environ['sq_test_model_parallel'] = str(model_parallel)
    return_code = os.system(
        "msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 "
        "--master_port=10926 --join=True --log_dir=./test_sq_predict_llama2_13b_logs "
        "pytest -s test_kvcache_int8.py::test_kv_fusion_ops_llama2_predict_2stage_1p"
    )
    if return_code != 0:
        log_file = open("./test_sq_predict_llama2_13b_logs/worker_1.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()

    assert return_code == 0
