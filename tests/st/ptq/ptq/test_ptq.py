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

import mindspore as ms
from mindspore import set_context, context, nn, Tensor, dtype, GRAPH_MODE, PYNATIVE_MODE
from mindspore.dataset import GeneratorDataset
from mindformers.modules import Linear
from mindspore_gs.ptq.ptq import PTQ
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQConfig, PTQMode, OutliersSuppressionType
from mindspore_gs.ptq.network_helpers.network_helper import LayerInfo
from mindspore_gs.ptq.network_helpers import NetworkHelper
from tests.st.test_utils import get_available_port


class SimpleNet(nn.Cell):
    """
    Network with single linear to be quant
    """
    class DecoderCell(nn.Cell):
        """decoder cell"""
        def __init__(self, linear):
            super().__init__()
            self.linear = linear

        def construct(self, *args, **kwargs):
            """linear"""
            return self.linear(*args, **kwargs)

    def __init__(self):
        super(SimpleNet, self).__init__()
        linear = Linear(in_channels=1024, out_channels=1024, weight_init="ones")
        self.decoder = SimpleNet.DecoderCell(linear)

    def construct(self, x):
        """decoder"""
        return self.decoder(x)


class SimpleNetworkHelper(NetworkHelper):
    """SimpleNetworkHelper"""
    def __init__(self, **kwargs) -> None:
        self.attrs = kwargs

    def create_network(self):
        return SimpleNet()

    def get_spec(self, name: str):
        return self.attrs.get(name, None)

    def create_tokenizer(self, **kwargs):
        return None

    def generate(self, network: nn.Cell, input_ids: np.ndarray, max_new_tokens=1, **kwargs):
        input_ids = np.pad(input_ids, ((0, 0), (0, self.get_spec("seq_length") - input_ids.shape[1])), 'constant',
                           constant_values=0)
        return network(Tensor(input_ids, dtype=dtype.float16))

    def assemble_inputs(self, input_ids: np.ndarray, **kwargs):
        raise RuntimeError("InnerError, should not invoke SimpleNetworkHelper.assemble_inputs()")

    def analysis_decoder_groups(self, network):
        pass

    def get_pre_layer(self, linear_name):
        return None


def create_foo_ds(repeat=1):
    """create_hello_ds"""
    class SimpleIterable:
        """SimpleIterable"""
        def __init__(self, repeat=1):
            self._index = 0
            self.data = []
            for _ in range(repeat):
                self.data.append(np.array([[1, 1, 1]], dtype=np.int32))

        def __next__(self):
            if self._index >= len(self.data):
                raise StopIteration
            item = (self.data[self._index],)
            self._index += 1
            return item

        def __iter__(self):
            self._index = 0
            return self

        def __len__(self):
            return len(self.data)

    return GeneratorDataset(source=SimpleIterable(repeat), column_names=["input_ids"])


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
    foo_input = Tensor(np.ones((1, 512), dtype=np.float16))
    catcher = InputCatcher(net.decoder)
    net.decoder = catcher

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
        assert catcher.args[i][0].shape == (1, 512)
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
    config_path_ = os.path.join(cur_dir, "../../../data/test_llama2/predict_llama2_13b_1p.yaml")
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


def quant_simplenet():
    """
    Feature: quant simplenet which including one linear.
    Description: quant simplenet with A8W8C8 PTQ algorithm.
    Expectation: correct quant simplenet.
    """
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"

    net_helper = SimpleNetworkHelper(seq_length=1024)
    network = net_helper.create_network()
    ds = create_foo_ds(1)

    cfg = PTQConfig(mode=PTQMode.QUANTIZE,
                    backend=BackendTarget.ASCEND,
                    opname_blacklist=["w2", "lm_head"],
                    act_quant_dtype=dtype.int8,
                    weight_quant_dtype=dtype.int8)
    set_context(mode=PYNATIVE_MODE, jit_config={"jit_level": "O0", "infer_boost": "on"})
    ptq = PTQ(config=cfg)
    # pylint: disable=w0212
    ptq._config.enable_deploy_fusion = False
    ptq.decoder_layer_types.append(SimpleNet.DecoderCell)
    network = ptq.apply(network, net_helper, datasets=ds)
    network = ptq.convert(network)
    ms.save_checkpoint(network.parameters_dict(), os.path.join("./simplenet-quant.ckpt"),
                       choice_func=lambda x: "key_cache" not in x and "value_cache" not in x and \
                        "float_weight" not in x)


def eval_simplenet():
    """
    Feature: eval simplenet which including one linear.
    Description: eval the accuracy of quantized simplenet.
    Expectation: correct accuracy.
    """
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
    os.environ['MS_INTERNAL_ENABLE_CUSTOM_KERNAL_LIST'] = "QbmmAllReduceAdd,QbmmAdd"
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    set_context(mode=GRAPH_MODE, jit_config={"jit_level": "O0", "infer_boost": "on"})

    net_helper = SimpleNetworkHelper(seq_length=1024)
    network = net_helper.create_network()
    ds = create_foo_ds(1)

    for _, ds_item in enumerate(ds.create_dict_iterator()):
        input_ids = ds_item['input_ids'].asnumpy()
        foutput = net_helper.generate(network, input_ids, max_new_tokens=100)

    cfg = PTQConfig(mode=PTQMode.DEPLOY,
                    backend=BackendTarget.ASCEND,
                    opname_blacklist=["w2", "lm_head"],
                    act_quant_dtype=dtype.int8,
                    weight_quant_dtype=dtype.int8)
    ptq = PTQ(config=cfg)
    ptq.decoder_layer_types.append(SimpleNet.DecoderCell)
    network = ptq.apply(network, ds=ds)
    network = ptq.convert(network)
    param_dict = ms.load_checkpoint('./simplenet-quant.ckpt')
    ms.load_param_into_net(network, param_dict)
    for _, ds_item in enumerate(ds.create_dict_iterator()):
        input_ids = ds_item['input_ids'].asnumpy()
        qoutput = net_helper.generate(network, input_ids, max_new_tokens=100)
    np.allclose(foutput.asnumpy(), qoutput.asnumpy(), 0, 0)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_ptq_simplenet():
    """
    Feature: quant and eval simplenet which including one linear.
    Description: quant net and eval the accuracy of quantized simplenet.
    Expectation: correct accuracy.
    """
    quant_simplenet()
    eval_simplenet()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_algo", ['A8W8', 'A16W8', 'A8W8C8', 'A16W8C8', 'C8'])
def test_ptq_llama2_predict_2stage_1p_run(quant_algo):
    """
    Feature: test omni quant adjust parameter in two stages with one cards.
    Description: apply OQ on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptq_network_runner.py")
    port = get_available_port()
    os.system(f"kill -9 $(lsof -i:{port} | " + "awk '{print $2}')")
    time.sleep(1.0)
    return_code = os.system(
        f"msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 "
        f"--master_port={port} --join=True --log_dir=./test_ptq_{quant_algo}_predict_llama2_1p_logs "
        f"python {run_file} -m 1 -a {quant_algo}"
    )
    if return_code != 0:
        log_file = open(f"./test_ptq_{quant_algo}_predict_llama2_1p_logs/worker_0.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()
    os.system("ps -u | grep 'ptq_network_runner' | grep -v grep | awk -F ' ' '{print$2}' | xargs kill -9")
    os.system(f"kill -9 $(lsof -i:{port} | " + "awk '{print $2}')")
    time.sleep(1.0)
    assert return_code == 0


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
@pytest.mark.parametrize("quant_algo", ['A8W8', 'A16W8', 'A8W8C8', 'A16W8C8', 'C8', 'A8W8_FallBack'])
def test_ptq_llama2_predict_2stage_2p_run(quant_algo):
    """
    Feature: test omni quant adjust parameter in two stages with two cards.
    Description: apply OQ on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    os.environ['quant_algo'] = f"{quant_algo}"
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptq_network_runner.py")
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
