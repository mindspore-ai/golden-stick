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
from mindspore.ops.auto_generate import SiLU, SplitWithSize
from mindspore.ops import operations as P
from mindformers.modules import Linear
from mindformers.experimental.infer.core.layers import RowParallelLinear, ColumnParallelLinear
from mindspore_gs.ptq.ptq import PTQ
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import (PTQConfig, PTQMode, OutliersSuppressionType,
                              PrecisionRecovery, GPTQQuantConfig, AWQConfig, QuantGranularity)
from mindspore_gs.ptq.network_helpers import NetworkHelper
from tests.st.test_utils import get_available_port


class SwiGLU(nn.Cell):
    """
    SwiGLU
    """
    _size = 0

    def __init__(self):
        super(SwiGLU, self).__init__()
        self.size = SwiGLU._size
        self.silu = SiLU()
        self.split = SplitWithSize()
        self.mul = P.Mul()

    @classmethod
    def set_size(cls, size):
        cls._size = size

    def construct(self, x):
        x0, x1 = self.split(x, (self.size, self.size), -1)
        output = self.mul(x1, self.silu(x0))
        return output


class SimpleSwiGLUNet(nn.Cell):
    """
    Network with single linear and SwiGLU activation to be quant
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
        super(SimpleSwiGLUNet, self).__init__()
        self.hidden_act = SwiGLU
        SwiGLU.set_size(512)
        linear = Linear(in_channels=1024, out_channels=1024, activation=self.hidden_act, weight_init="ones")
        linear.out_channels = 512
        self.decoder = SimpleNet.DecoderCell(linear)

    def construct(self, x):
        """decoder"""
        return self.decoder(x)


class SimpleSwiGLUNetworkHelper(NetworkHelper):
    """SimpleSwiGLUNetworkHelper"""
    def __init__(self, **kwargs) -> None:
        self.attrs = kwargs

    def create_network(self):
        return SimpleSwiGLUNet()

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


class SimpleGmmNet(nn.Cell):
    """
    Network with single GroupedMatmul linear
    """
    class DecoderCell(nn.Cell):
        """decoder cell"""
        def __init__(self, linear):
            super().__init__()
            self.linear = linear

        def construct(self, *args, **kwargs):
            """linear"""
            return self.linear(*args, **kwargs)

    class ParallelConfig(nn.Cell):
        """ParallelConfig"""
        def __init__(self):
            super().__init__()
            self.use_sequence_parallel = False

    def __init__(self, linear_type):
        super(SimpleGmmNet, self).__init__()
        self.config = SimpleGmmNet.ParallelConfig()
        if linear_type == "ColumnParallelLinear":
            linear = ColumnParallelLinear(
                1024,
                1024,
                config=self.config,
                bias=False,
                transpose_b=True,
                gather_output=False,
                param_init_type=dtype.bfloat16,
                compute_dtype=dtype.bfloat16,
                is_expert=True,
                expert_num=10
            )
        elif linear_type == "RowParallelLinear":
            linear = RowParallelLinear(
                1024,
                1024,
                config=self.config,
                input_is_parallel=True,
                bias=False,
                skip_bias_add=True,
                transpose_b=True,
                param_init_type=dtype.bfloat16,
                compute_dtype=dtype.bfloat16,
                is_expert=True,
                expert_num=10
            )

        self.decoder = SimpleGmmNet.DecoderCell(linear)
        self.group_list = Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=dtype.int64)

    def construct(self, x):
        """decoder"""
        return self.decoder(x, group_list=self.group_list)


class SimpleGmmNetworkHelper(NetworkHelper):
    """SimpleGmmNetworkHelper"""
    def __init__(self, **kwargs) -> None:
        self.attrs = kwargs

    def create_network(self):
        return SimpleGmmNet(self.attrs["linear_type"])

    def get_spec(self, name: str):
        return self.attrs.get(name, None)

    def create_tokenizer(self, **kwargs):
        return None

    def generate(self, network: nn.Cell, input_ids: np.ndarray, max_new_tokens=1, **kwargs):
        input_ids = np.pad(input_ids, ((0, 0), (0, self.get_spec("seq_length") - input_ids.shape[1])), 'constant',
                           constant_values=0)
        return network(Tensor(input_ids, dtype=dtype.bfloat16))

    def assemble_inputs(self, input_ids: np.ndarray, **kwargs):
        raise RuntimeError("InnerError, should not invoke SimpleNetworkHelper.assemble_inputs()")


def quant_simple_swiglu_net(non_decoder, quant_type):
    """
    Feature: quant simplenet which including one linear and SwiGLU activation.
    Description: quant simplenet with A8W8C8 PTQ algorithm.
    Expectation: correct quant simplenet.
    """
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"

    net_helper = SimpleSwiGLUNetworkHelper(seq_length=1024)
    network = net_helper.create_network()
    ds = create_foo_ds(1)

    if quant_type == "w8a8":
        cfg = PTQConfig(mode=PTQMode.QUANTIZE,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        act_quant_dtype=dtype.int8,
                        weight_quant_dtype=dtype.int8)
    else:
        cfg = PTQConfig(mode=PTQMode.QUANTIZE,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        weight_quant_dtype=dtype.int8)
    set_context(mode=PYNATIVE_MODE, jit_config={"jit_level": "O0", "infer_boost": "on"})
    ptq = PTQ(config=cfg)
    if non_decoder:
        ptq.decoder_layer_types.append(SimpleSwiGLUNet.DecoderCell)
    network = ptq.apply(network, net_helper, datasets=ds)
    network = ptq.convert(network)
    ms.save_checkpoint(network.parameters_dict(), os.path.join("./simpleswiglunet-quant.ckpt"),
                       choice_func=lambda x: "key_cache" not in x and "value_cache" not in x and \
                        "float_weight" not in x)


def eval_simple_swiglu_net(non_decoder, quant_type):
    """
    Feature: eval simplenet which including one linear and SwiGLU activation.
    Description: eval the accuracy of quantized simplenet.
    Expectation: correct accuracy.
    """
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
    os.environ['MS_INTERNAL_ENABLE_CUSTOM_KERNAL_LIST'] = "QbmmAllReduceAdd,QbmmAdd"
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    set_context(mode=GRAPH_MODE, jit_config={"jit_level": "O0", "infer_boost": "on"})

    net_helper = SimpleSwiGLUNetworkHelper(seq_length=1024)
    network = net_helper.create_network()
    ds = create_foo_ds(1)

    for _, ds_item in enumerate(ds.create_dict_iterator()):
        input_ids = ds_item['input_ids'].asnumpy()
        foutput = net_helper.generate(network, input_ids, max_new_tokens=100)
    if quant_type == "w8a8":
        cfg = PTQConfig(mode=PTQMode.DEPLOY,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        act_quant_dtype=dtype.int8,
                        weight_quant_dtype=dtype.int8)
    else:
        cfg = PTQConfig(mode=PTQMode.DEPLOY,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"])
    ptq = PTQ(config=cfg)
    if non_decoder:
        ptq.decoder_layer_types.append(SimpleSwiGLUNet.DecoderCell)
    network = ptq.apply(network, ds=ds)
    network = ptq.convert(network)
    param_dict = ms.load_checkpoint('./simpleswiglunet-quant.ckpt')
    ms.load_param_into_net(network, param_dict)
    for _, ds_item in enumerate(ds.create_dict_iterator()):
        input_ids = ds_item['input_ids'].asnumpy()
        qoutput = net_helper.generate(network, input_ids, max_new_tokens=100)
    np.allclose(foutput.asnumpy(), qoutput.asnumpy(), 0, 0)


def quant_simple_gmm_net(non_decoder, linear_type):
    """
    Feature: quant simplenet which including one gmm linear.
    Description: quant simplenet with A8W8C8 PTQ algorithm.
    Expectation: correct quant simplenet.
    """
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"

    net_helper = SimpleGmmNetworkHelper(seq_length=1024, linear_type=linear_type)
    network = net_helper.create_network()
    ds = create_foo_ds(1)

    cfg = PTQConfig(mode=PTQMode.QUANTIZE,
                    backend=BackendTarget.ASCEND,
                    opname_blacklist=["w2", "lm_head"],
                    act_quant_granularity=QuantGranularity.PER_TOKEN,
                    act_quant_dtype=dtype.int8,
                    weight_quant_dtype=dtype.int8)
    set_context(mode=PYNATIVE_MODE, jit_config={"jit_level": "O0", "infer_boost": "on"})
    ptq = PTQ(config=cfg)
    if non_decoder:
        ptq.decoder_layer_types.append(SimpleGmmNet.DecoderCell)
    network = ptq.apply(network, net_helper, datasets=ds)
    network = ptq.convert(network)
    ms.save_checkpoint(network.parameters_dict(), os.path.join("./simplegmm-quant.ckpt"),
                       choice_func=lambda x: "key_cache" not in x and "value_cache" not in x and \
                        "float_weight" not in x)


def eval_simple_gmm_net(non_decoder, linear_type):
    """
    Feature: eval simplenet which including one GroupedMatMul linear.
    Description: simple GroupedMatMul network inference.
    Expectation: network inference normally.
    """
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
    os.environ['MS_INTERNAL_ENABLE_CUSTOM_KERNAL_LIST'] = "QbmmAllReduceAdd,QbmmAdd"
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    set_context(mode=GRAPH_MODE, jit_config={"jit_level": "O0", "infer_boost": "on"})

    net_helper = SimpleGmmNetworkHelper(seq_length=1024, linear_type=linear_type)
    network = net_helper.create_network()
    ds = create_foo_ds(1)

    cfg = PTQConfig(mode=PTQMode.DEPLOY,
                    backend=BackendTarget.ASCEND,
                    opname_blacklist=["w2", "lm_head"],
                    act_quant_granularity=QuantGranularity.PER_TOKEN,
                    act_quant_dtype=dtype.int8,
                    weight_quant_dtype=dtype.int8)
    ptq = PTQ(config=cfg)
    if non_decoder:
        ptq.decoder_layer_types.append(SimpleGmmNet.DecoderCell)
    network = ptq.apply(network, ds=ds)
    network = ptq.convert(network)
    param_dict = ms.load_checkpoint('./simplegmm-quant.ckpt')
    ms.load_param_into_net(network, param_dict)
    for _, ds_item in enumerate(ds.create_dict_iterator()):
        input_ids = ds_item['input_ids'].asnumpy()
        net_helper.generate(network, input_ids, max_new_tokens=100)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("non_decoder", [True, False])
@pytest.mark.parametrize("quant_type", ["w8a16", "w8a8"])
def test_ptq_simple_swiglu_net(non_decoder, quant_type):
    """
    Feature: quant and eval simplenet which including one linear and SwiGLU activation.
    Description: quant net and eval the accuracy of quantized simplenet.
    Expectation: correct accuracy.
    """
    quant_simple_swiglu_net(non_decoder, quant_type)
    eval_simple_swiglu_net(non_decoder, quant_type)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("non_decoder", [True, False])
@pytest.mark.parametrize("linear_type", ["RowParallelLinear", "ColumnParallelLinear"])
def test_ptq_simple_gmm_net(non_decoder, linear_type):
    """
    Feature: eval simplenet which including one GroupedMatMul linear.
    Description: simple GroupedMatMul network inference.
    Expectation: network inference normally.
    """
    quant_simple_gmm_net(non_decoder, linear_type)
    eval_simple_gmm_net(non_decoder, linear_type)


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
    catcher = InputCatcher()
    catcher.patch(net.decoder)

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

    config = PTQConfig(act_quant_dtype=dtype.int8, weight_quant_dtype=None)
    with pytest.raises(ValueError):
        _ = PTQ(config)

    config = PTQConfig(act_quant_dtype=dtype.int8, weight_quant_dtype=None,
                       kvcache_quant_dtype=dtype.int8,
                       outliers_suppression=OutliersSuppressionType.SMOOTH)
    with pytest.raises(ValueError):
        _ = PTQ(config)

    config = PTQConfig(act_quant_dtype=dtype.int8, weight_quant_dtype=dtype.int8,
                       precision_recovery=PrecisionRecovery.GPTQ)
    with pytest.raises(ValueError):
        _ = PTQ(config)

    config = PTQConfig(act_quant_dtype=dtype.int8, weight_quant_dtype=dtype.int8,
                       outliers_suppression=OutliersSuppressionType.AWQ)
    with pytest.raises(ValueError):
        _ = PTQ(config)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gptq_config_error():
    """
    Feature: simulated GPTQQuantConfig __post_init__ function.
    Description: Feed invalid value of GPTQQuantConfig to __post_init__ function.
    Expectation: Except ValueError.
    """
    with pytest.raises(TypeError):
        _ = GPTQQuantConfig(block_size=0.1)
    with pytest.raises(TypeError):
        _ = GPTQQuantConfig(desc_act="0")
    with pytest.raises(TypeError):
        _ = GPTQQuantConfig(damp_percent="1")
    with pytest.raises(TypeError):
        _ = GPTQQuantConfig(static_groups="2")
    with pytest.raises(ValueError):
        _ = GPTQQuantConfig(block_size=-100)
    with pytest.raises(ValueError):
        _ = GPTQQuantConfig(damp_percent=-0.5)
    with pytest.raises(ValueError):
        _ = GPTQQuantConfig(damp_percent=2.1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_awq_config_error():
    """
    Feature: simulated AWQConfig __post_init__ function.
    Description: Feed invalid value of AWQConfig to __post_init__ function.
    Expectation: Except ValueError.
    """
    with pytest.raises(TypeError):
        _ = AWQConfig(duo_scaling=1)
    with pytest.raises(TypeError):
        _ = AWQConfig(smooth_alpha="1")
    with pytest.raises(TypeError):
        _ = AWQConfig(weight_clip_ratio="1")
    with pytest.raises(ValueError):
        _ = AWQConfig(smooth_alpha=-0.5)
    with pytest.raises(ValueError):
        _ = AWQConfig(weight_clip_ratio=-0.5)
    with pytest.raises(ValueError):
        _ = AWQConfig(smooth_alpha=[-1, 0.1, 0.5])
    with pytest.raises(ValueError):
        _ = AWQConfig(weight_clip_ratio=[0.1, 0.5, 10])


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


def quant_simplenet(non_decoder):
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
    if non_decoder:
        ptq.decoder_layer_types.append(SimpleNet.DecoderCell)
    network = ptq.apply(network, net_helper, datasets=ds)
    network = ptq.convert(network)
    ms.save_checkpoint(network.parameters_dict(), os.path.join("./simplenet-quant.ckpt"),
                       choice_func=lambda x: "key_cache" not in x and "value_cache" not in x and \
                        "float_weight" not in x)


def eval_simplenet(non_decoder):
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
    if non_decoder:
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
@pytest.mark.parametrize("non_decoder", [True, False])
def test_ptq_simplenet(non_decoder):
    """
    Feature: quant and eval simplenet which including one linear.
    Description: quant net and eval the accuracy of quantized simplenet.
    Expectation: correct accuracy.
    """
    quant_simplenet(non_decoder)
    eval_simplenet(non_decoder)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
# 'A8W8', 'A16W8'
@pytest.mark.parametrize("quant_algo", ['A8W8C8', 'A16W8C8'])
def test_ptq_llama2_predict_2stage_1p_run_part1(quant_algo):
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


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
# 'A8W8_Dynamic', 'Quant_A8W16_Deploy_A8W8_Dynamic'
@pytest.mark.parametrize("quant_algo", ['C8', 'C8_Dynamic', 'A16W4_GPTQ', 'A16W4_AWQ'])
def test_ptq_llama2_predict_2stage_1p_run_part2(quant_algo):
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
@pytest.mark.parametrize("quant_algo", ['A16W4_GPTQ_per_group'])
def test_ptq_llama2_predict_2stage_1p_run_per_group(quant_algo):
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


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
# 'A16W8C8'
@pytest.mark.parametrize("quant_algo", ['A8W8', 'A16W8', 'A8W8C8', 'OmniQuant_A8W8'])
def test_ptq_llama2_predict_2stage_2p_run_part1(quant_algo):
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


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
# 'A8W8_FallBack', 'A16W4_GPTQ', 'A16W4_AWQ'
@pytest.mark.parametrize("quant_algo", ['C8'])
def test_ptq_llama2_predict_2stage_2p_run_part2(quant_algo):
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


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
@pytest.mark.parametrize("quant_algo", ['A16W4_GPTQ_per_group'])
def test_ptq_llama2_predict_2stage_2p_run_per_group(quant_algo):
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


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
@pytest.mark.parametrize("quant_algo", ['A8W8_Dynamic', 'C8_Dynamic', 'Quant_A8W16_Deploy_A8W8_Dynamic'])
def test_ptq_dynamic_llama2_predict_2stage_2p_run(quant_algo):
    """
    Feature: test dynamic quant adjust parameter in two stages with two cards.
    Description: apply ptq on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    os.environ['quant_algo'] = f"{quant_algo}"
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptq_network_runner.py")
    port = get_available_port()
    os.system(f"kill -9 $(lsof -i:{port} | " + "awk '{print $2}')")
    time.sleep(1.0)
    return_code = os.system(
        f"msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 "
        f"--master_port={port} --join=True --log_dir=./test_ptq_dynamic_{quant_algo}_predict_llama2_2p_logs "
        f"python {run_file} -m 2 -a {quant_algo}"
    )
    if return_code != 0:
        log_file = open(f"./test_ptq_dynamic_{quant_algo}_predict_llama2_2p_logs/worker_0.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()
    os.system("ps -u | grep 'ptq_network_runner' | grep -v grep | awk -F ' ' '{print$2}' | xargs kill -9")
    os.system(f"kill -9 $(lsof -i:{port} | " + "awk '{print $2}')")
    time.sleep(1.0)
    assert return_code == 0
