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
import sys
import time
import pytest
import numpy as np
import mindspore as ms
from mindspore import set_context, context, nn, Tensor, dtype, GRAPH_MODE, PYNATIVE_MODE
from mindspore.dataset import GeneratorDataset
from mindspore.ops.auto_generate import SiLU, SplitWithSize
from mindspore.ops import operations as P
from mindformers.modules import Linear
from mindspore_gs.ptq.ptq import PTQ
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import (PTQConfig, PTQMode, OutliersSuppressionType,
                              PrecisionRecovery, GPTQQuantConfig, AWQConfig, QuantGranularity)
from tests.st.test_utils import get_available_port

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../mindformers")))
# pylint: disable=wrong-import-position
from research.llama3_1.infer.layers import ColumnParallelLinear, RowParallelLinear


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

    def __init__(self, foo_seq_length=1024):
        super(SimpleSwiGLUNet, self).__init__()
        self.hidden_act = SwiGLU
        self.foo_seq_length = foo_seq_length
        SwiGLU.set_size(512)
        linear = Linear(in_channels=foo_seq_length, out_channels=foo_seq_length, activation=self.hidden_act,
                        weight_init="ones")
        linear.out_channels = 512
        self.decoder = SimpleNet.DecoderCell(linear)

    def construct(self, x):
        """decoder"""
        return self.decoder(x)

    # pylint: disable=unused-argument
    def generate(self, input_ids, do_sample=False, max_new_tokens=1):
        input_ids = np.pad(input_ids, ((0, 0), (0, self.foo_seq_length - input_ids.shape[1])), 'constant',
                           constant_values=0)
        return self.construct(Tensor(input_ids, dtype=dtype.float16))


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

    def __init__(self, linear_type, foo_seq_length=1024):
        super(SimpleGmmNet, self).__init__()
        self.config = SimpleGmmNet.ParallelConfig()
        self.foo_seq_length = foo_seq_length
        if linear_type == "ColumnParallelLinear":
            linear = ColumnParallelLinear(
                foo_seq_length,
                foo_seq_length,
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
                foo_seq_length,
                foo_seq_length,
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

    # pylint: disable=unused-argument
    def generate(self, input_ids, do_sample=False, max_new_tokens=1):
        input_ids = np.pad(input_ids, ((0, 0), (0, self.foo_seq_length - input_ids.shape[1])), 'constant',
                           constant_values=0)
        return self.construct(Tensor(input_ids, dtype=dtype.bfloat16))


def quant_simple_swiglu_net(non_decoder, quant_type):
    """
    Feature: quant simplenet which including one linear and SwiGLU activation.
    Description: quant simplenet with A8W8C8 PTQ algorithm.
    Expectation: correct quant simplenet.
    """
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
    os.environ['FORCE_EAGER'] = "true"
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"

    network = SimpleSwiGLUNet(1024)
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
    network = ptq.apply(network, datasets=ds)
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
    os.environ.pop('FORCE_EAGER', None)
    os.environ.pop('MS_JIT', None)
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    set_context(mode=GRAPH_MODE, jit_config={"jit_level": "O0", "infer_boost": "on"})

    network = SimpleSwiGLUNet(1024)
    ds = create_foo_ds(1)

    for _, ds_item in enumerate(ds.create_dict_iterator()):
        input_ids = ds_item['input_ids'].asnumpy()
        foutput = network.generate(input_ids, max_new_tokens=100)
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
        qoutput = network.generate(input_ids, max_new_tokens=100)
    np.allclose(foutput.asnumpy(), qoutput.asnumpy(), 0, 0)


def quant_simple_gmm_net(non_decoder, linear_type, quant_type):
    """
    Feature: quant simplenet which including one gmm linear.
    Description: quant simplenet with A8W8C8 PTQ algorithm.
    Expectation: correct quant simplenet.
    """
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
    os.environ['FORCE_EAGER'] = "true"
    os.environ["RUN_MODE"] = "predict"
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"

    network = SimpleGmmNet(linear_type, 1024)
    ds = create_foo_ds(1)
    fpoutput = []
    for _, ds_item in enumerate(ds.create_dict_iterator()):
        input_ids = ds_item['input_ids'].asnumpy()
        fpoutput.append(network.generate(input_ids, max_new_tokens=100))
    if quant_type == "w8perchannela8pertoken":
        cfg = PTQConfig(mode=PTQMode.QUANTIZE,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        act_quant_granularity=QuantGranularity.PER_TOKEN,
                        act_quant_dtype=dtype.int8,
                        weight_quant_dtype=dtype.int8)
    elif quant_type == "pertoken-smooth":
        cfg = PTQConfig(mode=PTQMode.QUANTIZE,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        act_quant_granularity=QuantGranularity.PER_TOKEN,
                        outliers_suppression=OutliersSuppressionType.SMOOTH,
                        act_quant_dtype=dtype.int8,
                        weight_quant_dtype=dtype.int8)
    elif quant_type == "w8a8-smoothquant":
        cfg = PTQConfig(mode=PTQMode.QUANTIZE,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        outliers_suppression=OutliersSuppressionType.SMOOTH,
                        act_quant_dtype=dtype.int8,
                        weight_quant_dtype=dtype.int8)
    elif quant_type == "w8a8":
        cfg = PTQConfig(mode=PTQMode.DEPLOY,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        act_quant_dtype=dtype.int8,
                        weight_quant_dtype=dtype.int8)
    elif quant_type == "a16w4-gptq":
        gptq_config = GPTQQuantConfig()
        cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND,
                        weight_quant_dtype=dtype.qint4x2,
                        algo_args=gptq_config,
                        act_quant_dtype=None,
                        precision_recovery=PrecisionRecovery.GPTQ,
                        weight_quant_granularity=QuantGranularity.PER_GROUP,
                        opname_blacklist=['lm_head', 'lkv2kv'],
                        group_size=64)
    else:
        raise ValueError(f"Unsupported quant_algo : {quant_type}")
    set_context(mode=PYNATIVE_MODE, jit_config={"jit_level": "O0", "infer_boost": "on"})
    ptq = PTQ(config=cfg)
    if non_decoder:
        ptq.decoder_layer_types.append(SimpleGmmNet.DecoderCell)
    network = ptq.apply(network, datasets=ds)
    network = ptq.convert(network)
    ms.save_checkpoint(network.parameters_dict(), os.path.join("./simplegmm-quant.ckpt"),
                       choice_func=lambda x: "key_cache" not in x and "value_cache" not in x and \
                        "float_weight" not in x)
    return fpoutput


def eval_simple_gmm_net(non_decoder, linear_type, quant_type):
    """
    Feature: eval simplenet which including one GroupedMatMul linear.
    Description: simple GroupedMatMul network inference.
    Expectation: network inference normally.
    """
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
    os.environ['MS_INTERNAL_ENABLE_CUSTOM_KERNAL_LIST'] = "QbmmAllReduceAdd,QbmmAdd"
    os.environ.pop('FORCE_EAGER', None)
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    set_context(mode=GRAPH_MODE, jit_config={"jit_level": "O0", "infer_boost": "on"})

    network = SimpleGmmNet(linear_type, 1024)
    ds = create_foo_ds(1)

    if quant_type == "w8perchannela8pertoken":
        cfg = PTQConfig(mode=PTQMode.DEPLOY,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        act_quant_granularity=QuantGranularity.PER_TOKEN,
                        act_quant_dtype=dtype.int8,
                        weight_quant_dtype=dtype.int8)
    elif quant_type == "pertoken-smooth":
        cfg = PTQConfig(mode=PTQMode.DEPLOY,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        act_quant_granularity=QuantGranularity.PER_TOKEN,
                        outliers_suppression=OutliersSuppressionType.SMOOTH,
                        act_quant_dtype=dtype.int8,
                        weight_quant_dtype=dtype.int8)
    elif quant_type == "w8a8-smoothquant":
        cfg = PTQConfig(mode=PTQMode.DEPLOY,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        outliers_suppression=OutliersSuppressionType.SMOOTH,
                        act_quant_dtype=dtype.int8,
                        weight_quant_dtype=dtype.int8)
    elif quant_type == "w8a8":
        cfg = PTQConfig(mode=PTQMode.DEPLOY,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        act_quant_dtype=dtype.int8,
                        weight_quant_dtype=dtype.int8)
    elif quant_type == "a16w4-gptq":
        gptq_config = GPTQQuantConfig()
        cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND,
                        weight_quant_dtype=dtype.qint4x2,
                        algo_args=gptq_config,
                        act_quant_dtype=None,
                        precision_recovery=PrecisionRecovery.GPTQ,
                        weight_quant_granularity=QuantGranularity.PER_GROUP,
                        opname_blacklist=['lm_head', 'lkv2kv'],
                        group_size=64)
    else:
        raise ValueError(f"Unsupported quant_algo : {quant_type}")
    ptq = PTQ(config=cfg)
    if non_decoder:
        ptq.decoder_layer_types.append(SimpleGmmNet.DecoderCell)
    network = ptq.apply(network, ds=ds)
    network = ptq.convert(network)
    param_dict = ms.load_checkpoint('./simplegmm-quant.ckpt')
    qoutput = []
    ms.load_param_into_net(network, param_dict)
    for _, ds_item in enumerate(ds.create_dict_iterator()):
        input_ids = ds_item['input_ids'].asnumpy()
        qoutput.append(network.generate(input_ids, max_new_tokens=100))
    return qoutput


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


def get_cos_similar(a: list, b: list):
    '''get_cos_similar'''
    a = a.astype(np.float32).flatten()
    b = b.astype(np.float32).flatten()
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cosine_similarity = dot_product / (norm_a * norm_b)
    return cosine_similarity

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("non_decoder", [True, False])
@pytest.mark.parametrize("linear_type", ["RowParallelLinear", "ColumnParallelLinear"])
@pytest.mark.parametrize("quant_type", ["w8perchannela8pertoken", "pertoken-smooth",
                                        "w8a8-smoothquant", "w8a8", "a16w4-gptq"])
def test_ptq_simple_gmm_net(non_decoder, linear_type, quant_type):
    """
    Feature: eval simplenet which including one GroupedMatMul linear.
    Description: simple GroupedMatMul network inference.
    Expectation: network inference normally.
    """
    fpoutput = quant_simple_gmm_net(non_decoder, linear_type, quant_type)
    qoutput = eval_simple_gmm_net(non_decoder, linear_type, quant_type)
    if quant_type != "w8a8":
        for fpout, qout in zip(fpoutput, qoutput):
            assert get_cos_similar(fpout, qout) > 0.99


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

    def __init__(self, foo_seq_length=1024):
        super(SimpleNet, self).__init__()
        self.foo_seq_length = foo_seq_length
        linear = Linear(in_channels=foo_seq_length, out_channels=foo_seq_length, weight_init="ones")
        self.decoder = SimpleNet.DecoderCell(linear)

    def construct(self, x):
        """decoder"""
        return self.decoder(x)

    # pylint: disable=unused-argument
    def generate(self, input_ids, do_sample=False, max_new_tokens=1):
        input_ids = np.pad(input_ids, ((0, 0), (0, self.foo_seq_length - input_ids.shape[1])), 'constant',
                           constant_values=0)
        return self.construct(Tensor(input_ids, dtype=dtype.float16))


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
    os.environ['FORCE_EAGER'] = "true"
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
    os.environ['FORCE_EAGER'] = "true"
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
    with pytest.raises(ValueError, match="In QUANTIZE phase, please set mode=PYNATIVE_MODE."):
        ptq.apply(network, helper)


def quant_simplenet(non_decoder):
    """
    Feature: quant simplenet which including one linear.
    Description: quant simplenet with A8W8C8 PTQ algorithm.
    Expectation: correct quant simplenet.
    """
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    os.environ['FORCE_EAGER'] = "true"
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"

    network = SimpleNet(1024)
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
    network = ptq.apply(network, datasets=ds)
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
    os.environ.pop('FORCE_EAGER', None)
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    set_context(mode=GRAPH_MODE, jit_config={"jit_level": "O0", "infer_boost": "on"})

    network = SimpleNet(1024)
    ds = create_foo_ds(1)

    for _, ds_item in enumerate(ds.create_dict_iterator()):
        input_ids = ds_item['input_ids'].asnumpy()
        foutput = network.generate(input_ids, max_new_tokens=100)

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
        qoutput = network.generate(input_ids, max_new_tokens=100)
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
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_algo", ['A16W4_AWQ'])
def test_ptq_llama2_predict_2stage_1p_run_level0(quant_algo):
    """
    Feature: test PTQ adjust parameter in two stages with one cards.
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


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_algo", ['A8W8C8', 'A8W8_FallBack', 'A16W4_GPTQ', 'A16W4_GPTQ_per_group'])
def test_ptq_llama2_predict_2stage_1p_run_leval1(quant_algo):
    """
    Feature: test PTQ adjust parameter in two stages with one cards.
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
@pytest.mark.parametrize("quant_algo", ['OSPQuant_A8W8', 'OSL_A8W8'])
def test_ptq_llama2_predict_2stage_2p_run_level0(quant_algo):
    """
    Feature: test PTQ adjust parameter in two stages with two cards.
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


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
@pytest.mark.parametrize("quant_algo", ['A16W8C8', 'OSL_A8W8', 'Quant_A8W16_Deploy_A8W8_Dynamic'])
def test_ptq_llama2_predict_2stage_2p_run_level1(quant_algo):
    """
    Feature: test PTQ adjust parameter in two stages with two cards.
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
