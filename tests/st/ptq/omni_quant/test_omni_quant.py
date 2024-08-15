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
import numpy as np

from mindspore import Tensor, context, save_checkpoint, load_checkpoint, nn
from mindspore import GRAPH_MODE, PYNATIVE_MODE, dtype
from mindspore import ops as msops
from mindspore.dataset import GeneratorDataset
from mindformers.modules import Linear
from mindformers.models.llama.llama_tokenizer import LlamaTokenizer
from mindformers import LlamaForCausalLM, MindFormerConfig, LlamaConfig, init_context, TransformerOpParallelConfig

from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQConfig, PTQMode
from mindspore_gs.ptq.ptq_config import InnerPTQConfig
from mindspore_gs.ptq.omni_quant.quant_cells import SQLinearWrapper
from mindspore_gs.quantization.quant_utils import (cal_quantization_params,
                                                   quant_tensor_data, quant_bias_data)
from mindspore_gs.ptq.omni_quant import OmniQuant
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [PYNATIVE_MODE])
@pytest.mark.parametrize("transpose_b", [True, False])
def test_linear_wrapper_float_forwrad(mode, transpose_b):
    """
    Feature: test float_forward in SQLinearWrapper.
    Description: Input fake data and check output of each float_forward.
    Expectation: Same with real linear output.
    """
    context.set_context(device_target="Ascend", mode=mode)
    cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND)
    inner_cfg = InnerPTQConfig.inner_config(cfg)
    inner_cfg.act_bits = 8

    act_in = 5
    act_out = 6
    linear = Linear(in_channels=act_in, out_channels=act_out, transpose_b=transpose_b, bias_init="normal",
                    weight_init="normal")
    t_x = Tensor(np.random.normal(size=(act_in, act_in)), dtype=dtype.float16)
    # real float output
    real_fp_output = linear(t_x)

    # sq linear float output
    linear.add_flags(infer_mode="float")
    sq_linear_wrapper = SQLinearWrapper('linear', linear, inner_cfg)
    sq_fp_output = sq_linear_wrapper(t_x)
    print("-------real float output:\n", real_fp_output)
    print("-------sq linear float forward output:\n", sq_fp_output)
    assert np.allclose(real_fp_output.asnumpy(), sq_fp_output.asnumpy(), 0, 0)


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [PYNATIVE_MODE])
@pytest.mark.parametrize("transpose_b", [True, False])
def test_linear_quant_forward(mode, transpose_b):
    """
    Feature: test quant_forward in SQLinearWrapper.
    Description: Input fake data and check output of each quant_forward.
    Expectation: Same with numpy output.
    """
    context.set_context(device_target="Ascend", mode=mode, jit_config={"jit_level": "O0", "infer_boost": "on"})
    cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, act_quant_dtype=dtype.int8)
    inner_cfg = InnerPTQConfig.inner_config(cfg)
    inner_cfg.act_bits = 8

    pre_clip_ratio = 1.0
    post_clip_ratio = 1.0
    smooth_alpha = 0.5
    smooth_type = "smooth_quant"

    act_in = 5
    act_out = 6
    linear = Linear(in_channels=act_in, out_channels=act_out, transpose_b=transpose_b, bias_init="normal",
                    weight_init="normal")
    t_x = Tensor(np.random.normal(size=(act_in, act_in)), dtype=dtype.float16)
    real_output = linear(t_x)

    linear.add_flags(infer_mode="observer_x")
    sq_linear_wrapper = SQLinearWrapper('linear', linear, inner_cfg)
    sq_linear_wrapper(t_x)
    #pylint: disable=protected-access
    sq_linear_wrapper._handler.infer_mode = "quant"
    sq_linear_wrapper.set_search_args(pre_clip_ratio, post_clip_ratio, smooth_alpha, smooth_type)
    quant_forward_output = sq_linear_wrapper.quant_forward(t_x)
    assert np.isclose(quant_forward_output.asnumpy(), real_output.asnumpy(), 0, 1e-3).all()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [PYNATIVE_MODE])
@pytest.mark.parametrize("transpose_b", [True, False])
def test_linear_quant_forward_by_numpy(mode, transpose_b):
    """
    Feature: test quant_forward in SQLinearWrapper.
    Description: Input fake data and check output of each quant_forward.
    Expectation: Same with numpy output.
    """
    context.set_context(device_target="Ascend", mode=mode, jit_config={"jit_level": "O0", "infer_boost": "on"})
    cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, act_quant_dtype=dtype.int8)
    inner_cfg = InnerPTQConfig.inner_config(cfg)

    pre_clip_ratio = 1.0
    post_clip_ratio = 1.0
    act_clip_ratio = 1.0
    smooth_alpha = 0.5
    smooth_type = "smooth_quant"

    act_in = 5
    act_out = 6
    linear = Linear(in_channels=act_in, out_channels=act_out, transpose_b=transpose_b, bias_init="normal",
                    weight_init="normal")
    t_x = Tensor(np.random.normal(size=(act_in, act_in)), dtype=dtype.float16)
    fp_weight = linear.weight.asnumpy()

    # sq linear observer all inputs
    linear.add_flags(infer_mode="observer_x")
    sq_linear_wrapper = SQLinearWrapper('linear', linear, inner_cfg)
    sq_linear_wrapper(t_x)
    observer_x = sq_linear_wrapper.observer_x

    # sq linear quant forward
    #pylint: disable=protected-access
    sq_linear_wrapper._handler.infer_mode = "quant"

    # 1. set param
    sq_linear_wrapper.set_search_args(pre_clip_ratio, post_clip_ratio, smooth_alpha, smooth_type)

    # 2. weight observer
    ## sq linear
    #pylint: disable=protected-access
    sq_linear_wrapper._weight_observer()
    observer_w_max = sq_linear_wrapper.observer_w_max
    observer_w_min = sq_linear_wrapper.observer_w_min
    quantizer_w_max = sq_linear_wrapper.quantizer_w_max
    quantizer_w_min = sq_linear_wrapper.quantizer_w_min
    ## numpy
    if transpose_b:
        observer_axis = 0
        weight_axis = 0
        quantizer_axis = 1
    else:
        observer_axis = 1
        weight_axis = 1
        quantizer_axis = 0
    observer_w_max_np = np.max(fp_weight, observer_axis)
    observer_w_min_np = np.min(fp_weight, observer_axis)
    quantizer_w_max_np = np.max(fp_weight, quantizer_axis, keepdims=True)
    quantizer_w_min_np = np.min(fp_weight, quantizer_axis, keepdims=True)

    assert np.allclose(observer_w_max.asnumpy(), observer_w_max_np, 0, 0)
    assert np.allclose(observer_w_min.asnumpy(), observer_w_min_np, 0, 0)
    assert np.allclose(quantizer_w_max.asnumpy(), quantizer_w_max_np, 0, 0)
    assert np.allclose(quantizer_w_min.asnumpy(), quantizer_w_min_np, 0, 0)

    # 3. pre clip weight
    ## sq linear
    #pylint: disable=protected-access
    pre_clip_weight = sq_linear_wrapper._clip_weight(linear.weight, pre_clip_ratio)
    ## numpy
    quantizer_w_max_np = np.maximum(np.abs(quantizer_w_max_np), np.abs(quantizer_w_min_np))
    quantizer_w_max_np = quantizer_w_max_np * pre_clip_ratio
    quantizer_w_min_np = -quantizer_w_max_np
    pre_clip_weight_np = np.clip(fp_weight, quantizer_w_min_np, quantizer_w_max_np)
    assert np.allclose(pre_clip_weight.asnumpy(), pre_clip_weight_np, 0, 0)

    if inner_cfg.act_quant_dtype == dtype.int8:
        # 4. calc smooth scale
        ## sq linear
        sq_linear_wrapper.observer_x = msops.cat(tuple(sq_linear_wrapper.observer_x))
        #pylint: disable=protected-access
        smooth_scale = sq_linear_wrapper._calc_smooth_scale()
        ## numpy
        observer_x_np = msops.cat(tuple(observer_x)).asnumpy()
        act_max_np = np.maximum(np.abs(np.min(observer_x_np, 0)), np.abs(np.max(observer_x_np, 0)))
        input_max_pow_np = np.power(act_max_np, smooth_alpha)
        weight_max_np = np.maximum(np.abs(observer_w_max_np), np.abs(observer_w_min_np))
        weight_max_pow_np = np.power(weight_max_np, 1-smooth_alpha)
        smooth_scale_np = np.divide(input_max_pow_np, weight_max_pow_np).clip(1e-5)
        smooth_scale_np[input_max_pow_np == 0] = 1.0
        smooth_scale_np[weight_max_pow_np == 0] = 1.0
        assert np.isclose(smooth_scale.asnumpy(), smooth_scale_np, 0, 1e-5).all()

        # 5. smooth
        ## sq_linear
        if transpose_b:
            smooth_weight = msops.mul(pre_clip_weight, smooth_scale)
        else:
            weight_scale = smooth_scale.expand_dims(1)
            smooth_weight = msops.mul(pre_clip_weight, weight_scale)
        ## numpy
        if transpose_b:
            smooth_weight_np = pre_clip_weight_np * smooth_scale_np
        else:
            weight_scale_np = np.expand_dims(smooth_scale_np, 1)
            smooth_weight_np = pre_clip_weight_np * weight_scale_np
        assert np.isclose(smooth_weight.asnumpy(), smooth_weight_np, 0, 1e-7).all()

        # 6. update quantize w minmax
        sq_linear_wrapper.quantizer_w_max = msops.max(smooth_weight, quantizer_axis, keepdims=True)[0]
        sq_linear_wrapper.quantizer_w_min = msops.min(smooth_weight, quantizer_axis, keepdims=True)[0]
        quantizer_w_max_np = np.max(smooth_weight_np, quantizer_axis, keepdims=True)
        quantizer_w_min_np = np.min(smooth_weight_np, quantizer_axis, keepdims=True)
        assert np.isclose(sq_linear_wrapper.quantizer_w_max.asnumpy(), quantizer_w_max_np, 0, 1e-7).all()
        assert np.isclose(sq_linear_wrapper.quantizer_w_min.asnumpy(), quantizer_w_min_np, 0, 1e-7).all()

        # 7. post clip weight
        ## sq linear
        #pylint: disable=protected-access
        post_clip_weight = sq_linear_wrapper._clip_weight(smooth_weight, sq_linear_wrapper.post_clip_ratio)
        ## numpy
        quantizer_w_max_np = np.maximum(np.abs(quantizer_w_max_np), np.abs(quantizer_w_min_np))
        quantizer_w_max_np = quantizer_w_max_np * post_clip_ratio
        quantizer_w_min_np = -quantizer_w_max_np
        post_clip_weight_np = np.clip(smooth_weight_np, quantizer_w_min_np, quantizer_w_max_np)
        assert np.isclose(post_clip_weight.asnumpy(), post_clip_weight_np, 0, 1e-7).all()

        # 8. act smooth
        ## sq linear
        observer_x = msops.mul(sq_linear_wrapper.observer_x, msops.div(1.0, smooth_scale))
        ## numpy
        observer_x_np = observer_x_np / smooth_scale_np
        assert np.isclose(observer_x.asnumpy(), observer_x_np, 0, 1e-5).all()

        # 9. act statistic
        ## sq linear
        sq_linear_wrapper.quantizer_x_max = msops.max(observer_x)[0]
        sq_linear_wrapper.quantizer_x_min = msops.min(observer_x)[0]
        ## numpy
        quantizer_x_max_np = np.max(observer_x_np) * act_clip_ratio
        quantizer_x_min_np = np.min(observer_x_np) * act_clip_ratio
        assert np.isclose(sq_linear_wrapper.quantizer_x_max.asnumpy(), quantizer_x_max_np, 0, 1e-5).all()
        assert np.isclose(sq_linear_wrapper.quantizer_x_min.asnumpy(), quantizer_x_min_np, 0, 1e-5).all()

        # 10. act scale zp
        ## sq linear
        x_scale, x_zp = cal_quantization_params(sq_linear_wrapper.quantizer_x_min.asnumpy(),
                                                sq_linear_wrapper.quantizer_x_max.asnumpy(),
                                                -128, 127, symmetric=False)
        ## numpy
        x_scale_np = (quantizer_x_max_np - quantizer_x_min_np) / (127 + 128)
        x_zp_np = np.round(-128 - quantizer_x_min_np / x_scale_np)
        assert np.isclose(x_scale.squeeze(), x_scale_np, 0, 1e-5).all()
        assert np.isclose(x_zp.squeeze(), x_zp_np, 0, 1e-5).all()

    # 11. weight scale zp
    ## sq linear
    w_scale, w_zp = cal_quantization_params(sq_linear_wrapper.quantizer_w_min.asnumpy(),
                                            sq_linear_wrapper.quantizer_w_max.asnumpy(),
                                            -128, 127, symmetric=True)
    ## numpy
    w_scale_np = (quantizer_w_max_np * 2) / (127 + 128)
    w_zp_np = np.zeros_like(w_scale_np).round()
    assert np.isclose(w_scale, w_scale_np, 0, 1e-7).all()
    assert np.isclose(w_zp, w_zp_np, 0, 1e-7).all()

    # 13. weight quant
    if inner_cfg.act_quant_dtype == dtype.int8:
        ## sq linear
        weight_quant = quant_tensor_data(post_clip_weight, w_scale.squeeze(),
                                         w_zp.squeeze(), -128, 127, weight_axis)
        ## numpy
        weight_quant_np = post_clip_weight_np / w_scale_np + w_zp_np
        weight_quant_np = np.round(weight_quant_np)
        weight_quant_np = np.clip(weight_quant_np, -128, 127).astype(np.int32)
    else:
        weight_quant = quant_tensor_data(pre_clip_weight, w_scale.squeeze(),
                                         w_zp.squeeze(), -128, 127, weight_axis)
        ## numpy
        weight_quant_np = pre_clip_weight_np / w_scale_np + w_zp_np
        weight_quant_np = np.round(weight_quant_np)
        weight_quant_np = np.clip(weight_quant_np, -128, 127).astype(np.int32)
    assert np.isclose(weight_quant.asnumpy(), weight_quant_np, 0, 1).all()

    # 14. quant matmul
    if inner_cfg.act_quant_dtype == dtype.int8:
        #pylint: disable=protected-access
        if sq_linear_wrapper._handler.has_bias:
            ## sq linear
            #pylint: disable=protected-access
            quant_bias = quant_bias_data(sq_linear_wrapper._handler.bias, w_scale * x_scale)
            ## numpy
            dequant_scale_np = np.squeeze(w_scale_np * x_scale_np)
            #pylint: disable=protected-access
            quant_bias_np = np.round(sq_linear_wrapper._handler.bias.asnumpy()
                                     / dequant_scale_np).clip(-2 ** 31, 2 ** 31 - 1).astype(np.int32)
            assert np.isclose(quant_bias.asnumpy(), quant_bias_np, 0, 1e-4).all()


def _set_config(config_path):
    """setup MindFormerConfig"""
    mfconfig = MindFormerConfig(config_path)
    mfconfig.model.model_config = LlamaConfig(**mfconfig.model.model_config)
    init_context(use_parallel=mfconfig.use_parallel, context_config=mfconfig.context, parallel_config=mfconfig.parallel)
    if mfconfig.use_parallel:
        parallel_config = TransformerOpParallelConfig(**mfconfig.parallel_config)
        mfconfig.model.model_config.parallel_config = parallel_config
    mfconfig.model.model_config.checkpoint_name_or_path = mfconfig.load_checkpoint
    return mfconfig


def create_hello_ds(tokenizer, repeat=1):
    """create_hello_ds"""
    class SimpleIterable:
        """SimpleIterable"""
        def __init__(self, tokenizer, repeat=1):
            self._index = 0
            self.data = []
            for _ in range(repeat):
                input_ids = tokenizer("Hello")['input_ids']
                self.data.append(input_ids)

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

    return GeneratorDataset(source=SimpleIterable(tokenizer, repeat), column_names=["input_ids"])


def oq_quant_llama2(device, mode, model_parallel, fp16_config_path, fp16_ckpt_path, omni_ckpt_path):
    """omni quant to quant llama2"""
    print(f"---------------- Testing params: {device} {mode} ", flush=True)
    context.set_context(device_target=device, mode=mode, jit_config={"jit_level": "O0", "infer_boost": "on"})
    cur_dir, _ = os.path.split(os.path.abspath(__file__))
    tokenizer_path = os.path.join(cur_dir, "../../../data/llama2-tokenizer.model")
    fp16_config_path = os.path.join(cur_dir, fp16_config_path)
    fp16_ckpt_path = os.path.join(cur_dir, fp16_ckpt_path)
    omni_ckpt_path = os.path.join(cur_dir, omni_ckpt_path)

    config = _set_config(fp16_config_path)
    network = LlamaForCausalLM(config.model.model_config)
    tokenizer = LlamaTokenizer(vocab_file=tokenizer_path)
    if model_parallel == 1:
        load_checkpoint(fp16_ckpt_path, network)
    else:
        raise ValueError("only support model_parallel = 1 right now.")
    cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND,
                    opname_blacklist=["w2", "lm_head"], act_quant_dtype=dtype.int8)
    ptq = OmniQuant(config=cfg)
    network_helper = MFLlama2Helper(config)
    ds = create_hello_ds(tokenizer, 1)
    network = ptq.apply(network, network_helper, ds=ds)
    network = ptq.convert(network)
    if model_parallel == 1:
        save_checkpoint(network.parameters_dict(), omni_ckpt_path, integrated_save=False)
    else:
        raise ValueError("only support model_parallel = 1 right now.")


def eval_llama2(device, mode, model_parallel, is_quant, input_, config_path, ckpt_path):
    """eval llama2 by float ckpt and int ckpt"""
    os.environ['GRAPH_OP_RUN'] = "1"
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    print(f"---------------- Testing params: {device} {mode} ", flush=True)
    context.set_context(device_target=device, mode=mode, jit_config={"jit_level": "O0", "infer_boost": "on"})
    cur_dir, _ = os.path.split(os.path.abspath(__file__))
    tokenizer_path = os.path.join(cur_dir, "../../../data/llama2-tokenizer.model")
    config_path = os.path.join(cur_dir, config_path)
    ckpt_path = os.path.join(cur_dir, ckpt_path)

    config = _set_config(config_path)
    network = LlamaForCausalLM(config.model.model_config)
    if is_quant:
        cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        act_quant_dtype=dtype.int8)
        ptq = OmniQuant(config=cfg)
        network_helper = MFLlama2Helper(config)
        network = ptq.apply(network, network_helper)
        network = ptq.convert(network)
    tokenizer = LlamaTokenizer(vocab_file=tokenizer_path)
    if model_parallel == 1:
        load_checkpoint(ckpt_path, network)
    else:
        raise ValueError("only support model_parallel = 1 right now.")
    seq_len = 100
    input_ids = tokenizer(input_)['input_ids']
    outputs = network.generate(input_ids, do_sample=False, max_length=seq_len, top_p=1, top_k=3)
    answer = tokenizer.decode(outputs, skip_special_tokens=True)
    return outputs, answer


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", ["Ascend"])
def test_oq_llama2_predict_2stage_1p(device):
    """
    Feature: test omni quant adjust parameter in two stages with one cards.
    Description: apply OQ on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    model_parallel = int(os.environ.get("sq_test_model_parallel", 1))
    fp16_pynative_config_path = "../../../data/test_llama2/predict_llama2_13b_fp16_910b_pynative_1p.yaml"
    fp16_graph_config_path = "../../../data/test_llama2/predict_llama2_13b_fp16_910b_1p.yaml"
    fp16_ckpt_path = "../../../data/test_llama2/llama2-13b-fp16-1decoder.ckpt"
    omni_config_path = "../../../data/test_llama2/predict_llama2_13b_w8a8_910b_1p.yaml"
    omni_ckpt_path = "../../../data/test_llama2/llama2-13b-omniquant-1decoder.ckpt"

    oq_quant_llama2(device, PYNATIVE_MODE, model_parallel, fp16_pynative_config_path, fp16_ckpt_path, omni_ckpt_path)
    example = "Hello"
    foutput, _ = eval_llama2(device, GRAPH_MODE, model_parallel, False, example, fp16_graph_config_path, fp16_ckpt_path)
    qoutput, _ = eval_llama2(device, GRAPH_MODE, model_parallel, True, example, omni_config_path, omni_ckpt_path)
    npfoutput = np.array(foutput)
    npqoutput = np.array(qoutput)
    assert np.allclose(npqoutput[:, :24], npfoutput[:, :24], 0, 0)


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
    from mindspore_gs.ptq.omni_quant.omni_quant import InputCatcher
    context.set_context(mode=context.PYNATIVE_MODE, device_target=device)

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
        assert isinstance(catcher.args[i], tuple)
        assert len(catcher.args[i]) == 1
        assert isinstance(catcher.args[i][0], Tensor)
        assert catcher.args[i][0].shape == (1, 3)
        assert catcher.args[i][0].dtype == dtype.float16

        assert isinstance(catcher.kwargs[i], dict)
        assert not catcher.kwargs[i]
