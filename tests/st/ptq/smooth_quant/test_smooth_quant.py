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
from collections import OrderedDict
import pytest
import numpy as np
import mindspore.communication.management as D
from mindspore.context import ParallelMode
from mindspore.parallel import set_algo_parameters
from mindspore import Tensor, context, save_checkpoint, load_checkpoint, Model, load_param_into_net
from mindspore import nn, Parameter, GRAPH_MODE, dtype
from mindspore.common.dtype import QuantDtype
from mindformers.modules import Linear

from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQConfig, PTQMode
from mindspore_gs.ptq.ptq_config import InnerPTQConfig
from mindspore_gs.ptq.smooth_quant.smooth_quant import SmoothQuant
from mindspore_gs.ptq.smooth_quant.sq_layer_policy import LinearLayerPolicy
from mindspore_gs.ptq.quant_cells import SQLinearWrapper
from mindspore_gs.ptq.convert_utils import QuantCell, DequantBMMCell
from mindspore_gs.ptq.fake_quantizer import MinMaxPerLayer, MinMaxPerChannel

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
# pylint: disable=wrong-import-position
from tests.st.models.llama2 import llama2, create_dummy_inputs
from tests.st.test_utils import check_network_contain_layer, relative_tolerance_acceptable, \
    absolute_tolerance_acceptable


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_constructor():
    """
    Feature: smooth quant algorithm.
    Description: Call constructor of smooth quant and check config.
    Expectation: smooth_quant related is updated according to argument `config` of constructor.
    """
    sq = SmoothQuant()
    assert isinstance(sq._config, InnerPTQConfig)


class SimpleNet(nn.Cell):
    """
    Network with single linear to be quant
    """

    def __init__(self,
                 in_channels=5,
                 out_channels=6,
                 transpose_b=True,
                 strategy=None):
        super().__init__()
        self.linear = Linear(in_channels=in_channels,
                             out_channels=out_channels,
                             transpose_b=transpose_b,
                             bias_init="normal",
                             weight_init="normal")
        if strategy is not None:
            self.linear.shard(strategy)

    def construct(self, x):
        return self.linear(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_apply_convert():
    """
    Feature: test apply and convert api of smooth quant
    Description: Invoke apply and convert api of smooth quant and check network structure.
    Expectation: network structure changed.
    """

    cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND)
    ptq = SmoothQuant(cfg)
    network = SimpleNet()

    network = ptq.apply(network)
    cells: OrderedDict = network.name_cells()
    quant_cell = cells.get("linear", None)
    assert isinstance(quant_cell, SQLinearWrapper)
    weight_fake_quant = quant_cell.weight_quantizer()
    assert isinstance(weight_fake_quant, MinMaxPerChannel)
    assert weight_fake_quant.symmetric()
    assert weight_fake_quant.signed()
    assert weight_fake_quant.quant_dtype() == QuantDtype.INT8
    assert weight_fake_quant.is_per_channel()
    assert not weight_fake_quant.narrow_range()
    assert weight_fake_quant.num_bits() == 8
    act_fake_quant = quant_cell.input_quantizer()
    assert isinstance(act_fake_quant, MinMaxPerLayer)
    assert isinstance(act_fake_quant.symmetric(), bool) and not act_fake_quant.symmetric()
    assert act_fake_quant.quant_dtype() == QuantDtype.INT8
    assert isinstance(act_fake_quant.is_per_channel(), bool) and not act_fake_quant.is_per_channel()
    assert isinstance(act_fake_quant.narrow_range(), bool) and not act_fake_quant.narrow_range()
    assert act_fake_quant.signed()
    assert act_fake_quant.num_bits() == 8
    act_observer = quant_cell._act_observer
    assert isinstance(act_observer, MinMaxPerChannel)
    assert act_observer.symmetric()
    assert act_observer.quant_dtype() == QuantDtype.INT8
    assert act_observer.is_per_channel()
    assert isinstance(act_observer.narrow_range(), bool) and not act_observer.narrow_range()
    assert act_observer.signed()
    assert act_observer.num_bits() == 8
    assert act_observer.axis == 1
    weight_in_observer = quant_cell._weight_in_observer
    assert isinstance(weight_in_observer, MinMaxPerChannel)
    assert weight_in_observer.symmetric()
    assert weight_in_observer.quant_dtype() == QuantDtype.INT8
    assert weight_in_observer.is_per_channel()
    assert isinstance(weight_in_observer.narrow_range(), bool) and not weight_in_observer.narrow_range()
    assert weight_in_observer.signed()
    assert weight_in_observer.num_bits() == 8
    assert weight_in_observer.axis == 1 if network.linear._linear.transpose_b else 0
    assert quant_cell.output_quantizer() is None

    network = ptq.convert(network)
    assert not check_network_contain_layer(network, Linear, (SQLinearWrapper,))
    assert isinstance(network.linear, SQLinearWrapper)
    assert isinstance(network.linear._input_quantizer, QuantCell)
    assert isinstance(network.linear._input_quantizer.t_scale, Parameter)
    assert isinstance(network.linear._input_quantizer.t_zp, Parameter)
    assert isinstance(network.linear._output_quantizer, DequantBMMCell)
    assert isinstance(network.linear._act_observer, MinMaxPerChannel)
    assert isinstance(network.linear._act_observer.float_min, Parameter)
    assert isinstance(network.linear._act_observer.float_max, Parameter)
    assert isinstance(network.linear._weight_in_observer, MinMaxPerChannel)
    assert isinstance(network.linear._weight_in_observer.float_min, Parameter)
    assert isinstance(network.linear._weight_in_observer.float_max, Parameter)
    assert isinstance(network.linear._linear.weight, Parameter)
    assert network.linear._linear.weight.dtype == dtype.int8
    assert network.linear._linear.has_bias


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", ["CPU", "Ascend"])
@pytest.mark.parametrize("mode", [GRAPH_MODE])
@pytest.mark.parametrize("transpose_b", [True, False])
def test_sq_linear_wrapper(device, mode, transpose_b):
    """
    Feature: test FakeQuantizer in SQLinearWrapper.
    Description: Input fake data and check output of each FakeQuantizer.
    Expectation: Same with numpy.
    """
    context.set_context(device_target=device, mode=mode)
    cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND)
    inner_cfg = InnerPTQConfig.inner_config(cfg)
    act_in = 5
    act_out = 6
    linear = Linear(in_channels=act_in, out_channels=act_out, transpose_b=transpose_b, bias_init="normal",
                    weight_init="normal")
    policy = LinearLayerPolicy([], [], inner_cfg)
    sqlinear = SQLinearWrapper(linear, policy, inner_cfg)
    t_x = Tensor(np.random.normal(size=(act_in, act_in)), dtype=dtype.float16)
    t_w = sqlinear._linear.weight
    # observe x
    t_x_fq = sqlinear._act_observer(t_x)
    n_x_fq = t_x.asnumpy()
    act_obv_min = sqlinear._act_observer.float_min.asnumpy()
    act_obv_max = sqlinear._act_observer.float_max.asnumpy()
    act_obv_min_expect = np.min(t_x.asnumpy(), axis=0)
    act_obv_max_expect = np.max(t_x.asnumpy(), axis=0)
    assert np.allclose(act_obv_max, act_obv_max_expect)
    assert np.allclose(act_obv_min, act_obv_min_expect)
    # observe w
    t_w_fq = sqlinear._weight_in_observer(t_w)
    n_w_fq = t_w.asnumpy()
    weight_obv_min = sqlinear._weight_in_observer.float_min.asnumpy()
    weight_obv_max = sqlinear._weight_in_observer.float_max.asnumpy()
    weight_obv_min_expect = np.min(t_w.asnumpy(), axis=0 if transpose_b else 1)
    weight_obv_max_expect = np.max(t_w.asnumpy(), axis=0 if transpose_b else 1)
    assert np.allclose(weight_obv_min, weight_obv_min_expect)
    assert np.allclose(weight_obv_max, weight_obv_max_expect)
    # calculate smooth scale
    t_smooth_scale = sqlinear._calc_input_scale()
    act_maxnorm = np.maximum(np.abs(act_obv_min), np.abs(act_obv_max))
    act_maxnorm_pow = np.power(act_maxnorm, sqlinear._alpha)
    weight_maxnorm = np.maximum(np.abs(weight_obv_min), np.abs(weight_obv_max))
    weight_maxnorm_pow = np.power(weight_maxnorm, sqlinear._alpha)
    n_smooth_scale = np.clip(act_maxnorm_pow / weight_maxnorm_pow, 1e-5, None)
    n_smooth_scale[act_maxnorm_pow == 0] = 1.0
    n_smooth_scale[weight_maxnorm_pow == 0] = 1.0
    assert np.allclose(t_smooth_scale.asnumpy(), n_smooth_scale)
    # smooth x and fq x
    n_x_smooth = n_x_fq / n_smooth_scale
    t_x_smooth = sqlinear._act_mul(t_x_fq, sqlinear._div(1.0, t_smooth_scale))
    assert np.allclose(t_x_smooth.asnumpy(), n_x_smooth)
    t_x_smooth_fq = sqlinear._input_quantizer(t_x_smooth)
    n_x_smooth_fq = n_x_smooth
    assert np.allclose(t_x_smooth_fq.asnumpy(), n_x_smooth_fq)
    x_q_min = sqlinear._input_quantizer.float_min.asnumpy()
    x_q_max = sqlinear._input_quantizer.float_max.asnumpy()
    x_q_min_expect = np.min(n_x_smooth)
    x_q_max_expect = np.max(n_x_smooth)
    assert np.allclose(x_q_min, x_q_min_expect)
    assert np.allclose(x_q_max, x_q_max_expect)
    t_x_restored = sqlinear._act_mul(t_x_smooth_fq, t_smooth_scale)
    n_x_restored = n_x_smooth_fq * n_smooth_scale
    assert np.allclose(t_x_restored.asnumpy(), n_x_restored)
    # smooth w and fq w
    if transpose_b:
        t_w_smooth_scale = t_smooth_scale
        n_w_smooth_scale = n_smooth_scale
    else:
        t_w_smooth_scale = sqlinear._expand(t_smooth_scale, 1)
        n_w_smooth_scale = np.expand_dims(n_smooth_scale, 1)
    n_w_smooth = n_w_fq * n_w_smooth_scale
    t_w_smooth = sqlinear._weight_mul(t_w_fq, t_w_smooth_scale)
    assert np.allclose(t_w_smooth.asnumpy(), n_w_smooth)
    n_w_smooth_fq = n_w_smooth
    t_w_smooth_fq = sqlinear._weight_quantizer(t_w_smooth)
    assert np.allclose(t_w_smooth_fq.asnumpy(), n_w_smooth_fq)
    weight_q_min = sqlinear._weight_quantizer.float_min.asnumpy()
    weight_q_max = sqlinear._weight_quantizer.float_max.asnumpy()
    weight_q_min_expect = np.min(n_w_smooth_fq, axis=1 if transpose_b else 0)
    weight_q_max_expect = np.max(n_w_smooth_fq, axis=1 if transpose_b else 0)
    assert np.allclose(weight_q_min, weight_q_min_expect)
    assert np.allclose(weight_q_max, weight_q_max_expect)
    t_w_restored = sqlinear._weight_div(t_w_smooth_fq, t_w_smooth_scale)
    n_w_restored = n_w_smooth_fq / n_w_smooth_scale
    assert np.allclose(t_w_restored.asnumpy(), n_w_restored)


def sq_predict_simplenet_2stage(device, mode, transpose_b, model_parallel, p_strategy):
    """test_sq_predict_simplenet_2stage"""
    print(f"---------------- Testing params: {device} {mode} {transpose_b} {model_parallel} {p_strategy}", flush=True)
    use_parallel = model_parallel > 1
    act_in, act_out = 4, 8
    weight_in, weight_out = act_in, act_out
    rank_id = 0
    if use_parallel:
        rank_id = os.getenv('RANK_ID')
        strategy_ckpt_save_file = "test_sq_predict_simplenet_2stage_parallel_strategy.ckpt"

    cur_dir, _ = os.path.split(os.path.abspath(__file__))
    fp_ckpt_path = os.path.join(cur_dir, "../../../data/test_ckpt/test_sq_predict_simplenet_2stage_fp.ckpt")
    input_path = os.path.join(cur_dir, "../../../data/test_input/test_sq_predict_simplenet_2stage.npy")
    ckpt_path = f"test_sq_predict_simplenet_2stage_int8_rank{rank_id}.ckpt"
    if transpose_b:
        fp_ckpt_path = os.path.join(cur_dir, "../../../data/test_ckpt/test_sq_predict_simplenet_2stage_fp.ckpt")
    else:
        fp_ckpt_path = os.path.join(cur_dir, "../../../data/test_ckpt/test_sq_predict_simplenet_2stage_fp_notranb.ckpt")

    context.set_context(device_target=device, mode=mode)
    if use_parallel:
        D.init()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, gradients_mean=False,
                                          full_batch=True, strategy_ckpt_save_file=strategy_ckpt_save_file)
        set_algo_parameters(elementwise_op_strategy_follow=True)

    def quant(input_):
        cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND)
        ptq = SmoothQuant(cfg)

        network = SimpleNet(in_channels=weight_in, out_channels=weight_out, transpose_b=transpose_b,
                            strategy=p_strategy)
        load_checkpoint(fp_ckpt_path, network)
        network = ptq.apply(network)

        def _calibrate(net, calibrate_size):
            for _ in range(calibrate_size):
                example = Tensor(np.load(input_path), dtype=dtype.float16)
                _ = net(example)

        _calibrate(network, 2)
        network = ptq.convert(network)
        save_checkpoint(network.parameters_dict(), ckpt_path, integrated_save=False)

        fp_network = SimpleNet(in_channels=weight_in, out_channels=weight_out, transpose_b=transpose_b,
                               strategy=p_strategy)
        load_checkpoint(fp_ckpt_path, fp_network)
        return fp_network(input_)

    def infer(input_):
        context.set_context(mode=mode)
        cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND)
        ptq = SmoothQuant(cfg)
        network = SimpleNet(in_channels=weight_in, out_channels=weight_out, transpose_b=transpose_b,
                            strategy=p_strategy)
        network = ptq.apply(network)
        network = ptq.convert(network)
        assert network.linear._linear.has_bias and network.linear._linear.bias is not None
        if use_parallel:
            model = Model(network)
            model.infer_predict_layout(input_)
        param_dict = load_checkpoint(ckpt_path)
        unused_param_names = ['linear._scale_store', 'linear._act_observer.float_min', 'linear._act_observer.float_max',
                              'linear._weight_in_observer.float_min', 'linear._weight_in_observer.float_max']
        for item in unused_param_names:
            param_dict.pop(item)
        load_param_into_net(network, param_dict)
        return network(input_)

    example = Tensor(np.load(input_path), dtype=dtype.float16)
    foutput = quant(example)
    qoutput = infer(example)
    res = relative_tolerance_acceptable(qoutput[1].asnumpy(), foutput[1].asnumpy(), 7e-3)
    if not res:
        return False
    if use_parallel:
        return absolute_tolerance_acceptable(qoutput[1].asnumpy(), foutput[1].asnumpy(), 0.5)
    return absolute_tolerance_acceptable(qoutput[1].asnumpy(), foutput[1].asnumpy(), 11e-5)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", ["Ascend"])
@pytest.mark.parametrize("mode", [GRAPH_MODE])
@pytest.mark.parametrize("transpose_b", [True, False])
def test_sq_predict_simplenet_2stage(device, mode, transpose_b):
    """
    Feature: test smooth quant adjust parameter in two stages.
    Description: Feed invalid type of bn_fold to convert function.
    Expectation: adjust error is in certain range.
    """

    model_parallel = int(os.environ.get("sq_test_model_parallel", 1))
    if model_parallel == 1:
        assert sq_predict_simplenet_2stage(device, mode, transpose_b, 1, None)
        return

    p_strategies = [((1, model_parallel), (model_parallel, 1)),
                    ((1, 1), (1, model_parallel)),
                    ((model_parallel, 1), (1, 1)),
                    ((1, 1), (1, 1))]
    for p_strategy in p_strategies:
        if transpose_b and p_strategy is not None:
            weight_strategy = p_strategy[1]
            new_weight_strategy = (weight_strategy[1], weight_strategy[0])
            p_strategy = (p_strategy[0], new_weight_strategy)
        assert sq_predict_simplenet_2stage(device, mode, transpose_b, model_parallel, p_strategy)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_sq_predict_simplenet_2stage_2p():
    """
    Feature: test smooth quant adjust parameter in two stages with two cards.
    Description: apply SQ on simplenet and check accuracy.
    Expectation: accuracy is good.
    """
    model_parallel = 2
    os.environ['sq_test_model_parallel'] = str(model_parallel)
    return_code = os.system(
        "msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 "
        "--master_port=10926 --join=True --log_dir=./test_sq_predict_simplenet_logs "
        "pytest -s test_smooth_quant.py::test_sq_predict_simplenet_2stage"
    )
    if return_code != 0:
        log_file = open("./test_sq_predict_simplenet_logs/worker_1.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()

    assert return_code == 0


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", ["Ascend"])
@pytest.mark.parametrize("mode", [GRAPH_MODE])
def test_sq_predict_llama2_2stage(device, mode):
    """
    Feature: test smooth quant adjust parameter in two stages.
    Description: Feed invalid type of bn_fold to convert function.
    Expectation: adjust error is in certain range.
    """

    ckpt_path = "test_sq_predict_llama2_2stage_int8.ckpt"

    def quant(inputs):
        context.set_context(device_target=device, mode=mode)
        cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND)
        ptq = SmoothQuant(cfg)

        network = llama2(2, 512, 2048, 2)
        network = ptq.apply(network)

        def _calibrate(net, calibrate_size):
            for _ in range(calibrate_size):
                example = create_dummy_inputs(2, 512, 512)
                _ = net(*example)

        _calibrate(network, 2)
        network = ptq.convert(network)
        save_checkpoint(network, ckpt_path)

        fp_network = llama2(2, 512, 2048, 2)
        return fp_network(*inputs)

    def infer(inputs):
        context.set_context(device_target=device, mode=mode)
        cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND)
        ptq = SmoothQuant(cfg)
        network = llama2(2, 512, 2048, 2)
        network = ptq.apply(network)
        network = ptq.convert(network)
        load_checkpoint(ckpt_path, network)
        return network(*inputs)

    inputs = create_dummy_inputs(2, 512, 512)
    foutput = quant(inputs)
    qoutput = infer(inputs)
    print(f"-------------------foutput {foutput}", flush=True)
    print(f"-------------------qoutput {qoutput}", flush=True)
    print(f"rel error: {relative_tolerance_acceptable(qoutput[1].asnumpy(), foutput[1].asnumpy(), 10)}")
    print(f"abs error: {absolute_tolerance_acceptable(qoutput[1].asnumpy(), foutput[1].asnumpy(), 10)}")
