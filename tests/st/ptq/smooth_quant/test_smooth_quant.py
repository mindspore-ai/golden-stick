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
import warnings

import os
import sys
from collections import OrderedDict
import pytest
import numpy as np
import mindspore.communication.management as D
from mindspore.context import ParallelMode
from mindspore.nn import Cell
from mindspore.dataset import GeneratorDataset
from mindspore.parallel import set_algo_parameters
from mindspore import Tensor, context, save_checkpoint, load_checkpoint, Model, load_param_into_net
from mindspore import nn, Parameter, GRAPH_MODE, dtype, ops
from mindspore.communication import get_rank
from mindformers.modules import Linear
from mindformers.trainer.utils import transform_and_load_checkpoint

from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQConfig, PTQMode
from mindspore_gs.ptq.ptq_config import InnerPTQConfig, SmoothQuantConfig, PTQApproach, OutliersSuppressionType
from mindspore_gs.ptq.smooth_quant.smooth_quant import SmoothQuant
from mindspore_gs.ptq.smooth_quant.quant_cells.mindformers.layer_policys import LinearLayerPolicy
from mindspore_gs.ptq.smooth_quant.quant_cells.mindformers.quant_cells import SQLinearActObserver, \
    SQLinearWeightObserver, SQLinearWrapper
from mindspore_gs.ptq.convert_utils import SmoothAndQuantCell, DequantBMMCell
from mindspore_gs.ptq.fake_quantizer import MinMaxPerLayer, MinMaxPerChannel
from mindspore_gs.ptq.network_helpers import NetworkHelper
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
# pylint: disable=wrong-import-position
from tests.st.test_utils import check_network_contain_layer, relative_tolerance_acceptable, \
    absolute_tolerance_acceptable
from tests.st.mindformers_utils import create_hello_ds


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_constructor():
    """
    Feature: smooth quant algorithm.
    Description: Call constructor of smooth quant and check config.
    Expectation: smooth_quant related is updated according to argument `config` of constructor.
    """
    context.set_context(device_target="CPU")
    config = PTQConfig(act_quant_dtype=dtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH)
    sq = SmoothQuant(config)
    assert isinstance(sq._config, InnerPTQConfig)
    assert sq._config.algo_args.get("alpha", None) == 0.5
    assert sq._config.act_quant_dtype == dtype.int8
    assert sq._config.weight_quant_dtype == dtype.int8
    assert sq._config.kvcache_quant_dtype is None

    sq_args = SmoothQuantConfig(alpha=0.8)
    cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, act_quant_dtype=dtype.int8,
                    outliers_suppression=OutliersSuppressionType.SMOOTH, algo_args=sq_args)
    sq = SmoothQuant(cfg)
    assert isinstance(sq._config, InnerPTQConfig)
    assert sq._config.algo_args.get("alpha", None) == 0.8
    assert sq._config.act_quant_dtype == dtype.int8
    assert sq._config.weight_quant_dtype == dtype.int8
    assert sq._config.kvcache_quant_dtype is None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ptq_config_error():
    """
    Feature: simulated SmoothQuant _ptq_config_check function.
    Description: Feed invalid value of PTQConfig to _ptq_config_check function.
    Expectation: Except ValueError.
    """
    context.set_context(device_target="CPU")
    config = PTQConfig(act_quant_dtype=dtype.int8, weight_quant_dtype=None)
    with pytest.raises(ValueError):
        _ = SmoothQuant(config)

    config = PTQConfig()
    with pytest.raises(ValueError):
        _ = SmoothQuant(config)

    config = PTQConfig(weight_quant_dtype=None, kvcache_quant_dtype=dtype.int8)
    with pytest.raises(ValueError):
        _ = SmoothQuant(config)

    config = PTQConfig(act_quant_dtype=dtype.int8, weight_quant_dtype=None,
                       kvcache_quant_dtype=dtype.int8)
    with pytest.raises(ValueError):
        _ = SmoothQuant(config)



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


#pylint: disable=w0223
class SimpleNetworkHelper(NetworkHelper):
    """SimpleNetworkHelper"""
    def __init__(self, **kwargs):
        self.attrs = kwargs

    def get_spec(self, name: str):
        return self.attrs.get(name, None)

    def create_tokenizer(self, **kwargs):
        return None

    def generate(self, network: Cell, input_ids: np.ndarray, max_new_tokens=1, **kwargs):
        input_ids = np.pad(input_ids, ((0, 0), (0, self.get_spec("seq_length") - input_ids.shape[1])), 'constant',
                           constant_values=0)
        network(Tensor(input_ids, dtype=dtype.float16))

    def assemble_inputs(self, input_ids: np.ndarray, **kwargs):
        raise RuntimeError("InnerError, should not invoke SimpleNetworkHelper.assemble_inputs()")


def create_simple_ds(np_paths: [str], repeat=1):
    """create_simple_ds"""
    class SimpleIterable:
        """SimpleIterable"""
        def __init__(self, np_paths: [str], repeat=1):
            self._index = 0
            self.data = []
            for _ in range(repeat):
                for np_path in np_paths:
                    self.data.append(np.load(np_path).astype(np.float16))

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

    return GeneratorDataset(source=SimpleIterable(np_paths, repeat), column_names=["input_ids"])


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
# FIXME: wait for cpu round ops bugfix
@pytest.mark.parametrize("device_target", ['Ascend'])
def test_apply_convert(device_target):
    """
    Feature: test apply and convert api of smooth quant
    Description: Invoke apply and convert api of smooth quant and check network structure.
    Expectation: network structure changed.
    """
    context.set_context(device_target=device_target)
    cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND,
                    act_quant_dtype=dtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH)
    ptq = SmoothQuant(cfg)
    network = SimpleNet(transpose_b=False)
    net_helper = SimpleNetworkHelper(batch_size=1, seq_length=5)
    ds = create_foo_ds(1)
    network = ptq.apply(network, net_helper, ds)
    cells: OrderedDict = network.name_cells()
    quant_cell = cells.get("linear", None)
    assert isinstance(quant_cell, SQLinearWrapper)
    weight_fake_quant = quant_cell.weight_quantizer()
    assert isinstance(weight_fake_quant, MinMaxPerChannel)
    assert weight_fake_quant.symmetric()
    assert weight_fake_quant.signed()
    assert weight_fake_quant.quant_dtype() == dtype.int8
    assert weight_fake_quant.is_per_channel()
    assert not weight_fake_quant.narrow_range()
    assert weight_fake_quant.num_bits() == 8
    weight_fake_quant.foo_init()
    act_fake_quant = quant_cell.input_quantizer()
    assert isinstance(act_fake_quant, MinMaxPerLayer)
    assert isinstance(act_fake_quant.symmetric(), bool) and not act_fake_quant.symmetric()
    assert act_fake_quant.quant_dtype() == dtype.int8
    assert isinstance(act_fake_quant.is_per_channel(), bool) and not act_fake_quant.is_per_channel()
    assert isinstance(act_fake_quant.narrow_range(), bool) and not act_fake_quant.narrow_range()
    assert act_fake_quant.signed()
    assert act_fake_quant.num_bits() == 8
    act_fake_quant.foo_init()
    act_observer = quant_cell.act_observer
    assert isinstance(act_observer, MinMaxPerChannel)
    assert act_observer.symmetric()
    assert act_observer.quant_dtype() == dtype.int8
    assert act_observer.is_per_channel()
    assert isinstance(act_observer.narrow_range(), bool) and not act_observer.narrow_range()
    assert act_observer.signed()
    assert act_observer.num_bits() == 8
    assert act_observer.axis == 1
    weight_in_observer = quant_cell.weight_observer
    assert isinstance(weight_in_observer, MinMaxPerChannel)
    assert weight_in_observer.symmetric()
    assert weight_in_observer.quant_dtype() == dtype.int8
    assert weight_in_observer.is_per_channel()
    assert isinstance(weight_in_observer.narrow_range(), bool) and not weight_in_observer.narrow_range()
    assert weight_in_observer.signed()
    assert weight_in_observer.num_bits() == 8
    assert weight_in_observer.axis == (1 if network.linear.handler().transpose_b else 0)
    assert quant_cell.output_quantizer() is None

    network = ptq.convert(network)
    assert not check_network_contain_layer(network, Linear, (SQLinearWrapper,))
    assert isinstance(network.linear, SQLinearWrapper)
    assert isinstance(network.linear._input_quantizer, SmoothAndQuantCell)
    assert isinstance(network.linear._input_quantizer.t_scale, Parameter)
    assert isinstance(network.linear._input_quantizer.t_zp, Parameter)
    assert isinstance(network.linear._output_quantizer, DequantBMMCell)
    assert isinstance(network.linear.act_observer, MinMaxPerChannel)
    assert isinstance(network.linear.act_observer.float_min, Parameter)
    assert isinstance(network.linear.act_observer.float_max, Parameter)
    assert isinstance(network.linear.weight_observer, MinMaxPerChannel)
    assert isinstance(network.linear.weight_observer.float_min, Parameter)
    assert isinstance(network.linear.weight_observer.float_max, Parameter)
    assert isinstance(network.linear.handler().weight, Parameter)
    assert network.linear.handler().weight.dtype == dtype.int8
    assert network.linear.handler().has_bias


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_apply_convert_error():
    """
    Feature: SmoothQuant algorithm set functions.
    Description: Apply SmoothQuant on SimpleNet with error arguments.
    Expectation: raise error.
    """
    context.set_context(device_target="CPU")
    ptq = SmoothQuant(PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND,
                                act_quant_dtype=dtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH))
    network = SimpleNet()
    net_helper = SimpleNetworkHelper(batch_size=1, seq_length=1)
    ds = create_foo_ds(1)
    with pytest.raises(TypeError, match="Type of network should be"):
        ptq.apply(1, net_helper, ds)
    with pytest.raises(TypeError, match="Type of network_helper should be"):
        ptq.apply(network, 1, ds)
    with pytest.raises(TypeError, match="Type of datasets should be"):
        ptq.apply(network, net_helper, 1)
    with pytest.raises(TypeError, match="Type of net_opt should be"):
        ptq.convert(1)
    with pytest.raises(TypeError, match="Type of ckpt_path should be"):
        ptq.convert(network, 1)


class NoQuantNet(nn.Cell):
    """
    Network with no linear to be quant
    """
    def construct(self, x):
        return ops.add(x, x)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
# FIXME: wait for cpu round ops bugfix
@pytest.mark.parametrize("device_target", ['Ascend'])
def test_nothing_to_apply_convert(device_target):
    """
    Feature: SmoothQuant algorithm set functions.
    Description: Apply SmoothQuant on NoQuantNet.
    Expectation: warning log.
    """
    context.set_context(device_target=device_target)
    ptq = SmoothQuant(PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND,
                                act_quant_dtype=dtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH))
    network = NoQuantNet()
    net_helper = SimpleNetworkHelper(batch_size=1, seq_length=1)
    ds = create_foo_ds(1)
    with pytest.warns(expected_warning=RuntimeWarning, match="No layer found in network is suitable for quantization"):
        ptq.apply(network, net_helper, ds)
    network = NoQuantNet()
    with pytest.warns(expected_warning=RuntimeWarning, match="and make sure call apply before convert"):
        ptq.convert(network)
    network = SimpleNet()
    net_helper = SimpleNetworkHelper(batch_size=1, seq_length=5)
    ds = create_foo_ds(1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        network = ptq.apply(network, net_helper, ds)
        ptq.convert(network)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [GRAPH_MODE])
@pytest.mark.parametrize("transpose_b", [True, False])
def test_sq_linear_wrapper(mode, transpose_b):
    """
    Feature: test FakeQuantizer in SQLinearWrapper.
    Description: Input fake data and check output of each FakeQuantizer.
    Expectation: Same with numpy.
    """
    context.set_context(device_target="Ascend", mode=mode)
    cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND,
                    act_quant_dtype=dtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH)
    inner_cfg = InnerPTQConfig.inner_config(cfg, PTQApproach.SMOOTH_QUANT)
    act_in = 5
    act_out = 6
    linear = Linear(in_channels=act_in, out_channels=act_out, transpose_b=transpose_b, bias_init="normal",
                    weight_init="normal")
    policy = LinearLayerPolicy([], [], inner_cfg)
    sq_act_obs = SQLinearActObserver(linear, policy, inner_cfg)
    t_x = Tensor(np.random.normal(size=(act_in, act_in)), dtype=dtype.float16)
    t_w = sq_act_obs.handler().weight
    # observe x
    t_x_fq = sq_act_obs.act_observer(t_x)
    n_x_fq = t_x.asnumpy()
    act_obv_min = sq_act_obs.act_observer.float_min.asnumpy()
    act_obv_max = sq_act_obs.act_observer.float_max.asnumpy()
    act_obv_min_expect = np.min(t_x.asnumpy(), axis=0)
    act_obv_max_expect = np.max(t_x.asnumpy(), axis=0)
    assert np.allclose(act_obv_max, act_obv_max_expect)
    assert np.allclose(act_obv_min, act_obv_min_expect)
    # observe w
    sq_weight_obs = SQLinearWeightObserver(sq_act_obs)
    t_w_fq = sq_weight_obs.weight_observer(t_w)
    n_w_fq = t_w.asnumpy()
    weight_obv_min = sq_weight_obs.weight_observer.float_min.asnumpy()
    weight_obv_max = sq_weight_obs.weight_observer.float_max.asnumpy()
    weight_obv_min_expect = np.min(t_w.asnumpy(), axis=0 if transpose_b else 1)
    weight_obv_max_expect = np.max(t_w.asnumpy(), axis=0 if transpose_b else 1)
    assert np.allclose(weight_obv_min, weight_obv_min_expect)
    assert np.allclose(weight_obv_max, weight_obv_max_expect)
    # calculate smooth scale
    t_smooth_scale = sq_weight_obs._calc_smooth_scale()
    act_maxnorm = np.maximum(np.abs(act_obv_min), np.abs(act_obv_max))
    act_maxnorm_pow = np.power(act_maxnorm, sq_weight_obs._alpha)
    weight_maxnorm = np.maximum(np.abs(weight_obv_min), np.abs(weight_obv_max))
    weight_maxnorm_pow = np.power(weight_maxnorm, sq_weight_obs._alpha)
    n_smooth_scale = np.clip(act_maxnorm_pow / weight_maxnorm_pow, 1e-5, None)
    n_smooth_scale[act_maxnorm_pow == 0] = 1.0
    n_smooth_scale[weight_maxnorm_pow == 0] = 1.0
    assert np.allclose(t_smooth_scale.asnumpy(), n_smooth_scale)
    # smooth w and fq w
    if transpose_b:
        t_w_smooth_scale = t_smooth_scale
        n_w_smooth_scale = n_smooth_scale
    else:
        t_w_smooth_scale = sq_weight_obs._expand(t_smooth_scale, 1)
        n_w_smooth_scale = np.expand_dims(n_smooth_scale, 1)
    n_w_smooth = n_w_fq * n_w_smooth_scale
    t_w_smooth = sq_weight_obs._weight_mul(t_w_fq, t_w_smooth_scale)
    assert np.allclose(t_w_smooth.asnumpy(), n_w_smooth)
    n_w_smooth_fq = n_w_smooth
    t_w_smooth_fq = sq_weight_obs.weight_quantizer_(t_w_smooth)
    assert np.allclose(t_w_smooth_fq.asnumpy(), n_w_smooth_fq)
    weight_q_min = sq_weight_obs.weight_quantizer_.float_min.asnumpy()
    weight_q_max = sq_weight_obs.weight_quantizer_.float_max.asnumpy()
    weight_q_min_expect = np.min(n_w_smooth_fq, axis=1 if transpose_b else 0)
    weight_q_max_expect = np.max(n_w_smooth_fq, axis=1 if transpose_b else 0)
    assert np.allclose(weight_q_min, weight_q_min_expect)
    assert np.allclose(weight_q_max, weight_q_max_expect)
    t_w_restored = sq_weight_obs._weight_div(t_w_smooth_fq, t_w_smooth_scale)
    n_w_restored = n_w_smooth_fq / n_w_smooth_scale
    assert np.allclose(t_w_restored.asnumpy(), n_w_restored)
    # smooth x and fq x
    sqlinear = SQLinearWrapper(sq_weight_obs)
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


def sq_predict_simplenet_2stage(device, mode, transpose_b, model_parallel, p_strategy):
    """test_sq_predict_simplenet_2stage"""
    os.environ['GRAPH_OP_RUN'] = "1"
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    print(f"---------------- Testing params: {device} {mode} {transpose_b} {model_parallel} {p_strategy}", flush=True)
    use_parallel = model_parallel > 1
    act_in, act_out = 4, 8
    weight_in, weight_out = act_in, act_out
    rank_id = 0
    if use_parallel:
        rank_id = os.getenv('RANK_ID')
        strategy_ckpt_save_file = "test_sq_predict_simplenet_2stage_parallel_strategy.ckpt"

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    fp_ckpt_path = os.path.join(cur_dir, "../../../data/test_ckpt/test_sq_predict_simplenet_2stage_fp.ckpt")
    input_path = os.path.join(cur_dir, "../../../data/test_input/test_sq_predict_simplenet_2stage.npy")
    ckpt_path = f"test_sq_predict_simplenet_2stage_int8_rank{rank_id}.ckpt"
    if transpose_b:
        fp_ckpt_path = os.path.join(cur_dir, "../../../data/test_ckpt/test_sq_predict_simplenet_2stage_fp.ckpt")
    else:
        fp_ckpt_path = os.path.join(cur_dir, "../../../data/test_ckpt/test_sq_predict_simplenet_2stage_fp_notranb.ckpt")

    context.set_context(device_target=device, mode=mode, jit_config={"jit_level": "O0", "infer_boost": "on"})
    if use_parallel:
        D.init()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, gradients_mean=False,
                                          full_batch=True, strategy_ckpt_save_file=strategy_ckpt_save_file)
        set_algo_parameters(elementwise_op_strategy_follow=True)

    def quant(input_):
        cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND,
                        act_quant_dtype=dtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH)
        ptq = SmoothQuant(cfg)

        network = SimpleNet(in_channels=weight_in, out_channels=weight_out, transpose_b=transpose_b,
                            strategy=p_strategy)
        load_checkpoint(fp_ckpt_path, network)
        net_helper = SimpleNetworkHelper(batch_size=act_out, seq_length=act_in)
        ds = create_simple_ds([input_path], 2)
        network = ptq.apply(network, net_helper, datasets=ds)
        network = ptq.convert(network)
        save_checkpoint(network.parameters_dict(), ckpt_path, integrated_save=False)

        fp_network = SimpleNet(in_channels=weight_in, out_channels=weight_out, transpose_b=transpose_b,
                               strategy=p_strategy)
        load_checkpoint(fp_ckpt_path, fp_network)
        return fp_network(input_)

    def infer(input_):
        context.set_context(mode=mode)
        cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND,
                        act_quant_dtype=dtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH)
        ptq = SmoothQuant(cfg)
        network = SimpleNet(in_channels=weight_in, out_channels=weight_out, transpose_b=transpose_b,
                            strategy=p_strategy)
        network = ptq.apply(network)
        network = ptq.convert(network)
        assert network.linear.handler().has_bias and network.linear.handler().bias is not None
        if use_parallel:
            model = Model(network)
            model.infer_predict_layout(input_)
        param_dict = load_checkpoint(ckpt_path)
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


# @pytest.mark.level0
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
    use_parallel = os.environ.get("sq_test_use_parallel", '0')
    if use_parallel == '0':
        assert sq_predict_simplenet_2stage(device, mode, transpose_b, 1, None)
        return

    p_strategies = [((1, 2), (2, 1)), ((1, 1), (1, 2)), ((2, 1), (1, 1)), ((1, 1), (1, 1))]
    for p_strategy in p_strategies:
        if transpose_b and p_strategy is not None:
            weight_strategy = p_strategy[1]
            new_weight_strategy = (weight_strategy[1], weight_strategy[0])
            p_strategy = (p_strategy[0], new_weight_strategy)
        assert sq_predict_simplenet_2stage(device, mode, transpose_b, 2, p_strategy)


# @pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_sq_predict_simplenet_2stage_2p():
    """
    Feature: test smooth quant adjust parameter in two stages with two cards.
    Description: apply SQ on simplenet and check accuracy.
    Expectation: accuracy is good.
    """
    os.environ['sq_test_use_parallel'] = '1'
    cur_file = os.path.abspath(__file__)
    return_code = os.system(
        f"msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 "
        "--master_port=10926 --join=True --log_dir=./test_sq_predict_simplenet_logs "
        f"pytest -s {cur_file}::test_sq_predict_simplenet_2stage"
    )
    if return_code != 0:
        log_file = open("./test_sq_predict_simplenet_logs/worker_1.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()

    assert return_code == 0


def sq_predict_llama2_2stage():
    """test_sq_predict_llama2_2stage"""
    device = str(os.environ.get("device"))
    if os.environ.get("mode") == "GRAPH_MODE":
        mode = GRAPH_MODE
    else:
        raise ValueError("SmoothQuant do not support PYNATIVE_MODE now!")
    model_parallel = int(os.environ.get("sq_test_model_parallel", 1))
    os.environ['GRAPH_OP_RUN'] = "1"
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    print(f"---------------- Testing params: {device} {mode} ", flush=True)
    context.set_context(device_target=device, mode=mode, jit_config={"jit_level": "O0", "infer_boost": "on"})
    if model_parallel == 1:
        config_path = "../../../data/test_llama2/predict_llama2_13b_1p.yaml"
        fp16_ckpt_path = "../../../data/test_llama2/llama2-13b-1p"
        w8a8_ckpt_path = "../../../data/test_llama2/llama2-13b-w8a8-1p"
    else:
        config_path = "../../../data/test_llama2/predict_llama2_13b_2p.yaml"
        fp16_ckpt_path = "../../../data/test_llama2/llama2-13b-2p"
        w8a8_ckpt_path = "../../../data/test_llama2/llama2-13b-w8a8-2p"

    cur_dir_ = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(cur_dir_, config_path)
    vocab_file = os.path.join(cur_dir_, "../../../data/llama2-tokenizer.model")
    fp16_ckpt_path = os.path.join(cur_dir_, fp16_ckpt_path)
    w8a8_ckpt_path = os.path.join(cur_dir_, w8a8_ckpt_path)

    helper = MFLlama2Helper(config_path)
    helper.mf_config.processor.tokenizer.vocab_file = vocab_file
    device_id = int(os.environ.get('DEVICE_ID', '0'))
    helper.mf_config.context.device_id = device_id

    def quant():
        helper.mf_config.load_checkpoint = fp16_ckpt_path
        network = helper.create_network()
        tokenizer = helper.create_tokenizer()
        cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, opname_blacklist=["w2", "lm_head"],
                        act_quant_dtype=dtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH)
        ptq = SmoothQuant(config=cfg)
        ds = create_hello_ds(tokenizer, 1)
        network = ptq.apply(network, helper, datasets=ds)
        network = ptq.convert(network)
        try:
            rank_id = get_rank()
        except RuntimeError:
            rank_id = 0
        save_path = os.path.join(w8a8_ckpt_path, f"rank_{rank_id}")
        os.makedirs(save_path, exist_ok=True)
        save_checkpoint(network.parameters_dict(), os.path.join(save_path, "quant.ckpt"),
                        choice_func=lambda x: "key_cache" not in x and "value_cache" not in x)

    def w8a8_infer(input_):
        helper.mf_config.load_checkpoint = ""
        network = helper.create_network()
        cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, opname_blacklist=["w2", "lm_head"],
                        act_quant_dtype=dtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH)
        ptq = SmoothQuant(config=cfg)
        network = ptq.apply(network)
        network = ptq.convert(network)

        helper.mf_config.load_checkpoint = w8a8_ckpt_path
        model = Model(network)
        input_ids = np.ones(shape=[helper.get_spec('batch_size'), helper.get_spec('seq_length')], dtype=np.int32)
        infer_data = network.prepare_inputs_for_predict_layout(input_ids)
        if helper.mf_config.use_parallel:
            network.phase = 'infer_predict_layout'
            model.infer_predict_layout(*infer_data)
        transform_and_load_checkpoint(helper.mf_config, model, network, infer_data, do_predict=True)

        seq_len = 500
        tokenizer = helper.create_tokenizer()
        input_ids = tokenizer(input_)['input_ids']
        outputs = network.generate(input_ids, do_sample=False, max_length=seq_len, top_p=1, top_k=3)
        answer = tokenizer.decode(outputs, skip_special_tokens=True)
        return outputs, answer

    def fp16_infer(input_):
        helper.mf_config.load_checkpoint = fp16_ckpt_path
        network = helper.create_network()
        tokenizer = helper.create_tokenizer()

        seq_len = 500
        input_ids = tokenizer(input_)['input_ids']
        outputs = network.generate(input_ids, do_sample=False, max_length=seq_len, top_p=1, top_k=3)
        answer = tokenizer.decode(outputs, skip_special_tokens=True)
        return outputs, answer

    example = "Hello"
    quant()
    foutput, _ = fp16_infer(example)
    qoutput, _ = w8a8_infer(example)
    npfoutput = np.array(foutput)
    npqoutput = np.array(qoutput)
    if model_parallel == 1:
        res = np.allclose(npqoutput[:, :5], npfoutput[:, :5], 0, 0)
    elif model_parallel == 2:
        res = np.allclose(npqoutput, npfoutput, 0, 0)
    if not res:
        print(f"npfoutput: {npfoutput}", flush=True)
        print(f"npqoutput: {npqoutput}", flush=True)
        print(f"First not equal index {np.min(np.where((npfoutput - npqoutput) != 0))}", flush=True)
    return res


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", ["Ascend"])
@pytest.mark.parametrize("mode", ["GRAPH_MODE"])
def test_sq_llama2_predict_2stage_1p(device, mode):
    """
    Feature: test smooth quant adjust parameter in two stages with one cards.
    Description: apply SQ on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    os.environ['device'] = device
    os.environ['mode'] = mode

    assert sq_predict_llama2_2stage()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
@pytest.mark.parametrize("device", ["Ascend"])
@pytest.mark.parametrize("mode", ["GRAPH_MODE"])
def test_sq_llama2_predict_2stage_2p(device, mode):
    """
    Feature: test smooth quant adjust parameter in two stages with two cards.
    Description: apply SQ on llama2 and check accuracy.
    Expectation: accuracy is good.
    """
    os.environ['sq_test_model_parallel'] = "2"
    os.environ['device'] = device
    os.environ['mode'] = mode

    cur_file = os.path.abspath(__file__)
    return_code = os.system(
        f"msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 "
        "--master_port=10926 --join=True --log_dir=./test_sq_predict_llama2_13b_logs "
        f"pytest -s {cur_file}::test_sq_llama2_predict_2stage_1p"
    )
    if return_code != 0:
        log_file = open("./test_sq_predict_llama2_13b_logs/worker_1.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()

    assert return_code == 0
