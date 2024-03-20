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
from mindspore import (Tensor, context, save_checkpoint, load_checkpoint)
from mindspore import nn, Parameter, GRAPH_MODE, dtype
from mindspore.common.dtype import QuantDtype
from mindformers.modules import Linear

from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQConfig, PTQMode
from mindspore_gs.ptq.ptq_config import InnerPTQConfig
from mindspore_gs.ptq.smooth_quant.smooth_quant import SmoothQuant
from mindspore_gs.ptq.quant_cells import SQLinearWrapper
from mindspore_gs.ptq.convert_utils import QuantCell, DequantBMMCell
from mindspore_gs.ptq.fake_quantizer import MinMaxPerLayer, MinMaxPerChannel

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
# pylint: disable=wrong-import-position
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
                             bias_init="ones",
                             weight_init="ones")
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
    assert weight_fake_quant.quant_dtype() == QuantDtype.INT8
    assert weight_fake_quant.is_per_channel()
    assert not weight_fake_quant.narrow_range()
    assert weight_fake_quant.num_bits() == 8
    act_fake_quant = quant_cell.input_quantizer()
    assert isinstance(act_fake_quant, MinMaxPerLayer)
    assert isinstance(act_fake_quant.symmetric(), bool) and not act_fake_quant.symmetric()
    assert act_fake_quant.quant_dtype() == QuantDtype.INT8
    assert not act_fake_quant.is_per_channel()
    assert not act_fake_quant.narrow_range()
    assert act_fake_quant.num_bits() == 8
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


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", ["Ascend"])
@pytest.mark.parametrize("mode", [GRAPH_MODE])
def test_sq_predict_2stage(device, mode):
    """
    Feature: test smooth quant adjust parameter in two stages
    in parallel mode using 2 cards
    Description: Feed invalid type of bn_fold to convert function.
    Expectation: adjust error is in certain range.
    """

    act_in, act_out = 8, 8
    ckpt_path = "test_sq_predict_2stage.ckpt"

    def quant(input_):
        context.set_context(device_target=device, mode=mode)
        cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND)
        ptq = SmoothQuant(cfg)

        network = SimpleNet(in_channels=act_in, out_channels=act_out)
        network = ptq.apply(network)

        def _calibrate(net, calibrate_size):
            for _ in range(calibrate_size):
                example = Tensor(np.random.normal(size=(act_in, act_out)), dtype=dtype.float16)
                _ = net(example)

        _calibrate(network, 2)
        network = ptq.convert(network)
        save_checkpoint(network, ckpt_path)

        fp_network = SimpleNet(in_channels=act_in, out_channels=act_out)
        return fp_network(input_)

    def infer(input_):
        context.set_context(device_target=device, mode=mode)
        cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND)
        ptq = SmoothQuant(cfg)
        network = SimpleNet(in_channels=act_in, out_channels=act_out)
        network = ptq.apply(network)
        network = ptq.convert(network)
        load_checkpoint(ckpt_path, network)
        ptq.fix_param_after_load_ckpt(network)
        return network(input_)

    example = Tensor(np.random.normal(size=(act_in, act_out)), dtype=dtype.float16)
    foutput = quant(example)
    qoutput = infer(example)
    print(f"-------------------foutput {foutput}", flush=True)
    print(f"-------------------qoutput {qoutput}", flush=True)
    print(f"rel error: {relative_tolerance_acceptable(qoutput[1].asnumpy(), foutput[1].asnumpy(), 10)}")
    print(f"abs error: {absolute_tolerance_acceptable(qoutput[1].asnumpy(), foutput[1].asnumpy(), 10)}")
