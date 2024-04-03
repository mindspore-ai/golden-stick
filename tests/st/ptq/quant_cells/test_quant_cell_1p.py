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
"""test quant cells."""

import os
import pytest
import numpy as np

from mindspore import Parameter, context, GRAPH_MODE, Tensor, PYNATIVE_MODE
from mindspore import dtype as mstype
from mindspore.nn import Cell
from mindspore.ops.operations._inner_ops import Quant
from mindspore.ops.auto_generate import QuantBatchMatmul

from mindspore_gs.ptq.convert_utils import AntiquantBMMCell, QuantCell, DequantBMMCell
from mindspore_gs.common.numpy_quant_common import NumpyQuantOps, NumpyFullQuant
from tests.st.test_utils import relative_tolerance_acceptable


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [GRAPH_MODE, PYNATIVE_MODE])
def test_weight_quant_bmm_cell_as_antiquant_1p(mode):
    """
    Feature: weight quant bmm cell for antiquant
    Description: test antiquant using weight quant bmm cell
    Expectation: accuracy in tolerance
    """

    os.environ['GRAPH_OP_RUN'] = "1"
    context.set_context(device_target="Ascend", mode=mode)
    weight = np.array([[100, 200], [10, 25]]).astype(np.int8)
    activation = np.array([[0.1, 1.], [0.5, 2.4]]).astype(np.float16)
    scale = np.array([0.5, 0.27]).astype(np.float16)
    offset = np.array([-127, -10]).astype(np.float16)
    expect = np.matmul(activation, NumpyQuantOps.anti_quant(weight, scale, offset))
    wqmm_cell = AntiquantBMMCell(scale, offset)
    t_activation = Tensor(activation, dtype=mstype.float16)
    p_weight = Parameter(Tensor(weight, dtype=mstype.int8), 'weight')
    fact = wqmm_cell(t_activation, p_weight).asnumpy()
    os.environ.pop('GRAPH_OP_RUN')

    assert relative_tolerance_acceptable(fact, expect, 3e-2)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [GRAPH_MODE, PYNATIVE_MODE])
def test_quant_cell_1p(mode):
    """
    Feature: quant tensor from fp16 to int8
    Description: test quant ops
    Expectation: accuracy in tolerance
    """

    os.environ['GRAPH_OP_RUN'] = "1"
    context.set_context(device_target="Ascend", mode=mode)
    activation = np.array([[0.1, 1.], [0.5, 2.4]]).astype(np.float16)
    scale = np.array([0.5]).astype(np.float16)
    offset = np.array([-10]).astype(np.float16)
    expect = NumpyQuantOps.quant(activation, scale, offset)

    quant_cell = QuantCell(Tensor(scale, dtype=mstype.float16),
                           Tensor(offset, dtype=mstype.float16))
    t_activation = Tensor(activation, dtype=mstype.float16)
    fact = quant_cell(t_activation).asnumpy()
    os.environ.pop('GRAPH_OP_RUN')

    assert relative_tolerance_acceptable(fact, expect, 3e-2)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [GRAPH_MODE, PYNATIVE_MODE])
def test_dequant_bmm_cell_1p(mode):
    """
    Feature: a fused kernel which combine matmul and dequant ops
    Description: test deqaunt batch matmul
    Expectation: accuracy in tolerance
    """

    os.environ['GRAPH_OP_RUN'] = "1"
    context.set_context(device_target="Ascend", mode=mode)
    weight = np.array([[100, 120], [10, 25]]).astype(np.int32)
    activation = np.array([[3, 1], [2, 5]]).astype(np.int32)
    weight_scale = np.array([0.5, 0.27]).astype(np.float16)
    activation_scale = np.array([0.3], dtype=np.float16)
    dequant_scale = weight_scale * activation_scale
    bias = np.zeros([2]).astype(np.int32)
    expect = NumpyQuantOps.dequant(np.matmul(activation, weight) + bias, dequant_scale)

    dequant_bmm_cell = DequantBMMCell(dequant_scale)
    t_activation = Tensor(activation, dtype=mstype.int8)
    p_weight = Parameter(Tensor(weight, dtype=mstype.int8), 'weight')
    t_bias = Tensor(bias, dtype=mstype.int32)
    fact = dequant_bmm_cell(t_activation, p_weight, t_bias).asnumpy()
    os.environ.pop('GRAPH_OP_RUN')

    assert relative_tolerance_acceptable(fact, expect, 3e-2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_numpy_full_quant_with_bias_correction():
    """
    Feature: numpy implemented full quant process
    Description: test numpy full quant with bias correction
    Expectation: accuracy in tolerance
    """
    weight = np.array([[2., 4.], [1., 3.]]).astype(np.float16)
    activation = np.array([[1, 10.], [12, 14]]).astype(np.float16)
    weight_scale = np.array([0.5, 0.7]).astype(np.float16)
    act_offset = np.array([10]).astype(np.float16)
    activation_scale = np.array([0.5], dtype=np.float16)
    bias = np.ones([2]).astype(np.float16)

    net = NumpyFullQuant(weight_scale, activation_scale, act_offset)
    orin_out = net.orin_process(activation, weight, bias)
    quant_out = net.process(activation, weight, bias)
    assert relative_tolerance_acceptable(orin_out, quant_out, 7e-2)


class QuantDequantCell(Cell):
    """matmul and dequant fused cell"""
    def __init__(self,
                 weight,
                 weight_scale,
                 act_scale,
                 act_offset,
                 bias,
                 transpose_b=False,
                 dst_dtype=mstype.float16):
        super().__init__()
        self.dbmm = QuantBatchMatmul(transpose_x1=False,
                                     transpose_x2=transpose_b,
                                     dtype=dst_dtype)
        self.dequant_scale = self._dequant_scale(act_scale, weight_scale)
        self.act_scale = act_scale
        self.act_offset = act_offset
        self.weight_scale = weight_scale

        t_scale = 1.0 / act_scale
        self.quant = Quant(t_scale.tolist()[0], act_offset.tolist()[0])

        self.quant_weight = Tensor(NumpyQuantOps.quant(weight, weight_scale, 0), dtype=mstype.int8)
        self.bias = Tensor(self._fused_bias(bias), dtype=mstype.int32)
        self.dequant_offset = Parameter(Tensor(np.zeros(self.bias.shape), dtype=mstype.float32))

    def _dequant_scale(self, act_scale, weight_scale):
        dequant_scale = weight_scale * act_scale
        scale_ui64 = NumpyQuantOps.trans_fp32_to_u64(dequant_scale)
        return Parameter(Tensor(np.squeeze(scale_ui64), dtype=mstype.uint64))

    def _fused_bias(self,
                    bias):
        bias_int32 = (bias / (self.act_scale * self.weight_scale)).astype(np.int32)
        add_item = - np.sum(self.act_offset.astype(np.int32) * self.quant_weight.asnumpy().astype(np.int32),
                            axis=0).astype(np.int32)
        return bias_int32 + add_item

    def construct(self, x):
        quant_act = self.quant(x)
        # (matmul(quant_act, x2) + bias) * scale + offset
        return self.dbmm(quant_act, self.quant_weight,
                         self.dequant_scale, self.dequant_offset, self.bias)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [GRAPH_MODE, PYNATIVE_MODE])
def test_bias_correction_transpose_b_false(mode):
    """
    Feature: test quant and dequant cell with bias correction
    Description: test quant and dequant procedure correction
    Expectation: accuracy in tolerance
    """
    os.environ['GRAPH_OP_RUN'] = "1"
    context.set_context(device_target="Ascend", mode=mode)
    weight = np.array([[2., 4.], [1., 3.]]).astype(np.float16)
    activation = np.array([[1, 10.], [12, 14]]).astype(np.float16)
    weight_scale = np.array([0.5, 0.7]).astype(np.float16)
    act_offset = np.array([10]).astype(np.float16)
    activation_scale = np.array([0.5], dtype=np.float16)
    bias = np.ones([2]).astype(np.float16)

    net = NumpyFullQuant(weight_scale, activation_scale, act_offset)
    quant_out = net.process(activation, weight, bias)
    cell = QuantDequantCell(weight,
                            weight_scale,
                            activation_scale,
                            act_offset,
                            bias)
    t_activation = Tensor(activation, dtype=mstype.float16)
    ms_quant_out = cell(t_activation).asnumpy()
    os.environ.pop('GRAPH_OP_RUN')

    assert relative_tolerance_acceptable(quant_out, ms_quant_out, 7e-2)
