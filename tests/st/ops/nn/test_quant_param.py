# Copyright 2023 Huawei Technologies Co., Ltd
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
"""test quant param cells can serialize and deserialize successfully."""
import os
import sys

import pytest
import numpy as np

import mindspore
from mindspore import QuantDtype, Tensor, dtype, context, GRAPH_MODE, nn, Parameter, PYNATIVE_MODE
from mindspore.ops.operations import FakeQuantParam, BatchMatMul, MatMul
from mindspore_gs.quantization.fake_quantizer import FakeQuantParamCell
from mindspore_gs.ptq.convert_utils import AntiQuantCell, QuantCell

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../../'))
# pylint: disable=wrong-import-position
from tests.st.test_utils import relative_tolerance_acceptable


def test_fake_quant_cell():
    """
    Feature: FakeQuantParamCell.
    Description: test FakeQuantParamCell can serialize and deserialize successfully.
    Expectation: Success.
    """

    scale = [1.0, 2.0, 3.0]
    zp = [-1, 0, 1]
    extras = {"narrow_range": False}
    fq = FakeQuantParam.linear_quant_param(QuantDtype.INT4, scale, zp, True, **extras)
    fqcell = FakeQuantParamCell(fq)
    mindspore.save_checkpoint(fqcell, "fqcell.ckpt")

    fq = FakeQuantParam.linear_quant_param(QuantDtype.INT8, 2.0, 2, False)
    fqcell = FakeQuantParamCell(fq)
    mindspore.load_checkpoint("fqcell.ckpt", fqcell)
    assert fq.attrs["is_per_channel"]
    assert fq.attrs["quant_dtype"] == QuantDtype.INT4
    assert fq.attrs[FakeQuantParam.attr_key_linear_quant_scale] == scale
    assert fq.attrs[FakeQuantParam.attr_key_linear_quant_zero_point] == zp
    assert not fq.attrs["narrow_range"]


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [GRAPH_MODE, PYNATIVE_MODE])
def test_quant_cell_pertensor(mode):
    """
    Feature: QuantCell.
    Description: test QuantCell can serialize and deserialize successfully.
    Expectation: Success.
    """

    os.environ['GRAPH_OP_RUN'] = "1"
    context.set_context(device_target="Ascend", mode=mode)
    scale = 2.0
    zp = 1.0
    origin = np.ones((3, 4), dtype=np.float32)
    expect = np.round(origin / scale + zp)
    expect = expect.astype(np.int8)
    x = Tensor(origin, dtype=dtype.float32)
    t_scale = Tensor([scale], dtype=dtype.float32)
    t_zp = Tensor([zp], dtype=dtype.float32)
    qcell = QuantCell(t_scale, t_zp)
    output = qcell(x)
    assert (expect == output.asnumpy()).all()
    mindspore.save_checkpoint(qcell, "test_quant_cell_pertensor.ckpt")

    qcell2 = QuantCell(Tensor([1.0], dtype=dtype.float32), Tensor([0.0], dtype=dtype.float32))
    mindspore.load_checkpoint("test_quant_cell_pertensor.ckpt", qcell2)
    output2 = qcell2(x)
    os.environ.pop('GRAPH_OP_RUN')

    assert (expect == output2.asnumpy()).all()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [GRAPH_MODE, PYNATIVE_MODE])
def test_quant_cell_perchannel(mode):
    """
    Feature: QuantCell.
    Description: test QuantCell can serialize and deserialize successfully.
    Expectation: Success.
    """

    os.environ['GRAPH_OP_RUN'] = "1"
    context.set_context(device_target="Ascend", mode=mode)
    scale = [2.0, 3.0]
    zp = [1.0, 2.0]
    origin = np.ones((3, 2), dtype=np.float32)
    expect = np.round(origin / scale + zp)
    expect = expect.astype(np.int8)
    x = Tensor(origin, dtype=dtype.float32)
    t_scale = Tensor(scale, dtype=dtype.float32)
    t_zp = Tensor(zp, dtype=dtype.float32)
    qcell = QuantCell(t_scale, t_zp)
    output = qcell(x)
    assert (expect == output.asnumpy()).all()
    mindspore.save_checkpoint(qcell, "test_quant_cell_perchannel.ckpt")

    qcell2 = QuantCell(Tensor([1.0, 1.0], dtype=dtype.float32), Tensor([0.0, 0.0], dtype=dtype.float32))
    mindspore.load_checkpoint("test_quant_cell_perchannel.ckpt", qcell2)
    output2 = qcell2(x)
    os.environ.pop('GRAPH_OP_RUN')
    assert (expect == output2.asnumpy()).all()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [GRAPH_MODE])
def test_antiquant_cell(mode):
    """
    Feature: AntiQuantCell.
    Description: test AntiQuantCell can serialize and deserialize successfully.
    Expectation: Success.
    """

    context.set_context(device_target="Ascend", mode=mode)
    scale = [2.0, 3.0]
    zp = [-1.0, 1.0]
    origin = np.ones((4, 2), dtype=np.int8)
    expect = origin.astype(np.float32)
    expect = (expect - zp) * scale

    def check_1stage():
        x = Tensor(origin, dtype=dtype.int8)
        aqcell = AntiQuantCell(scale, zp, dtype.float16)
        output = aqcell(x)
        assert (output.asnumpy() == expect).all()
        mindspore.save_checkpoint(aqcell, "ascend-antiquant-cell.ckpt")

    def check_2stage():
        x = Tensor(origin, dtype=dtype.int8)
        aqcell2 = AntiQuantCell([1.0, 1.0], [0.0, 0.0], dtype.float16)
        mindspore.load_checkpoint("ascend-antiquant-cell.ckpt", aqcell2)
        output2 = aqcell2(x)
        assert (output2.asnumpy() == expect).all()

    check_1stage()
    check_2stage()


class AntiQuantBMMNet(nn.Cell):
    """BatchMatMul network with AntiQuantCell."""
    def __init__(self, scale, zp, ic=5, oc=6):
        super().__init__()
        self.antiquant = AntiQuantCell(scale, zp, dtype.float16)
        self.matmul = BatchMatMul(transpose_a=False, transpose_b=False)
        self.weight = Parameter(Tensor(np.ones((ic, oc), np.int8), dtype.int8))

    def construct(self, x):
        weight = self.antiquant(self.weight)
        y = self.matmul(x, weight)
        return y


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [GRAPH_MODE])
def test_antiquantbmm_cell(mode):
    """
    Feature: AntiQuantCell + BatchMatmul.
    Description: test AntiQuantCell + BatchMatmul can serialize, deserialize and predict successfully.
    Expectation: Success.
    """

    context.set_context(device_target="Ascend", mode=mode)
    bs = 2
    ic = 5
    oc = 2
    scale = [2.0, 3.0]
    zp = [-1.0, 1.0]
    expect = np.array([[[20., 0.], [20., 0.], [20., 0.], [20., 0.], [20., 0.]],
                       [[20., 0.], [20., 0.], [20., 0.], [20., 0.], [20., 0.]]])

    def check_1stage():
        net = AntiQuantBMMNet(scale, zp, ic, oc)
        x = Tensor(np.ones((bs, ic, ic)), dtype=dtype.float16)
        output = net(x)
        assert relative_tolerance_acceptable(output.asnumpy(), expect, 1e-5)
        mindspore.save_checkpoint(net, "test_antiquantbmm_cell.ckpt")

    def check_2stage():
        net = AntiQuantBMMNet([1.0, 1.0], [0.0, 0.0], ic, oc)
        mindspore.load_checkpoint("test_antiquantbmm_cell.ckpt", net)
        x = Tensor(np.ones((bs, ic, ic)), dtype=dtype.float16)
        output2 = net(x)
        assert relative_tolerance_acceptable(output2.asnumpy(), expect, 1e-5)

    check_1stage()
    check_2stage()


class AntiQuantMMNet(nn.Cell):
    """MatMul network with AntiQuantCell."""
    def __init__(self, scale, zp, ic=5, oc=6):
        super().__init__()
        self.antiquant = AntiQuantCell(scale, zp, dtype.float16)
        self.matmul = MatMul(transpose_a=False, transpose_b=False)
        self.weight = Parameter(Tensor(np.ones((ic, oc), np.int8), dtype.int8))

    def construct(self, x):
        weight = self.antiquant(self.weight)
        y = self.matmul(x, weight)
        return y


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [GRAPH_MODE])
def test_antiquantmm_cell(mode):
    """
    Feature: AntiQuantCell + Matmul.
    Description: test AntiQuantCell + Matmul can serialize, deserialize and predict successfully.
    Expectation: Success.
    """

    context.set_context(device_target="Ascend", mode=mode)
    ic = 5
    oc = 2
    scale = [2.0, 3.0]
    zp = [-1.0, 1.0]
    expect = np.array([[[20., 0.], [20., 0.], [20., 0.], [20., 0.], [20., 0.]]])

    def check_1stage():
        net = AntiQuantMMNet(scale, zp, ic, oc)
        x = Tensor(np.ones((ic, ic)), dtype=dtype.float16)
        output = net(x)
        assert relative_tolerance_acceptable(output.asnumpy(), expect, 1e-5)
        mindspore.save_checkpoint(net, "test_antiquantmm_cell.ckpt")

    def check_2stage():
        net = AntiQuantMMNet([1.0, 1.0], [0.0, 0.0], ic, oc)
        mindspore.load_checkpoint("test_antiquantmm_cell.ckpt", net)
        x = Tensor(np.ones((ic, ic)), dtype=dtype.float16)
        output2 = net(x)
        assert relative_tolerance_acceptable(output2.asnumpy(), expect, 1e-5)

    check_1stage()
    check_2stage()
