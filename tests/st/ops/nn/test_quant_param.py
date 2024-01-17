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

import pytest
import numpy as np

import mindspore
from mindspore import QuantDtype, Tensor, dtype, context, GRAPH_MODE
from mindspore.ops.operations import FakeQuantParam
from mindspore_gs.quantization.fake_quantizer import FakeQuantParamCell
from mindspore_gs.ptq.convert_utils import AntiQuantCell, QuantCell


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
@pytest.mark.parametrize("mode", [GRAPH_MODE])
def test_quant_cell(mode):
    """
    Feature: QuantCell.
    Description: test QuantCell can serialize and deserialize successfully.
    Expectation: Success.
    """

    context.set_context(device_target="Ascend", mode=mode)
    scale = 2.0
    zp = 1.0
    origin = np.ones((3, 4), dtype=np.float32)
    expect = origin * scale + zp
    expect = expect.astype(np.int8)
    x = Tensor(origin, dtype=dtype.float32)
    qcell = QuantCell(scale, zp)
    output = qcell(x)
    assert (expect == output.asnumpy()).all()
    mindspore.save_checkpoint(qcell, "ascend-quant-cell.ckpt")

    qcell2 = QuantCell(1.0, 0.0)
    mindspore.load_checkpoint("ascend-quant-cell.ckpt", qcell2)
    qcell2.update_ascend_quant()
    output2 = qcell2(x)
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
    x = Tensor(origin, dtype=dtype.int8)
    aqcell = AntiQuantCell(scale, zp)
    output = aqcell(x)
    assert (output.asnumpy() == expect).all()
    mindspore.save_checkpoint(aqcell, "ascend-antiquant-cell.ckpt")

    aqcell2 = AntiQuantCell([1.0, 1.0], [0.0, 0.0])
    mindspore.load_checkpoint("ascend-antiquant-cell.ckpt", aqcell2)
    output2 = aqcell2(x)
    assert (output2.asnumpy() == expect).all()
