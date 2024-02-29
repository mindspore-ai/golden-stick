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

import pytest
import numpy as np

from mindspore import Parameter, context, GRAPH_MODE, Tensor
from mindspore import dtype as mstype

from mindspore_gs.ptq.convert_utils import FusionAntiquantCell
from tests.st.test_utils import relative_tolerance_acceptable
from .quant_common import NumpyQuantOps


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_weight_quant_bmm_cell_as_antiquant_1p():
    """
    Feature: weight quant bmm cell for antiquant
    Description: test antiquant using weight quant bmm cell
    Expectation: accuracy in tolerance
    """

    context.set_context(device_target="Ascend", mode=GRAPH_MODE)
    weight = np.array([[100, 200], [10, 25]]).astype(np.int8)
    activation = np.array([[0.1, 1.], [0.5, 2.4]]).astype(np.float16)
    scale = np.array([0.5, 0.27]).astype(np.float16)
    offset = np.array([-127, -10]).astype(np.float16)
    expect = np.matmul(activation, NumpyQuantOps.anti_quant(weight, scale, offset))
    wqmm_cell = FusionAntiquantCell(scale, offset)
    t_activation = Tensor(activation, dtype=mstype.float16)
    p_weight = Parameter(Tensor(weight, dtype=mstype.int8), 'weight')
    fact = wqmm_cell(t_activation, p_weight).asnumpy()

    assert relative_tolerance_acceptable(fact, expect, 3e-2)
