# Copyright 2022 Huawei Technologies Co., Ltd
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
"""test lsq applied on lenet netwok and mnist dataset."""

from collections import OrderedDict
import pytest
from mindspore_gs.quantization.learned_step_size_quantization import LearnedStepSizeQuantizationAwareTraining as \
    LearnedQAT
from mindspore_gs.quantization.learned_step_size_quantization.learned_step_size_quantization_layer_policy import \
    LearnedStepSizeFakeQuantizerPerLayer, LearnedStepSizeFakeQuantizePerChannel
from mindspore_gs.quantization.ops.nn import Conv2dQuant, DenseQuant


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_lenet_apply():
    """
    Feature: LSQ quantization algorithm.
    Description: Apply LSQ quantization on lenet.
    Expectation: Apply success.
    """

    from ....models.research.cv.lenet.src.lenet import LeNet5
    network = LeNet5(10)
    config = {"per_channel": [False, True], "symmetric": [True, True], "quant_delay": [0, 0]}
    lsq = LearnedQAT(config)
    new_network = lsq.apply(network)
    cells: OrderedDict = new_network.name_cells()
    conv_quant = cells.get("Conv2d", None)
    assert isinstance(conv_quant, Conv2dQuant)
    weight_fake_quant: LearnedStepSizeFakeQuantizePerChannel = conv_quant.weight_quantizer()
    assert isinstance(weight_fake_quant, LearnedStepSizeFakeQuantizePerChannel)
    assert weight_fake_quant.symmetric()
    assert weight_fake_quant._quant_delay == 0
    act_fake_quant = conv_quant.output_quantizer()
    assert isinstance(act_fake_quant, LearnedStepSizeFakeQuantizerPerLayer)
    assert act_fake_quant._symmetric
    assert act_fake_quant._quant_delay == 0

    dense_quant = cells.get("Dense", None)
    assert isinstance(dense_quant, DenseQuant)
    weight_fake_quant: LearnedStepSizeFakeQuantizePerChannel = dense_quant.weight_quantizer()
    assert isinstance(weight_fake_quant, LearnedStepSizeFakeQuantizePerChannel)
    assert weight_fake_quant.symmetric()
    assert weight_fake_quant._quant_delay == 0
    act_fake_quant = dense_quant.output_quantizer()
    assert isinstance(act_fake_quant, LearnedStepSizeFakeQuantizerPerLayer)
    assert act_fake_quant.symmetric()
    assert act_fake_quant._quant_delay == 0
