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
"""test sim_qat applied on lenet network and mnist dataset."""

import os
from collections import OrderedDict
import pytest
from mindspore_gs.quantization.simulated_quantization import SimulatedQuantizationAwareTraining as SimQAT
from mindspore_gs.quantization.simulated_quantization.simulated_fake_quantizers import SimulatedFakeQuantizerPerLayer, \
    SimulatedFakeQuantizerPerChannel
from mindspore_gs.quantization.quantize_wrapper_cell import QuantizeWrapperCell
from tests.st import test_utils as utils


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_lenet_apply():
    """
    Feature: Simulated quantization algorithm.
    Description: Apply simulated_quantization on lenet.
    Expectation: Apply success.
    """

    from ....models.official.cv.lenet.src.lenet import LeNet5
    network = LeNet5(10)
    qat = SimQAT({"per_channel": [False, True], "symmetric": [False, True], "quant_delay": [900, 900]})
    new_network = qat.apply(network)
    cells: OrderedDict = new_network.name_cells()
    assert cells.get("Conv2dQuant", None) is not None
    conv_quant: QuantizeWrapperCell = cells.get("Conv2dQuant")
    assert isinstance(conv_quant, QuantizeWrapperCell)
    conv_handler = conv_quant._handler
    weight_fake_quant: SimulatedFakeQuantizerPerChannel = conv_handler.fake_quant_weight
    assert isinstance(weight_fake_quant, SimulatedFakeQuantizerPerChannel)
    assert weight_fake_quant._symmetric
    assert weight_fake_quant._quant_delay == 900
    act_fake_quant = conv_quant._output_quantizer
    assert isinstance(act_fake_quant, SimulatedFakeQuantizerPerLayer)
    assert not act_fake_quant._symmetric
    assert act_fake_quant._quant_delay == 900

    assert cells.get("DenseQuant", None) is not None
    dense_quant: QuantizeWrapperCell = cells.get("DenseQuant")
    assert isinstance(dense_quant, QuantizeWrapperCell)
    dense_handler = dense_quant._handler
    weight_fake_quant: SimulatedFakeQuantizerPerChannel = dense_handler.fake_quant_weight
    assert isinstance(weight_fake_quant, SimulatedFakeQuantizerPerChannel)
    assert weight_fake_quant._symmetric
    assert weight_fake_quant._quant_delay == 900
    act_fake_quant = dense_quant._output_quantizer
    assert isinstance(act_fake_quant, SimulatedFakeQuantizerPerLayer)
    assert not act_fake_quant._symmetric
    assert act_fake_quant._quant_delay == 900


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_accuracy_graph():
    """
    Feature: test accuracy of sim qat work on lenet5.
    Description: Apply sim qat on lenet5 and test accuracy.
    Expectation: accuracy is larger than 0.98.
    """

    cur_path = os.path.dirname(os.path.abspath(__file__))
    model_name = "lenet"
    config_name = "lenet_mnist_config.yaml"
    ori_model_path = os.path.join(cur_path, "../../../../tests/models/official/cv")

    model_path = utils.train_network(ori_model_path, model_name, config_name, "quantization/simqat",
                                     "run_standalone_train_gpu.sh", "GRAPH", "mnist", 500)

    acc = utils.eval_network(model_path, model_name, config_name, "quantization/simqat", "run_eval_gpu.sh",
                             "train/ckpt/checkpoint_lenet-10_1875.ckpt", "GRAPH", "mnist", 200)
    assert acc > 0.98



@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_accuracy_pynative():
    """
    Feature: test accuracy of sim qat work on lenet5.
    Description: Apply sim qat on lenet5 and test accuracy.
    Expectation: accuracy is larger than 0.98.
    """

    cur_path = os.path.dirname(os.path.abspath(__file__))
    model_name = "lenet"
    config_name = "lenet_mnist_config.yaml"
    ori_model_path = os.path.join(cur_path, "../../../../tests/models/official/cv")

    model_path = utils.train_network(ori_model_path, model_name, config_name, "quantization/simqat",
                                     "run_standalone_train_gpu.sh", "PYNATIVE", "mnist", 700)

    acc = utils.eval_network(model_path, model_name, config_name, "quantization/simqat", "run_eval_gpu.sh",
                             "train/ckpt/checkpoint_lenet-10_1875.ckpt", "PYNATIVE", "mnist", 200)
    assert acc > 0.98
