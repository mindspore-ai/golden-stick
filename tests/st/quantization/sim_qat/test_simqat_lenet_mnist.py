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
import numpy as np
import mindspore
from mindspore_gs.quantization.simulated_quantization import SimulatedQuantizationAwareTraining as SimQAT
from mindspore_gs.quantization.simulated_quantization.simulated_fake_quantizers import SimulatedFakeQuantizerPerLayer, \
    SimulatedFakeQuantizerPerChannel
from mindspore_gs.quantization.ops.nn import Conv2dQuant, DenseQuant
from tests.st import test_utils as utils

cur_path = os.path.dirname(os.path.abspath(__file__))
model_name = "lenet"
config_name = "lenet_mnist_config.yaml"
ori_model_path = os.path.join(cur_path, "../../../../tests/models/research/cv")
train_log_rpath = os.path.join("golden_stick", "scripts", "train", "log")


@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_lenet_apply():
    """
    Feature: simulated quantization algorithm.
    Description: apply simulated_quantization on lenet5.
    Expectation: apply success and structure of network as expect.
    """

    from ....models.research.cv.lenet.src.lenet import LeNet5
    network = LeNet5(10)
    qat = SimQAT({"per_channel": [False, True], "symmetric": [False, True], "quant_delay": [900, 900]})
    new_network = qat.apply(network)
    cells: OrderedDict = new_network.name_cells()
    conv_quant = cells.get("Conv2d", None)
    assert isinstance(conv_quant, Conv2dQuant)
    weight_fake_quant: SimulatedFakeQuantizerPerChannel = conv_quant.weight_quantizer()
    assert isinstance(weight_fake_quant, SimulatedFakeQuantizerPerChannel)
    assert weight_fake_quant.symmetric()
    assert weight_fake_quant._quant_delay == 900
    act_fake_quant = conv_quant.output_quantizer()
    assert isinstance(act_fake_quant, SimulatedFakeQuantizerPerLayer)
    assert not act_fake_quant.symmetric()
    assert act_fake_quant._quant_delay == 900

    dense_quant = cells.get("Dense", None)
    assert isinstance(dense_quant, DenseQuant)
    weight_fake_quant: SimulatedFakeQuantizerPerChannel = dense_quant.weight_quantizer()
    assert isinstance(weight_fake_quant, SimulatedFakeQuantizerPerChannel)
    assert weight_fake_quant.symmetric()
    assert weight_fake_quant._quant_delay == 900
    act_fake_quant = dense_quant.output_quantizer()
    assert isinstance(act_fake_quant, SimulatedFakeQuantizerPerLayer)
    assert not act_fake_quant.symmetric()
    assert act_fake_quant._quant_delay == 900


@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("enable_fusion", [True, False])
def test_lenet_convert(enable_fusion):
    """
    Feature: simulated quantization convert function.
    Description: convert a compressed network to a standard network, export to MindIR.
    Expectation: convert success and export MindIR to specified path.
    """

    from ....models.research.cv.lenet.src.lenet import LeNet5
    network = LeNet5(10)
    qat = SimQAT({"per_channel": [False, True], "symmetric": [False, True], "quant_delay": [900, 900],
                  "enable_fusion": enable_fusion})
    new_network = qat.apply(network)
    new_network = qat.convert(new_network)
    data_in = mindspore.Tensor(np.ones([1, 1, 32, 32]), mindspore.float32)
    file_name = "./lenet.mindir"
    mindspore.export(new_network, data_in, file_name=file_name, file_format="MINDIR")
    graph = mindspore.load(file_name)
    mindspore.nn.GraphCell(graph)


@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_accuracy_graph():
    """
    Feature: test accuracy of sim_qat on lenet5, mnist in Graph mode.
    Description: apply sim_qat on lenet5 and test accuracy in Graph mod.
    Expectation: accuracy is larger than 0.98.
    """

    model_path = utils.train_network(ori_model_path, model_name, "test_gpu_accuracy_graph", config_name,
                                     "quantization/simqat", "run_standalone_train_gpu.sh",
                                     utils.TrainEvalConfig.run_mode_train_eval_config("GRAPH"), "mnist", 700,
                                     train_log_rpath=train_log_rpath)
    acc = utils.eval_network(model_path, model_name, config_name, "quantization/simqat", "run_eval_gpu.sh",
                             "train/ckpt/checkpoint_lenet-10_1875.ckpt",
                             utils.TrainEvalConfig.run_mode_train_eval_config("GRAPH"), "mnist", 200,
                             train_log_rpath=train_log_rpath)
    assert acc > 0.98


def test_gpu_accuracy_pynative():
    """
    Feature: test accuracy of sim_qat on lenet5, mnist in PyNative mode.
    Description: apply sim_qat on lenet5 and test accuracy in PyNative mod.
    Expectation: accuracy is larger than 0.98.
    """

    model_path = utils.train_network(ori_model_path, model_name, "test_gpu_accuracy_pynative", config_name,
                                     "quantization/simqat", "run_standalone_train_gpu.sh",
                                     utils.TrainEvalConfig.run_mode_train_eval_config("PYNATIVE"), "mnist", 700,
                                     train_log_rpath=train_log_rpath)

    acc = utils.eval_network(model_path, model_name, config_name, "quantization/simqat", "run_eval_gpu.sh",
                             "train/ckpt/checkpoint_lenet-10_1875.ckpt",
                             utils.TrainEvalConfig.run_mode_train_eval_config("PYNATIVE"), "mnist", 200,
                             train_log_rpath=train_log_rpath)
    assert acc > 0.98


@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_continue_train():
    """
    Feature: test continue training feature of sim_qat on lenet5 in Graph mode.
    Description: applying sim_qat on lenet5 and continue training with ckpt.
    Expectation: continue training failed.
    """

    config = utils.TrainEvalConfig.run_mode_train_eval_config("GRAPH")
    config.set_config(utils.TrainEvalConfig.epoch_size_key, 1)

    model_path = utils.train_network(ori_model_path, model_name, "test_gpu_continue_train", config_name,
                                     "quantization/simqat", "run_standalone_train_gpu.sh", config, "mnist", 700,
                                     train_log_rpath=train_log_rpath)
    ckpt_file = os.path.join(cur_path, "checkpoint_lenet-1_1875_{}.ckpt".format("test_gpu_continue_train"))
    try:
        utils.copy_file(os.path.join(model_path, "golden_stick", "scripts", "train/ckpt/checkpoint_lenet-1_1875.ckpt"),
                        ckpt_file)
    except ValueError:
        log_path = os.path.join(model_path, train_log_rpath)
        if os.path.exists(log_path):
            os.system("cat {}".format(log_path))
        else:
            os.system("echo {}".format("No train log file exist: " + log_path))
        assert False
    assert os.path.exists(ckpt_file)

    config.set_config(utils.TrainEvalConfig.epoch_size_key, 2)
    model_path = utils.train_network(ori_model_path, model_name, "test_gpu_continue_train", config_name,
                                     "quantization/simqat", "run_standalone_train_gpu.sh", config, "mnist", 700,
                                     continue_train=True, ckpt_path=ckpt_file, train_log_rpath=train_log_rpath)
    # lenet not support continue train
    assert not os.path.exists(os.path.join(model_path, "golden_stick", "scripts", "train", "ckpt",
                                           "checkpoint_lenet-1_1875.ckpt"))


@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_pre_train():
    """
    Feature: test pre-training feature of sim_qat on lenet5 in Graph mode.
    Description: applying sim_qat on lenet5 and pre-training with ckpt.
    Expectation: pre-training success.
    """

    ckpt_file = os.path.join(utils.ckpt_root, "checkpoint_lenet-10_1875.ckpt")
    assert os.path.exists(ckpt_file)

    config = utils.TrainEvalConfig.run_mode_train_eval_config("GRAPH")
    config.set_config(utils.TrainEvalConfig.epoch_size_key, 1)
    model_path = utils.train_network(ori_model_path, model_name, "test_gpu_pre_train", config_name,
                                     "quantization/simqat", "run_standalone_train_gpu.sh", config, "mnist", 700,
                                     pretrained=True, ckpt_path=ckpt_file, train_log_rpath=train_log_rpath)
    if not os.path.exists(os.path.join(model_path, "golden_stick", "scripts", "train", "ckpt",
                                       "checkpoint_lenet-1_1875.ckpt")):
        log_path = os.path.join(model_path, train_log_rpath)
        if os.path.exists(log_path):
            os.system("cat {}".format(log_path))
        else:
            os.system("echo {}".format("No train log file exist: " + log_path))
        assert False
