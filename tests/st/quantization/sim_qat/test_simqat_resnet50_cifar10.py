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
"""test sim_qat applied on resnet50 network and cifar10 dataset."""

import os
import sys
import time
from collections import OrderedDict
import numpy
import pytest
import mindspore
from mindspore import context
from mindspore_gs.quantization.simulated_quantization.simulated_fake_quantizers import SimulatedFakeQuantizerPerLayer, \
    SimulatedFakeQuantizerPerChannel
from mindspore_gs.quantization.ops.nn import Conv2dBnFoldQuant, DenseQuant
from tests.st import test_utils as utils

cur_path = os.path.dirname(os.path.abspath(__file__))
model_name = "ResNet"
config_name = "resnet50_cifar10_config.yaml"
ori_model_path = os.path.join(cur_path, "../../../../tests/models/official/cv")
train_log_rpath = os.path.join("golden_stick", "scripts", "train_parallel", "log")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("run_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_resnet_apply(run_mode):
    """
    Feature: simulated quantization algorithm.
    Description: apply simulated_quantization on resnet50.
    Expectation: apply success and structure of network as expect.
    """

    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../models/official/cv/ResNet/'))
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
    from tests.models.official.cv.ResNet.golden_stick.quantization.simqat.simqat import create_simqat
    from tests.st.models.resnet import resnet50

    mindspore.context.set_context(mode=run_mode, device_target="GPU")

    network = resnet50(10)
    qat = create_simqat()
    new_network = qat.apply(network)

    cells: OrderedDict = new_network.name_cells()
    conv_quant = cells.get("Conv2dBn_1", None)
    assert isinstance(conv_quant, Conv2dBnFoldQuant)
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("bn_fold", [True, False])
@pytest.mark.parametrize("one_conv_fold", [True, False])
def test_resnet_convert_fusion(bn_fold, one_conv_fold):
    """
    Feature: simulated quantization convert function.
    Description: convert a compressed network to a standard network, export to MindIR.
    Expectation: convert success and export MindIR to specified path.
    """
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../models/official/cv/ResNet/'))
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
    from tests.models.official.cv.ResNet.golden_stick.quantization.simqat.simqat import create_simqat
    from tests.st.models.resnet import resnet50

    mindspore.context.set_context(device_target="GPU")
    network = resnet50(10)
    qat = create_simqat()
    qat.set_enable_fusion(enable_fusion=True)
    qat.set_bn_fold(bn_fold=bn_fold)
    qat.set_one_conv_fold(one_conv_fold=one_conv_fold)
    new_network = qat.apply(network)
    new_network = qat.convert(new_network)
    data_in = mindspore.Tensor(numpy.ones([1, 3, 224, 224]), mindspore.float32)
    file_name = "./resnet50.mindir"
    mindspore.export(new_network, data_in, file_name=file_name, file_format="MINDIR")
    time.sleep(5)
    assert os.path.exists(file_name)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("bn_fold", [True, False])
@pytest.mark.parametrize("one_conv_fold", [True, False])
def test_resnet_convert_no_fusion(bn_fold, one_conv_fold):
    """
    Feature: simulated quantization convert function.
    Description: convert a compressed network to a standard network, export to MindIR.
    Expectation: convert success and export MindIR to specified path.
    """
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../models/official/cv/ResNet/'))
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
    from tests.models.official.cv.ResNet.golden_stick.quantization.simqat.simqat import create_simqat
    from tests.st.models.resnet import resnet50

    mindspore.context.set_context(device_target="GPU")
    network = resnet50(10)
    qat = create_simqat()
    qat.set_one_conv_fold(one_conv_fold=one_conv_fold)
    qat.set_enable_fusion(enable_fusion=False)
    qat.set_bn_fold(bn_fold=bn_fold)
    new_network = qat.apply(network)
    new_network = qat.convert(new_network)
    file_name = "./resnet50.mindir"
    data_in = mindspore.Tensor(numpy.ones([1, 3, 224, 224]), mindspore.float32)
    mindspore.export(new_network, data_in, file_name=file_name, file_format="MINDIR")
    time.sleep(10)
    assert os.path.exists(file_name)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_gpu_accuracy_graph():
    """
    Feature: test accuracy of sim_qat on resnet50, mnist in Graph mode.
    Description: apply sim_qat on resnet50 and test accuracy in Graph mod.
    Expectation: accuracy is larger than 0.15 after 3 epochs.
    """

    is_self_check = os.getenv("SELF_CHECK", "False")
    if is_self_check == "True":
        epochs = 180
        acc_thres = 0.91
    else:
        epochs = 2
        acc_thres = 0.08
    config = utils.TrainEvalConfig.run_mode_train_eval_config("GRAPH")
    config.set_config(utils.TrainEvalConfig.epoch_size_key, epochs)

    model_path = utils.train_network(ori_model_path, model_name, "test_gpu_accuracy_graph", config_name,
                                     "quantization/simqat", "run_distribute_train_gpu.sh", config, "cifar10", 700,
                                     train_log_rpath=train_log_rpath)
    acc = utils.eval_network(model_path, model_name, config_name, "quantization/simqat", "run_eval_gpu.sh",
                             "train_parallel/output/checkpoint/ckpt_0/resnet-{}_195.ckpt".format(epochs), config,
                             "cifar10", 200, "log", "'top_1_accuracy': ([0-9.]*)", train_log_rpath=train_log_rpath)
    assert acc > acc_thres


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_gpu_accuracy_pynative():
    """
    Feature: test accuracy of sim_qat on resnet50, mnist in PyNative mode.
    Description: apply sim_qat on resnet50 and test accuracy in PyNative mod.
    Expectation: accuracy is larger than 0.08 after 2 epochs.
    """

    is_self_check = os.getenv("SELF_CHECK", "False")
    if is_self_check == "True":
        epochs = 180
        acc_thres = 0.91
    else:
        epochs = 1
        acc_thres = 0.04

    config = utils.TrainEvalConfig.run_mode_train_eval_config("PYNATIVE")
    config.set_config(utils.TrainEvalConfig.epoch_size_key, epochs)

    model_path = utils.train_network(ori_model_path, model_name, "test_gpu_accuracy_pynative", config_name,
                                     "quantization/simqat", "run_distribute_train_gpu.sh", config, "cifar10", 700,
                                     train_log_rpath=train_log_rpath)
    acc = utils.eval_network(model_path, model_name, config_name, "quantization/simqat", "run_eval_gpu.sh",
                             "train_parallel/output/checkpoint/ckpt_0/resnet-{}_195.ckpt".format(epochs), config,
                             "cifar10", 200, "log", "'top_1_accuracy': ([0-9.]*)", train_log_rpath=train_log_rpath)
    assert acc > acc_thres


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_gpu_continue_train():
    """
    Feature: test continue training feature of sim_qat on resnet50 in Graph mode.
    Description: applying sim_qat on resnet50 and continue training with ckpt.
    Expectation: continue training success.
    """

    config = utils.TrainEvalConfig.run_mode_train_eval_config("GRAPH")
    config.set_config(utils.TrainEvalConfig.epoch_size_key, 1)

    model_path = utils.train_network(ori_model_path, model_name, "test_gpu_continue_train", config_name,
                                     "quantization/simqat", "run_distribute_train_gpu.sh", config, "cifar10", 700,
                                     train_log_rpath=train_log_rpath)
    ckpt_file = os.path.join(cur_path, "resnet-1_195_{}.ckpt".format("test_gpu_continue_train"))
    try:
        utils.copy_file(os.path.join(model_path, "golden_stick", "scripts",
                                     "train_parallel/output/checkpoint/ckpt_0/resnet-1_195.ckpt"), ckpt_file)
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
                                     "quantization/simqat", "run_distribute_train_gpu.sh", config, "cifar10", 700,
                                     continue_train=True, ckpt_path=ckpt_file, train_log_rpath=train_log_rpath)
    # lenet not support continue train
    if not os.path.exists(os.path.join(model_path, "golden_stick", "scripts", "train_parallel", "output",
                                       "checkpoint", "ckpt_0", "resnet-1_195.ckpt")):
        log_path = os.path.join(model_path, train_log_rpath)
        if os.path.exists(log_path):
            os.system("cat {}".format(log_path))
        else:
            os.system("echo {}".format("No train log file exist: " + log_path))
        assert False
