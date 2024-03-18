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
"""test slb qat on lenet network and mnist dataset."""

import os
from collections import OrderedDict
import pytest
import numpy as np
import mindspore
from mindspore import nn, context
from mindspore import Model
from mindspore.nn.metrics import Accuracy
from mindspore.common.dtype import QuantDtype
from mindspore_gs.quantization.slb import SlbQuantAwareTraining as SlbQAT
from mindspore_gs.quantization.slb.slb_fake_quantizer import SlbFakeQuantizerPerLayer
from mindspore_gs.quantization.slb.slb_quant import Conv2dSlbQuant


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_bit", ["W4", "W2", "W1", "W4A8", "W2A8", "W1A8"])
@pytest.mark.parametrize("enable_bn_calibration", [True, False])
def test_lenet_apply(quant_bit, enable_bn_calibration):
    """
    Feature: slb quantization algorithm.
    Description: Apply slb qat on lenet.
    Expectation: Apply success.
    """

    from ....models.research.cv.lenet.src.lenet import LeNet5
    network = LeNet5(10)
    if quant_bit == "W4":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT4], "enable_act_quant": False,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})
    elif quant_bit == "W2":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT2], "enable_act_quant": False,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})
    elif quant_bit == "W1":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT1], "enable_act_quant": False,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})
    elif quant_bit == "W4A8":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT4], "enable_act_quant": True,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})
    elif quant_bit == "W2A8":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT2], "enable_act_quant": True,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})
    elif quant_bit == "W1A8":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT1], "enable_act_quant": True,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})

    new_network = qat.apply(network)
    cells: OrderedDict = new_network.name_cells()
    conv_quant = cells.get("Conv2d", None)
    assert isinstance(conv_quant, Conv2dSlbQuant)
    weight_fake_quant: SlbFakeQuantizerPerLayer = conv_quant.weight_quantizer()
    assert isinstance(weight_fake_quant, SlbFakeQuantizerPerLayer)
    assert qat._config.enable_bn_calibration == enable_bn_calibration
    assert qat._config.epoch_size == 10
    assert qat._config.has_trained_epoch == 0
    assert qat._config.t_start_val == 1.0
    assert qat._config.t_start_time == 0.2
    assert qat._config.t_end_time == 0.6
    assert qat._config.t_factor == 3.2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("enable_act_quant", [True, False])
@pytest.mark.parametrize("run_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_lenet_convert(run_mode, enable_act_quant):
    """
    Feature: SLB convert function.
    Description: convert a compressed network to a standard network before exporting to MindIR.
    Expectation: convert success and structure of network as expect.
    """

    from ....models.research.cv.lenet.src.lenet import LeNet5
    context.set_context(mode=run_mode)
    network = LeNet5(10)
    config = {"quant_dtype": [QuantDtype.INT8, QuantDtype.INT1], "enable_act_quant": enable_act_quant,
              "enable_bn_calibration": False, "epoch_size": 10,
              "has_trained_epoch": 0, "t_start_val": 1.0,
              "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2}
    qat = SlbQAT(config)
    new_network = qat.apply(network)
    new_network = qat.convert(new_network)
    data_in = mindspore.Tensor(np.ones([1, 1, 32, 32]), mindspore.float32)
    file_name = "./lenet_convert_{}_{}.mindir".format(run_mode, enable_act_quant)
    mindspore.export(new_network, data_in, file_name=file_name, file_format="MINDIR")
    graph = mindspore.load(file_name)
    mindspore.nn.GraphCell(graph)


@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_bit", ["W1", "W4A8", "W1A8"])
@pytest.mark.parametrize("enable_bn_calibration", [True])
def lenet_accuracy_bnon(quant_bit, enable_bn_calibration):
    """
    Feature: test accuracy of slb qat work on lenet5.
    Description: Apply slb qat on lenet5 and test accuracy.
    Expectation: accuracy is larger than 0.95.
    """

    from ....models.research.cv.lenet.src.lenet import LeNet5
    from ....models.research.cv.lenet.src.dataset import create_dataset as create_mnist_ds
    mnist_path = os.getenv("DATASET_PATH", "/home/workspace/mindspore_dataset/")
    data_path = os.path.join(mnist_path, "mnist/train")
    ds_train = create_mnist_ds(data_path, 32, 1)
    network = LeNet5(10)

    # convert network to quantization aware network
    if quant_bit == "W4":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT4], "enable_act_quant": False,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})
    elif quant_bit == "W2":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT2], "enable_act_quant": False,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})
    elif quant_bit == "W1":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT1], "enable_act_quant": False,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})
    elif quant_bit == "W4A8":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT4], "enable_act_quant": True,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})
    elif quant_bit == "W2A8":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT2], "enable_act_quant": True,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})
    elif quant_bit == "W1A8":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT1], "enable_act_quant": True,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})
    new_network = qat.apply(network)

    # define network loss
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    # define network optimization
    net_opt = nn.Momentum(new_network.trainable_params(), 0.01, 0.9)

    # define model
    model = Model(new_network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Training ==============")
    model.train(10, ds_train, callbacks=qat.callbacks(model, ds_train))
    print("============== End Training ==============")

    ds_eval = create_mnist_ds(os.path.join(mnist_path, "mnist/test"), 32, 1)

    print("============== Starting Testing ==============")
    acc = model.eval(ds_eval)
    print("============== {} ==============".format(acc))
    assert acc['Accuracy'] > 0.95


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_bit", ["W1"])
@pytest.mark.parametrize("enable_bn_calibration", [True])
def test_lenet_accuracy_bnon_graph_woq(quant_bit, enable_bn_calibration):
    """
    Feature: test accuracy of slb qat work on lenet5 Graph mode.
    Description: Apply slb qat on lenet5 and test accuracy.
    Expectation: accuracy is larger than 0.95.
    """
    context.set_context(mode=context.GRAPH_MODE)
    lenet_accuracy_bnon(quant_bit, enable_bn_calibration)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_bit", ["W4A8", "W1A8"])
@pytest.mark.parametrize("enable_bn_calibration", [True])
def test_lenet_accuracy_bnon_graph(quant_bit, enable_bn_calibration):
    """
    Feature: test accuracy of slb qat work on lenet5 Graph mode.
    Description: Apply slb qat on lenet5 and test accuracy.
    Expectation: accuracy is larger than 0.95.
    """
    context.set_context(mode=context.GRAPH_MODE)
    lenet_accuracy_bnon(quant_bit, enable_bn_calibration)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_bit", ["W4"])
@pytest.mark.parametrize("enable_bn_calibration", [True])
def test_lenet_accuracy_bnon_pynative(quant_bit, enable_bn_calibration):
    """
    Feature: test accuracy of slb qat work on lenet5 Pynative mode.
    Description: Apply slb qat on lenet5 and test accuracy.
    Expectation: accuracy is larger than 0.95.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    lenet_accuracy_bnon(quant_bit, enable_bn_calibration)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_bit", ["W1A8"])
@pytest.mark.parametrize("enable_bn_calibration", [False])
@pytest.mark.parametrize("run_mode", [context.GRAPH_MODE])
def test_lenet_accuracy_bnoff_w1a8(quant_bit, enable_bn_calibration, run_mode):
    """
    Feature: test accuracy of slb qat work on lenet5.
    Description: Apply slb qat on lenet5 and test accuracy.
    Expectation: accuracy is larger than 0.95.
    """

    from ....models.research.cv.lenet.src.lenet import LeNet5
    from ....models.research.cv.lenet.src.dataset import create_dataset as create_mnist_ds
    context.set_context(mode=run_mode)
    mnist_path = os.getenv("DATASET_PATH", "/home/workspace/mindspore_dataset/")
    data_path = os.path.join(mnist_path, "mnist/train")
    ds_train = create_mnist_ds(data_path, 64, 1)
    network = LeNet5(10)

    # convert network to quantization aware network
    if quant_bit == "W4":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT4], "enable_act_quant": False,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})
    elif quant_bit == "W2":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT2], "enable_act_quant": False,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})
    elif quant_bit == "W1":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT1], "enable_act_quant": False,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})
    elif quant_bit == "W4A8":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT4], "enable_act_quant": True,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})
    elif quant_bit == "W2A8":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT2], "enable_act_quant": True,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})
    elif quant_bit == "W1A8":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT1], "enable_act_quant": True,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})
    new_network = qat.apply(network)

    # define network loss
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    # define network optimization
    net_opt = nn.Momentum(new_network.trainable_params(), 0.01, 0.9)

    # define model
    model = Model(new_network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Training ==============")
    model.train(10, ds_train, callbacks=qat.callbacks(model, ds_train))
    print("============== End Training ==============")

    ds_eval = create_mnist_ds(os.path.join(mnist_path, "mnist/test"), 32, 1)

    print("============== Starting Testing ==============")
    acc = model.eval(ds_eval)
    print("============== {} ==============".format(acc))
    assert acc['Accuracy'] > 0.95

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_bit", ["W4", "W1"])
@pytest.mark.parametrize("enable_bn_calibration", [False])
@pytest.mark.parametrize("run_mode", [context.GRAPH_MODE])
def test_lenet_accuracy_bnoff(quant_bit, enable_bn_calibration, run_mode):
    """
    Feature: test accuracy of slb qat work on lenet5.
    Description: Apply slb qat on lenet5 and test accuracy.
    Expectation: accuracy is larger than 0.95.
    """

    from ....models.research.cv.lenet.src.lenet import LeNet5
    from ....models.research.cv.lenet.src.dataset import create_dataset as create_mnist_ds
    context.set_context(mode=run_mode)
    mnist_path = os.getenv("DATASET_PATH", "/home/workspace/mindspore_dataset/")
    data_path = os.path.join(mnist_path, "mnist/train")
    ds_train = create_mnist_ds(data_path, 32, 1)
    network = LeNet5(10)

    # convert network to quantization aware network
    if quant_bit == "W4":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT4], "enable_act_quant": False,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})
    elif quant_bit == "W2":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT2], "enable_act_quant": False,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})
    elif quant_bit == "W1":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT1], "enable_act_quant": False,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})
    elif quant_bit == "W4A8":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT4], "enable_act_quant": True,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})
    elif quant_bit == "W2A8":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT2], "enable_act_quant": True,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})
    elif quant_bit == "W1A8":
        qat = SlbQAT({"quant_dtype": [QuantDtype.INT8, QuantDtype.INT1], "enable_act_quant": True,
                      "enable_bn_calibration": enable_bn_calibration, "epoch_size": 10,
                      "has_trained_epoch": 0, "t_start_val": 1.0,
                      "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2})
    new_network = qat.apply(network)

    # define network loss
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    # define network optimization
    net_opt = nn.Momentum(new_network.trainable_params(), 0.01, 0.9)

    # define model
    model = Model(new_network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Training ==============")
    model.train(10, ds_train, callbacks=qat.callbacks(model, ds_train))
    print("============== End Training ==============")

    ds_eval = create_mnist_ds(os.path.join(mnist_path, "mnist/test"), 32, 1)

    print("============== Starting Testing ==============")
    acc = model.eval(ds_eval)
    print("============== {} ==============".format(acc))
    assert acc['Accuracy'] > 0.95
