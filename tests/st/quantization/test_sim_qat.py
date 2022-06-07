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
"""test qat."""

import os
import sys
from collections import OrderedDict
import pytest
import mindspore
from mindspore import nn
from mindspore.train.serialization import load_checkpoint
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore_gs.quantization.simulated_quantization import SimulatedQuantizationAwareTraining
from mindspore_gs.quantization.simulated_quantization.simulated_fake_quantizers import SimulatedFakeQuantizerPerLayer, \
    SimulatedFakeQuantizerPerChannel
from mindspore_gs.quantization.quantize_wrapper_cell import QuantizeWrapperCell

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../models/official/cv/'))


class NetToQuant(nn.Cell):
    """
    Network with single conv2d to be quanted
    """

    def __init__(self):
        super(NetToQuant, self).__init__()
        self.conv = nn.Conv2d(5, 6, 5, pad_mode='valid')

    def construct(self, x):
        x = self.conv(x)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_config():
    """
    Feature: DefaultQuantAwareTraining algorithm set functions.
    Description: Apply DefaultQuantAwareTraining on lenet.
    Expectation: Apply success and coordinate attributes are same as config.
    """

    network = NetToQuant()
    qat = SimulatedQuantizationAwareTraining()
    qat.set_act_quant_delay(900)
    qat.set_weight_quant_delay(900)
    qat.set_act_per_channel(False)
    qat.set_weight_per_channel(True)
    qat.set_act_narrow_range(False)
    qat.set_weight_narrow_range(False)
    qat.set_one_conv_fold(True)
    qat.set_bn_fold(False)
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_config_enable_fusion():
    """
    Feature: set_enable_fusion api of DefaultQuantAwareTraining.
    Description: Check default value of enable_fusion and value after called set_enable_fusion.
    Expectation: Config success.
    """
    qat = SimulatedQuantizationAwareTraining()
    assert not qat._config.enable_fusion
    qat.set_enable_fusion(True)
    assert qat._config.enable_fusion


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_config_one_conv_fold():
    """
    Feature: set_one_conv_fold api of DefaultQuantAwareTraining.
    Description: Check default value of one_conv_fold and value after called set_one_conv_fold.
    Expectation: Config success.
    """
    qat = SimulatedQuantizationAwareTraining()
    assert qat._config.one_conv_fold
    qat.set_one_conv_fold(False)
    assert not qat._config.one_conv_fold


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_lenet():
    """
    Feature: Simulated quantization algorithm.
    Description: Apply simulated_quantization on lenet.
    Expectation: Apply success.
    """

    from lenet.src.lenet import LeNet5
    network = LeNet5(10)
    qat = SimulatedQuantizationAwareTraining({"per_channel": [False, True], "symmetric": [False, True],
                                              "quant_delay": [900, 900]})
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
def test_lenet_accuracy(mnist_path_option, lenet_ckpt_path_option):
    """
    Feature: test accuracy of sim qat work on lenet5.
    Description: Apply sim qat on lenet5 and test accuracy.
    Expectation: accuracy is larger than 0.98.
    """

    from lenet.src.lenet import LeNet5
    from lenet.src.dataset import create_dataset as create_mnist_ds
    mnist_path = mnist_path_option
    if mnist_path_option is None:
        mnist_path = os.getenv("DATASET_PATH", "/home/workspace/mindspore_dataset/mnist")
    data_path = os.path.join(mnist_path, "train")
    ds_train = create_mnist_ds(data_path, 32, 1)
    step_size = ds_train.get_dataset_size()
    network = LeNet5(10)

    # load quantization aware network checkpoint
    ckpt_path = lenet_ckpt_path_option
    if ckpt_path is None:
        ckpt_path = os.path.join(os.getenv("CHECKPOINT_PATH", "/home/workspace/mindspore_ckpt"),
                                 "ckpt/checkpoint_lenet-10_1875.ckpt")
    param_dict = load_checkpoint(ckpt_path)
    mindspore.load_param_into_net(network, param_dict)

    # convert network to quantization aware network
    qat = SimulatedQuantizationAwareTraining({"per_channel": [False, True], "symmetric": [False, True],
                                              "quant_delay": [900, 900]})
    new_network = qat.apply(network)

    # define network loss
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    # define network optimization
    net_opt = nn.Momentum(new_network.trainable_params(), 0.01, 0.9)

    # call back and monitor
    config_ckpt = CheckpointConfig(save_checkpoint_steps=10 * step_size,
                                   keep_checkpoint_max=10)
    ckpt_callback = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ckpt)

    # define model
    model = Model(new_network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Training ==============")
    model.train(10, ds_train, callbacks=[ckpt_callback],
                dataset_sink_mode=True)
    print("============== End Training ==============")

    ds_eval = create_mnist_ds(os.path.join(mnist_path, "test"), 32, 1)

    print("============== Starting Testing ==============")
    acc = model.eval(ds_eval, dataset_sink_mode=True)
    print("============== {} ==============".format(acc))
    assert acc['Accuracy'] > 0.98
