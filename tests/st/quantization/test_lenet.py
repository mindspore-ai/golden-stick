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
"""test lenet."""

from collections import OrderedDict
import os
import pytest
from mindspore_gs.quantization.simulated_quantization import SimulatedQuantizationAwareTraining
from mindspore_gs.quantization.simulated_quantization.simulated_fake_quantizers import SimulatedFakeQuantizerPerLayer, \
    SimulatedFakeQuantizerPerChannel
from mindspore_gs.quantization.quantize_wrapper_cell import QuantizeWrapperCell
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.vision import Inter
from mindspore.common import dtype as mstype
from mindspore import nn
from mindspore.common.initializer import Normal
from mindspore.compression.quant.quant_utils import load_nonquant_param_into_quant_net
from mindspore.train.serialization import load_checkpoint
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy


def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    """
    create dataset for train or test
    """
    # define dataset
    mnist_ds = ds.MnistDataset(data_path)

    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)  # Bilinear mode
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # apply map operations on images
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)  # 10000 as in LeNet train script
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds


class LeNet5(nn.Cell):
    """
    Lenet network

    Args:
        num_class (int): Number of classes. Default: 10.
        num_channel (int): Number of channels. Default: 1.

    Returns:
        Tensor, output tensor
    Examples:
        >>> LeNet5(num_class=10)

    """

    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_lenet():
    """
    Feature: DefaultQuantAwareTraining algorithm.
    Description: Apply DefaultQuantAwareTraining on lenet.
    Expectation: Apply success.
    """

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
def test_lenet_accuracy():
    """
    Feature: DefaultQuantAwareTraining algorithm.
    Description: Apply DefaultQuantAwareTraining on lenet and test accuracy.
    Expectation: accuracy is larger than 0.98.
    """
    data_path = os.getenv("DATASET_PATH", "/home/workspace/mindspore_dataset/mnist")
    ds_train = create_dataset(os.path.join(data_path, "train"), 32, 1)
    step_size = ds_train.get_dataset_size()
    network = LeNet5(10)

    # load quantization aware network checkpoint
    ckpt_path = os.getenv("CHECKPOINT_PATH", "/home/workspace/mindspore_ckpt")
    ckpt_file = os.path.join(ckpt_path, "ckpt/checkpoint_lenet-10_1875.ckpt")
    param_dict = load_checkpoint(ckpt_file)
    load_nonquant_param_into_quant_net(network, param_dict)

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

    ds_eval = create_dataset(os.path.join(data_path, "test"), 32, 1)

    print("============== Starting Testing ==============")
    acc = model.eval(ds_eval, dataset_sink_mode=True)
    print("============== {} ==============".format(acc))
    assert acc['Accuracy'] > 0.98
