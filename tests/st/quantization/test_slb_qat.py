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
import random
from collections import OrderedDict
import pytest
import numpy as np
import mindspore
from mindspore import nn, context
import mindspore.train.callback as cb
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore_gs.quantization.slb import SlbQuantAwareTraining as SlbQAT
from mindspore_gs.quantization.constant import QuantDtype
from mindspore_gs.quantization.slb.slb_fake_quantizer import SlbFakeQuantizerPerLayer
from mindspore_gs.quantization.quantize_wrapper_cell import QuantizeWrapperCell

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../models/official/cv/'))


class NetToQuant(nn.Cell):
    """
    Network with single conv2d to be quanted.
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
@pytest.mark.parametrize("quant_bit", ["W4", "W2", "W1"])
def test_set_config(quant_bit):
    """
    Feature: SLB(Searching for Low-Bit Weights) QAT-algorithm set functions.
    Description: Apply SlbQuantAwareTraining on lenet.
    Expectation: Apply success and coordinate attributes are same as config.
    """

    network = NetToQuant()
    qat = SlbQAT()
    if quant_bit == "W4":
        qat.set_weight_quant_dtype(QuantDtype.INT4)
    elif quant_bit == "W2":
        qat.set_weight_quant_dtype(QuantDtype.INT2)
    elif quant_bit == "W1":
        qat.set_weight_quant_dtype(QuantDtype.INT1)
    new_network = qat.apply(network)
    cells: OrderedDict = new_network.name_cells()

    assert cells.get("Conv2dSlbQuant", None) is not None
    conv_quant: QuantizeWrapperCell = cells.get("Conv2dSlbQuant")
    assert isinstance(conv_quant, QuantizeWrapperCell)
    conv_handler = conv_quant._handler
    weight_fake_quant: SlbFakeQuantizerPerLayer = conv_handler.fake_quant_weight
    assert isinstance(weight_fake_quant, SlbFakeQuantizerPerLayer)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_bit", ["W4", "W2", "W1"])
def test_lenet(quant_bit):
    """
    Feature: slb quantization algorithm.
    Description: Apply slb qat on lenet.
    Expectation: Apply success.
    """

    from lenet.src.lenet import LeNet5
    network = LeNet5(10)
    if quant_bit == "W4":
        qat = SlbQAT({"quant_dtype": QuantDtype.INT4})
    elif quant_bit == "W2":
        qat = SlbQAT({"quant_dtype": QuantDtype.INT2})
    elif quant_bit == "W1":
        qat = SlbQAT({"quant_dtype": QuantDtype.INT1})
    new_network = qat.apply(network)
    cells: OrderedDict = new_network.name_cells()
    assert cells.get("Conv2dSlbQuant", None) is not None
    conv_quant: QuantizeWrapperCell = cells.get("Conv2dSlbQuant")
    assert isinstance(conv_quant, QuantizeWrapperCell)
    conv_handler = conv_quant._handler
    weight_fake_quant: SlbFakeQuantizerPerLayer = conv_handler.fake_quant_weight
    assert isinstance(weight_fake_quant, SlbFakeQuantizerPerLayer)



@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_bit", ["W4", "W2", "W1"])
@pytest.mark.parametrize("run_mode", [context.GRAPH_MODE])
def test_lenet_accuracy(mnist_path_option, quant_bit, run_mode):
    """
    Feature: test accuracy of slb qat work on lenet5.
    Description: Apply slb qat on lenet5 and test accuracy.
    Expectation: accuracy is larger than 0.98.
    """

    from lenet.src.lenet import LeNet5
    from lenet.src.dataset import create_dataset as create_mnist_ds
    context.set_context(mode=run_mode)
    mnist_path = mnist_path_option
    if mnist_path_option is None:
        mnist_path = os.getenv("DATASET_PATH", "/home/workspace/mindspore_dataset/mnist")
    data_path = os.path.join(mnist_path, "train")
    ds_train = create_mnist_ds(data_path, 32, 1)
    network = LeNet5(10)

    class TemperatureScheduler(cb.Callback):
        """
        TemperatureScheduler for SLB.
        """
        def __init__(self, model):
            super().__init__()
            self.epochs = 10
            self.t_start_val = 1.0
            self.t_start_time = 0.2
            self.t_end_time = 0.6
            self.t_factor = 5.2
            self.model = model

        def epoch_begin(self, run_context):
            """
            Epoch_begin.
            """
            cb_params = run_context.original_args()
            epoch = cb_params.cur_epoch_num
            # Compute temperature value
            t = self.t_start_val
            t_start_epoch = int(self.epochs*self.t_start_time)
            t_end_epoch = int(self.epochs*self.t_end_time)
            if epoch > t_start_epoch:
                t *= self.t_factor**(min(epoch, t_end_epoch) - t_start_epoch)
            # Assign new value to temperature parameter
            for _, cell in self.model.train_network.cells_and_names():
                if cell.cls_name == 'SlbFakeQuantizerPerLayer': # for SLB
                    cell.set_temperature(t)
                    if epoch >= t_end_epoch:
                        cell.set_temperature_end_flag()


    # convert network to quantization aware network
    if quant_bit == "W4":
        qat = SlbQAT({"quant_dtype": QuantDtype.INT4})
    elif quant_bit == "W2":
        qat = SlbQAT({"quant_dtype": QuantDtype.INT2})
    elif quant_bit == "W1":
        qat = SlbQAT({"quant_dtype": QuantDtype.INT1})
    new_network = qat.apply(network)

    # define network loss
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    # define network optimization
    net_opt = nn.Momentum(new_network.trainable_params(), 0.01, 0.9)

    # define model
    model = Model(new_network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Training ==============")
    model.train(10, ds_train, callbacks=[TemperatureScheduler(model)])
    print("============== End Training ==============")

    ds_eval = create_mnist_ds(os.path.join(mnist_path, "test"), 32, 1)

    print("============== Starting Testing ==============")
    acc = model.eval(ds_eval)
    print("============== {} ==============".format(acc))
    assert acc['Accuracy'] > 0.95



@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_bit", ["W4", "W2", "W1"])
@pytest.mark.parametrize("run_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_resnet(quant_bit, run_mode):
    """
    Feature: slb quantization algorithm.
    Description: Apply slb qat on resnet.
    Expectation: Apply success.
    """

    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../models/official/cv/resnet/'))
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
    from models.resnet import resnet18

    mindspore.context.set_context(mode=run_mode, device_target="GPU")

    network = resnet18(10)
    qat = SlbQAT()
    if quant_bit == "W4":
        qat.set_weight_quant_dtype(QuantDtype.INT4)
    elif quant_bit == "W2":
        qat.set_weight_quant_dtype(QuantDtype.INT2)
    elif quant_bit == "W1":
        qat.set_weight_quant_dtype(QuantDtype.INT1)
    new_network = qat.apply(network)

    cells: OrderedDict = new_network.name_cells()
    assert cells.get("Conv2dSlbQuant", None) is not None
    conv_quant: QuantizeWrapperCell = cells.get("Conv2dSlbQuant")
    assert isinstance(conv_quant, QuantizeWrapperCell)
    conv_handler = conv_quant._handler
    weight_fake_quant: SlbFakeQuantizerPerLayer = conv_handler.fake_quant_weight
    assert isinstance(weight_fake_quant, SlbFakeQuantizerPerLayer)
    print("============== test resnet slbqat success ==============")


def _create_resnet_accuracy_model(quant_bit, run_mode=context.GRAPH_MODE):
    """
    Create model lr dataset for resnet slbqat accuracy test.
    Merge into test_resnet_accuracy after pynative bug is fixed.
    """
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../models/official/cv/resnet/'))
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
    import mindspore.dataset as ds
    from resnet.src.lr_generator import get_lr
    from mindspore.train.loss_scale_manager import FixedLossScaleManager
    from models.resnet import resnet18

    # config
    dataset_path = os.path.join("/home/workspace/mindspore_dataset/cifar-10-batches-bin")
    target = "GPU"
    class_num = 10
    epoch_size = 1
    warmup_epochs = 0
    lr_decay_mode = "cosine"
    lr_init = 0.01
    lr_end = 0.00001
    lr_max = 0.1
    loss_scale = 1024
    momentum = 0.9
    weight_decay = 0.0001

    mindspore.set_seed(1)
    np.random.seed(1)
    random.seed(1)
    mindspore.context.set_context(mode=run_mode, device_target=target)

    def _init_weight(net):
        """init_weight"""
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(mindspore.common.initializer.initializer(
                    mindspore.common.initializer.XavierUniform(), cell.weight.shape, cell.weight.dtype))
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(mindspore.common.initializer.initializer(
                    mindspore.common.initializer.TruncatedNormal(), cell.weight.shape, cell.weight.dtype))
            if isinstance(cell, nn.BatchNorm2d):
                cell.use_batch_statistics = False

    def _create_dataset(dataset_path, batch_size=128, train_image_size=224):
        """
        Create a train or evaluate cifar10 dataset for resnet50.
        Args:
            dataset_path(string): the path of dataset.
            batch_size(int): the batch size of dataset. Default: 128

        Returns:
            dataset
        """
        ds.config.set_prefetch_size(64)
        data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=1, shuffle=False, num_samples=20000)

        # define map operations
        trans = [
            ds.vision.c_transforms.Resize((train_image_size, train_image_size)),
            ds.vision.c_transforms.Rescale(1.0 / 255.0, 0.0),
            ds.vision.c_transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ds.vision.c_transforms.HWC2CHW()
        ]

        type_cast_op = ds.transforms.c_transforms.TypeCast(mindspore.int32)

        data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=1)
        # only enable cache for eval
        data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=1)

        # apply batch operations
        data_set = data_set.batch(batch_size, drop_remainder=True)

        return data_set

    def _init_group_params(net):
        decayed_params = []
        no_decayed_params = []
        for param in net.trainable_params():
            if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
                decayed_params.append(param)
            else:
                no_decayed_params.append(param)

        group_params = [{'params': decayed_params, 'weight_decay': weight_decay},
                        {'params': no_decayed_params},
                        {'order_params': net.trainable_params()}]
        return group_params

    dataset = _create_dataset(dataset_path=dataset_path, batch_size=64, train_image_size=224)
    step_size = dataset.get_dataset_size()
    net = resnet18(class_num=class_num)
    _init_weight(net=net)

    # apply golden-stick algo
    qat = SlbQAT()
    if quant_bit == "W4":
        qat.set_weight_quant_dtype(QuantDtype.INT4)
    elif quant_bit == "W2":
        qat.set_weight_quant_dtype(QuantDtype.INT2)
    elif quant_bit == "W1":
        qat.set_weight_quant_dtype(QuantDtype.INT1)
    net = qat.apply(net)

    lr = get_lr(lr_init=lr_init, lr_end=lr_end, lr_max=lr_max, warmup_epochs=warmup_epochs, total_epochs=epoch_size,
                steps_per_epoch=step_size, lr_decay_mode=lr_decay_mode)
    lr = mindspore.Tensor(lr)
    # define opt
    group_params = _init_group_params(net)
    opt = nn.Momentum(group_params, lr, momentum, weight_decay=weight_decay, loss_scale=loss_scale)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss_scale = FixedLossScaleManager(loss_scale, drop_overflow_update=False)

    metrics = {"acc"}
    metrics.clear()
    model = mindspore.Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=metrics,
                            keep_batchnorm_fp32=False)
    return model, lr, dataset, qat


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_bit", ["W4", "W2", "W1"])
def test_resnet_accuracy_graph(quant_bit):
    """
    Feature: slb quantization algorithm.
    Description: Apply slb qat on resnet and test accuracy
    Expectation: Loss of first epoch is smaller than 2.5.
    """

    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
    from loss_monitor import LossMonitor

    step_threshold = 20
    target = "GPU"
    epoch_size = 1

    class TemperatureScheduler(cb.Callback):
        """
        TemperatureScheduler for QBNN.
        """
        def __init__(self, model):
            super().__init__()
            self.epochs = epoch_size
            self.t_start_val = 1.0
            self.t_start_time = 0.2
            self.t_end_time = 0.6
            self.t_factor = 1.2
            self.model = model

        def epoch_begin(self, run_context):
            """
            Epoch_begin.
            """
            cb_params = run_context.original_args()
            epoch = cb_params.cur_epoch_num
            # Compute temperature value
            t = self.t_start_val
            t_start_epoch = int(self.epochs*self.t_start_time)
            t_end_epoch = int(self.epochs*self.t_end_time)
            if epoch > t_start_epoch:
                t *= self.t_factor**(min(epoch, t_end_epoch) - t_start_epoch)
            # Assign new value to temperature parameter
            for _, cell in self.model.train_network.cells_and_names():
                if cell.cls_name == 'SlbFakeQuantizerPerLayer': # for SLB
                    cell.set_temperature(t)
                    if epoch >= t_end_epoch:
                        cell.set_temperature_end_flag()

    mindspore.context.set_context(mode=context.GRAPH_MODE, device_target=target)
    model, lr, dataset, qat = _create_resnet_accuracy_model(quant_bit, context.GRAPH_MODE)

    # define callbacks
    monitor = LossMonitor(lr_init=lr.asnumpy(), step_threshold=step_threshold)
    callbacks = [monitor, TemperatureScheduler(model), qat.callback()]
    # train model
    dataset_sink_mode = target != "CPU"
    print("============== Starting Training ==============")
    model.train(epoch_size, dataset, callbacks=callbacks, sink_size=dataset.get_dataset_size(),
                dataset_sink_mode=dataset_sink_mode)
    print("============== End Training ==============")
    expect_avg_step_loss = 2.5
    avg_step_loss = np.mean(np.array(monitor.losses))
    print("average step loss:{}".format(avg_step_loss))
    assert avg_step_loss <= expect_avg_step_loss


def test_resnet_accuracy_pynative(quant_bit):
    """
    Feature: Simulated quantization algorithm.
    Description: Apply simulated_quantization on resnet and test accuracy
    Expectation: Loss of first epoch is smaller than 2.8.
    """
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
    from loss_monitor import LossMonitor

    step_threshold = 20
    target = "GPU"
    epoch_size = 1

    class TemperatureScheduler(cb.Callback):
        """
        TemperatureScheduler for QBNN.
        """
        def __init__(self, model):
            super().__init__()
            self.epochs = epoch_size
            self.t_start_val = 1.0
            self.t_start_time = 0.2
            self.t_end_time = 0.6
            self.t_factor = 1.2
            self.model = model

        def epoch_begin(self, run_context):
            """
            Epoch_begin.
            """
            cb_params = run_context.original_args()
            epoch = cb_params.cur_epoch_num
            # Compute temperature value
            t = self.t_start_val
            t_start_epoch = int(self.epochs*self.t_start_time)
            t_end_epoch = int(self.epochs*self.t_end_time)
            if epoch > t_start_epoch:
                t *= self.t_factor**(min(epoch, t_end_epoch) - t_start_epoch)
            # Assign new value to temperature parameter
            for _, cell in self.model.train_network.cells_and_names():
                if cell.cls_name == 'SlbFakeQuantizerPerLayer': # for SLB
                    cell.set_temperature(t)
                    if epoch >= t_end_epoch:
                        cell.set_temperature_end_flag()

    mindspore.context.set_context(mode=context.PYNATIVE_MODE, device_target=target)
    model, lr, dataset, qat = _create_resnet_accuracy_model(quant_bit, context.PYNATIVE_MODE)
    # define callbacks
    monitor = LossMonitor(lr_init=lr.asnumpy(), step_threshold=step_threshold)
    callbacks = [monitor, TemperatureScheduler(model), qat.callback()]
    # train model
    dataset_sink_mode = target != "CPU"
    print("============== Starting Training ==============")
    model.train(epoch_size, dataset, callbacks=callbacks, sink_size=dataset.get_dataset_size(),
                dataset_sink_mode=dataset_sink_mode)
    print("============== End Training ==============")
    expect_avg_step_loss = 4.5
    avg_step_loss = np.mean(np.array(monitor.losses))
    print("average step loss:{}".format(avg_step_loss))
    assert avg_step_loss <= expect_avg_step_loss
