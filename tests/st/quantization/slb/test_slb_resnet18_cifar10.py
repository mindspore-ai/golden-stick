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
"""test slb qat on resnet18 network and cifar10 dataset."""

import os
import sys
import random
from collections import OrderedDict
import pytest
import numpy as np
import mindspore
from mindspore import nn, context
from mindspore import Model
from mindspore.common.dtype import QuantDtype
from mindspore_gs.quantization.slb import SlbQuantAwareTraining as SlbQAT
from mindspore_gs.quantization.slb.slb_fake_quantizer import SlbFakeQuantizerPerLayer
from mindspore_gs.quantization.quantize_wrapper_cell import QuantizeWrapperCell

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../models/official/cv/'))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_bit", ["W4", "W2", "W1", "W4A8", "W2A8", "W1A8"])
@pytest.mark.parametrize("enable_bn_calibration", [True, False])
@pytest.mark.parametrize("run_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_resnet(quant_bit, enable_bn_calibration, run_mode):
    """
    Feature: slb quantization algorithm.
    Description: Apply slb qat on resnet.
    Expectation: Apply success.
    """

    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../models/official/cv/ResNet/'))
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
    from models.resnet import resnet18

    mindspore.context.set_context(mode=run_mode, device_target="GPU")

    network = resnet18(10)
    qat = SlbQAT()
    if "W4" in quant_bit:
        qat.set_weight_quant_dtype(QuantDtype.INT4)
    elif "W2" in quant_bit:
        qat.set_weight_quant_dtype(QuantDtype.INT2)
    elif "W1" in quant_bit:
        qat.set_weight_quant_dtype(QuantDtype.INT1)
    if "A8" in quant_bit:
        qat.set_act_quant_dtype(QuantDtype.INT8)
        qat.set_enable_act_quant(True)
    else:
        qat.set_enable_act_quant(False)
    qat.set_enable_bn_calibration(enable_bn_calibration)
    qat.set_epoch_size(100)
    qat.set_has_trained_epoch(0)
    qat.set_t_start_val(1.0)
    qat.set_t_start_time(0.2)
    qat.set_t_end_time(0.6)
    qat.set_t_factor(1.2)
    new_network = qat.apply(network)

    cells: OrderedDict = new_network.name_cells()
    assert cells.get("Conv2dSlbQuant", None) is not None
    conv_quant: QuantizeWrapperCell = cells.get("Conv2dSlbQuant")
    assert isinstance(conv_quant, QuantizeWrapperCell)
    conv_handler = conv_quant._handler
    weight_fake_quant: SlbFakeQuantizerPerLayer = conv_handler.fake_quant_weight
    assert isinstance(weight_fake_quant, SlbFakeQuantizerPerLayer)
    assert qat._config.enable_bn_calibration == enable_bn_calibration
    assert qat._config.epoch_size == 100
    assert qat._config.has_trained_epoch == 0
    assert qat._config.t_start_val == 1.0
    assert qat._config.t_start_time == 0.2
    assert qat._config.t_end_time == 0.6
    assert qat._config.t_factor == 1.2
    print("============== test resnet slbqat success ==============")



@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("enable_act_quant", [True, False])
@pytest.mark.parametrize("run_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_resnet_convert(run_mode, enable_act_quant):
    """
    Feature: SLB convert function.
    Description: convert a compressed network to a standard network before exporting to MindIR.
    Expectation: convert success and structure of network as expect.
    """

    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../models/official/cv/ResNet/'))
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
    from models.resnet import resnet18
    context.set_context(mode=run_mode)

    network = resnet18(10)
    config = {"quant_dtype": [QuantDtype.INT8, QuantDtype.INT1], "enable_act_quant": enable_act_quant,
              "enable_bn_calibration": False, "epoch_size": 100,
              "has_trained_epoch": 0, "t_start_val": 1.0,
              "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 1.2}
    qat = SlbQAT(config)
    new_network = qat.apply(network)
    new_network = qat.convert(new_network)
    data_in = mindspore.Tensor(np.ones([1, 3, 32, 32]), mindspore.float32)
    file_name = "./resnet_convert_{}_{}.mindir".format(run_mode, enable_act_quant)
    mindspore.export(new_network, data_in, file_name=file_name, file_format="MINDIR")
    graph = mindspore.load(file_name)
    mindspore.nn.GraphCell(graph)


def _create_resnet_accuracy_model(quant_bit, enable_bn_calibration, run_mode=context.GRAPH_MODE):
    """
    Create model lr dataset for resnet slbqat accuracy test.
    Merge into test_resnet_accuracy after pynative bug is fixed.
    """

    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../models/official/cv/ResNet/'))
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
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
    if "W4" in quant_bit:
        qat.set_weight_quant_dtype(QuantDtype.INT4)
    elif "W2" in quant_bit:
        qat.set_weight_quant_dtype(QuantDtype.INT2)
    elif "W1" in quant_bit:
        qat.set_weight_quant_dtype(QuantDtype.INT1)
    if "A8" in quant_bit:
        qat.set_act_quant_dtype(QuantDtype.INT8)
        qat.set_enable_act_quant(True)
    else:
        qat.set_enable_act_quant(False)
    qat.set_enable_bn_calibration(enable_bn_calibration)
    qat.set_epoch_size(100)
    qat.set_has_trained_epoch(0)
    qat.set_t_start_val(1.0)
    qat.set_t_start_time(0.2)
    qat.set_t_end_time(0.6)
    qat.set_t_factor(1.2)
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
    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=metrics,
                  keep_batchnorm_fp32=False)
    return model, lr, dataset, qat


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_bit", ["W4", "W1", "W4A8", "W1A8"])
@pytest.mark.parametrize("enable_bn_calibration", [True])
def test_resnet_accuracy_graph_bnon(quant_bit, enable_bn_calibration):
    """
    Feature: slb quantization algorithm.
    Description: Apply slb qat on resnet and test accuracy
    Expectation: Loss of first epoch is smaller than 2.5.
    """

    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
    from loss_monitor import LossMonitor

    step_threshold = 20
    target = "GPU"
    epoch_size = 1

    mindspore.context.set_context(mode=context.GRAPH_MODE, device_target=target)
    model, lr, dataset, qat = _create_resnet_accuracy_model(quant_bit, enable_bn_calibration, context.GRAPH_MODE)

    # define callbacks
    monitor = LossMonitor(lr_init=lr.asnumpy(), step_threshold=step_threshold)
    callbacks = [monitor] + qat.callbacks(model, dataset)
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



@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_bit", ["W4", "W1", "W4A8", "W1A8"])
@pytest.mark.parametrize("enable_bn_calibration", [False])
def test_resnet_accuracy_graph_bnoff(quant_bit, enable_bn_calibration):
    """
    Feature: slb quantization algorithm.
    Description: Apply slb qat on resnet and test accuracy
    Expectation: Loss of first epoch is smaller than 2.5.
    """

    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
    from loss_monitor import LossMonitor

    step_threshold = 20
    target = "GPU"
    epoch_size = 1

    mindspore.context.set_context(mode=context.GRAPH_MODE, device_target=target)
    model, lr, dataset, qat = _create_resnet_accuracy_model(quant_bit, enable_bn_calibration, context.GRAPH_MODE)

    # define callbacks
    monitor = LossMonitor(lr_init=lr.asnumpy(), step_threshold=step_threshold)
    callbacks = [monitor] + qat.callbacks(model, dataset)
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


def test_resnet_accuracy_pynative(quant_bit, enable_bn_calibration):
    """
    Feature: slb quantization algorithm.
    Description: Apply simulated_quantization on resnet and test accuracy
    Expectation: Loss of first epoch is smaller than 2.8.
    """

    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
    from loss_monitor import LossMonitor

    step_threshold = 20
    target = "GPU"
    epoch_size = 1

    mindspore.context.set_context(mode=context.PYNATIVE_MODE, device_target=target)
    model, lr, dataset, qat = _create_resnet_accuracy_model(quant_bit, enable_bn_calibration, context.PYNATIVE_MODE)
    # define callbacks
    monitor = LossMonitor(lr_init=lr.asnumpy(), step_threshold=step_threshold)
    callbacks = [monitor] + qat.callbacks(model, dataset)
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
