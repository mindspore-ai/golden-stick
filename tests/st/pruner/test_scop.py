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
import pytest
import numpy as np
import mindspore
from mindspore import nn, context
from mindspore_gs.pruner.scop.scop_pruner import PrunerKfCompressAlgo, KfConv2d

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../models/official/cv/'))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("run_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_resnet(run_mode):
    """
    Feature: Simulated quantization algorithm.
    Description: Apply simulated_quantization on resnet.
    Expectation: Apply success.
    """

    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../models/official/cv/resnet/'))
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
    from models.resnet import resnet50

    mindspore.context.set_context(mode=run_mode, device_target="GPU")

    network = resnet50(10)
    pruner_kf = PrunerKfCompressAlgo({})
    new_network = pruner_kf.apply(network)
    flag = False
    for _, (_, module) in enumerate(new_network.cells_and_names()):
        if isinstance(module, KfConv2d):
            flag = True
    assert flag
    print("============== test resnet scop success ==============")


def _create_resnet_accuracy_model(run_mode=context.GRAPH_MODE):
    """
    create model lr dataset for resnet simqat accuracy test.
    merge into test_resnet_accuracy after pynative bug is fixed.
    """
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../models/official/cv/resnet/'))
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
    import mindspore.dataset as ds
    from resnet.src.lr_generator import get_lr
    from mindspore.train.loss_scale_manager import FixedLossScaleManager
    from models.resnet import resnet50

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
        create a train or evaluate cifar10 dataset for resnet50
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

    dataset = _create_dataset(dataset_path=dataset_path, batch_size=64, train_image_size=224)
    step_size = dataset.get_dataset_size()
    net = resnet50(class_num=class_num)
    _init_weight(net=net)

    # apply golden-stick algo
    algo = PrunerKfCompressAlgo({})
    net = algo.apply(net)
    net.set_train()

    lr = get_lr(lr_init=lr_init, lr_end=lr_end, lr_max=lr_max, warmup_epochs=warmup_epochs, total_epochs=epoch_size,
                steps_per_epoch=step_size, lr_decay_mode=lr_decay_mode)
    lr = mindspore.Tensor(lr)
    # define opt
    opt = nn.Momentum(filter(lambda p: p.requires_grad, net.get_parameters()), lr, momentum, loss_scale=loss_scale)
    loss_scale = FixedLossScaleManager(1024, drop_overflow_update=False)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    metrics = {"acc"}
    metrics.clear()
    model = mindspore.Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=metrics,
                            keep_batchnorm_fp32=False)
    return model, lr, dataset


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_resnet_accuracy_pynative():
    """
    Feature: Simulated quantization algorithm.
    Description: Apply simulated_quantization on resnet and test accuracy
    Expectation: Loss of first epoch is smaller than 4.52.
    """
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
    from loss_monitor import LossMonitor

    step_threshold = 20
    target = "GPU"
    epoch_size = 1

    mindspore.context.set_context(mode=context.PYNATIVE_MODE, device_target=target)
    model, lr, dataset = _create_resnet_accuracy_model(context.PYNATIVE_MODE)
    # define callbacks
    monitor = LossMonitor(lr_init=lr.asnumpy(), step_threshold=step_threshold)
    callbacks = [monitor]
    # train model
    dataset_sink_mode = target != "CPU"
    print("============== Starting Training ==============")
    model.train(epoch_size, dataset, callbacks=callbacks, sink_size=dataset.get_dataset_size(),
                dataset_sink_mode=dataset_sink_mode)
    print("============== End Training ==============")
    expect_avg_step_loss = 4.52
    avg_step_loss = np.mean(np.array(monitor.losses))
    print("average step loss:{}".format(avg_step_loss))
    assert avg_step_loss <= expect_avg_step_loss


def test_resnet_accuracy_graph():
    """
    Feature: Simulated quantization algorithm.
    Description: Apply simulated_quantization on resnet and test accuracy
    Expectation: Loss of first epoch is smaller than 2.5.
    """
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
    from loss_monitor import LossMonitor

    step_threshold = 20
    target = "GPU"
    epoch_size = 1

    mindspore.context.set_context(mode=context.GRAPH_MODE, device_target=target)
    model, lr, dataset = _create_resnet_accuracy_model(context.GRAPH_MODE)
    # define callbacks
    monitor = LossMonitor(lr_init=lr.asnumpy(), step_threshold=step_threshold)
    callbacks = [monitor]
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
