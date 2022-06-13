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
from mindspore import nn, context
from mindspore.train.serialization import load_checkpoint
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore_gs.quantization.simulated_quantization import SimulatedQuantizationAwareTraining as SimQAT
from mindspore_gs.quantization.learned_step_size_quantization import LearnedStepSizeQuantizationAwareTraining as \
    LearnedQAT
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
    Feature: SimQAT algorithm set functions.
    Description: Apply DefaultQuantAwareTraining on lenet.
    Expectation: Apply success and coordinate attributes are same as config.
    """

    network = NetToQuant()
    qat = SimQAT()
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
    Feature: set_enable_fusion api of SimQAT.
    Description: Check default value of enable_fusion and value after called set_enable_fusion.
    Expectation: Config success.
    """
    qat = SimQAT()
    assert not qat._config.enable_fusion
    qat.set_enable_fusion(True)
    assert qat._config.enable_fusion


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_config_one_conv_fold():
    """
    Feature: set_one_conv_fold api of SimQAT.
    Description: Check default value of one_conv_fold and value after called set_one_conv_fold.
    Expectation: Config success.
    """
    qat = SimQAT()
    assert qat._config.one_conv_fold
    qat.set_one_conv_fold(False)
    assert not qat._config.one_conv_fold


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_config_freeze_bn():
    """
    Feature: set_freeze_bn api of SimQAT.
    Description: Check default value of freeze_bn and value after called set_freeze_bn.
    Expectation: Config success.
    """
    qat = SimQAT()
    assert qat._config.freeze_bn == 10000000
    qat.set_freeze_bn(0)
    assert qat._config.freeze_bn == 0


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
@pytest.mark.parametrize("algorithm", [SimQAT])
@pytest.mark.parametrize("run_mode", [context.GRAPH_MODE])
def test_lenet_accuracy(mnist_path_option, lenet_ckpt_path_option, algorithm, run_mode):
    """
    Feature: test accuracy of sim qat work on lenet5.
    Description: Apply sim qat on lenet5 and test accuracy.
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

    # load quantization aware network checkpoint
    ckpt_path = lenet_ckpt_path_option
    if ckpt_path is None:
        ckpt_path = os.path.join(os.getenv("CHECKPOINT_PATH", "/home/workspace/mindspore_ckpt"),
                                 "ckpt/checkpoint_lenet-10_1875.ckpt")
    param_dict = load_checkpoint(ckpt_path)
    mindspore.load_param_into_net(network, param_dict)

    # convert network to quantization aware network
    if algorithm == SimQAT:
        qat = SimQAT({"per_channel": [False, True], "symmetric": [False, True],
                      "quant_delay": [900, 900]})
    else:
        qat = LearnedQAT()
    new_network = qat.apply(network)

    # define network loss
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    # define network optimization
    net_opt = nn.Momentum(new_network.trainable_params(), 0.01, 0.9)

    # define model
    model = Model(new_network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Training ==============")
    model.train(10, ds_train, callbacks=[])
    print("============== End Training ==============")

    ds_eval = create_mnist_ds(os.path.join(mnist_path, "test"), 32, 1)

    print("============== Starting Testing ==============")
    acc = model.eval(ds_eval)
    print("============== {} ==============".format(acc))
    assert acc['Accuracy'] > 0.98


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
    from resnet.golden_stick.quantization.simqat.simqat import create_simqat
    from models.resnet import resnet50

    mindspore.context.set_context(mode=run_mode, device_target="GPU")

    network = resnet50(10)
    qat = create_simqat()
    new_network = qat.apply(network)

    cells: OrderedDict = new_network.name_cells()
    assert cells.get("Conv2dBnFoldQuant", None) is not None
    conv_quant: QuantizeWrapperCell = cells.get("Conv2dBnFoldQuant")
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

    assert cells.get("layer1", None) is not None
    seq_cell: nn.Cell = cells.get("layer1")
    res_block: nn.Cell = seq_cell.name_cells().get("cell_list_0")
    res_block_cells: OrderedDict = res_block.name_cells()
    assert res_block_cells.get("Conv2dBnFoldQuant", None) is not None
    res_block_conv_quant: QuantizeWrapperCell = cells.get("Conv2dBnFoldQuant")
    assert isinstance(res_block_conv_quant, QuantizeWrapperCell)
    res_block_conv_handler = res_block_conv_quant._handler
    res_block_conv_weight_fake_quant: SimulatedFakeQuantizerPerChannel = res_block_conv_handler.fake_quant_weight
    assert isinstance(res_block_conv_weight_fake_quant, SimulatedFakeQuantizerPerChannel)
    assert res_block_conv_weight_fake_quant._symmetric
    assert res_block_conv_weight_fake_quant._quant_delay == 900
    res_block_conv_act_fake_quant = res_block_conv_quant._output_quantizer
    assert isinstance(res_block_conv_act_fake_quant, SimulatedFakeQuantizerPerLayer)
    assert not res_block_conv_act_fake_quant._symmetric
    assert res_block_conv_act_fake_quant._quant_delay == 900
    print("============== test resnet simqat success ==============")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_resnet_accuracy_graph():
    """
    Feature: Simulated quantization algorithm.
    Description: Apply simulated_quantization on resnet and test accuracy
    Expectation: Accuracy of is bigger than 0.45.
    """
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../models/official/cv/resnet/'))
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
    import mindspore.dataset as ds
    from resnet.golden_stick.quantization.simqat.simqat import create_simqat
    from resnet.src.lr_generator import get_lr
    from mindspore.train.loss_scale_manager import FixedLossScaleManager
    from models.resnet import resnet18

    # config
    train_ds_path = os.path.join("/home/workspace/mindspore_dataset/cifar-10-batches-bin")
    test_ds_path = os.path.join("/home/workspace/mindspore_dataset/cifar-10-verify-bin")
    target = "GPU"
    class_num = 10
    epoch_size = 70
    warmup_epochs = 0
    lr_decay_mode = "cosine"
    lr_init = 0.005
    lr_end = 0.00001
    lr_max = 0.005
    loss_scale = 1024
    momentum = 0.9
    weight_decay = 0.0001
    run_mode = context.GRAPH_MODE

    mindspore.set_seed(1)
    mindspore.context.set_context(mode=run_mode, device_target=target)

    def _init_weight(net):
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(mindspore.common.initializer.initializer(
                    mindspore.common.initializer.XavierUniform(), cell.weight.shape, cell.weight.dtype))
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(mindspore.common.initializer.initializer(
                    mindspore.common.initializer.TruncatedNormal(), cell.weight.shape, cell.weight.dtype))

    def _create_dataset(dataset_path, do_train, batch_size=128, train_image_size=224):
        ds.config.set_prefetch_size(64)
        if do_train:
            data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=1, shuffle=True, num_samples=500)
        else:
            data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=1, shuffle=True)

        # define map operations
        trans = []
        if do_train:
            trans += [
                ds.vision.RandomCrop((32, 32), (4, 4, 4, 4)),
                ds.vision.RandomHorizontalFlip(prob=0.5)
            ]

        trans += [
            ds.vision.Resize((train_image_size, train_image_size)),
            ds.vision.Rescale(1.0 / 255.0, 0.0),
            ds.vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ds.vision.HWC2CHW()
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

    train_ds = _create_dataset(dataset_path=train_ds_path, do_train=True, batch_size=16, train_image_size=224)
    step_size = train_ds.get_dataset_size()
    net = resnet18(class_num=class_num)
    _init_weight(net=net)

    # apply golden-stick algo
    algo = create_simqat()
    net = algo.apply(net)

    lr = get_lr(lr_init=lr_init, lr_end=lr_end, lr_max=lr_max, warmup_epochs=warmup_epochs, total_epochs=epoch_size,
                steps_per_epoch=step_size, lr_decay_mode=lr_decay_mode)
    lr = mindspore.Tensor(lr)
    # define opt
    group_params = _init_group_params(net)
    opt = nn.Momentum(group_params, lr, momentum, weight_decay=weight_decay, loss_scale=loss_scale)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss_scale = FixedLossScaleManager(loss_scale, drop_overflow_update=False)

    metrics = {"acc"}
    model = mindspore.Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=metrics,
                            keep_batchnorm_fp32=False)
    # train model
    dataset_sink_mode = target != "CPU"
    print("============== Starting Training ==============")
    model.train(epoch_size, train_ds, callbacks=[algo.callback()], sink_size=train_ds.get_dataset_size(),
                dataset_sink_mode=dataset_sink_mode)
    print("============== End Training ==============")
    print("============== Starting Test ==============")
    val_ds = _create_dataset(dataset_path=test_ds_path, do_train=False, batch_size=64, train_image_size=224)
    metrics_res = model.eval(val_ds)
    acc = metrics_res['acc']
    print(f"============== acc: {acc} ==============")
    assert acc >= 0.45
    print("============== End Test ==============")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_resnet_accuracy_pynative():
    """
    Feature: Simulated quantization algorithm.
    Description: Apply simulated_quantization on resnet and test accuracy
    Expectation: Accuracy of is bigger than 0.3.
    """
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../models/official/cv/resnet/'))
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
    import mindspore.dataset as ds
    from resnet.golden_stick.quantization.simqat.simqat import create_simqat
    from resnet.src.lr_generator import get_lr
    from mindspore.train.loss_scale_manager import FixedLossScaleManager
    from models.resnet import resnet18

    # config
    train_ds_path = os.path.join("/home/workspace/mindspore_dataset/cifar-10-batches-bin")
    test_ds_path = os.path.join("/home/workspace/mindspore_dataset/cifar-10-verify-bin")
    target = "GPU"
    class_num = 10
    epoch_size = 30
    warmup_epochs = 0
    lr_decay_mode = "cosine"
    lr_init = 0.005
    lr_end = 0.00001
    lr_max = 0.005
    loss_scale = 1024
    momentum = 0.9
    weight_decay = 0.0001
    run_mode = context.PYNATIVE_MODE

    mindspore.set_seed(1)
    mindspore.context.set_context(mode=run_mode, device_target=target)
    if run_mode == context.PYNATIVE_MODE:
        epoch_size = 30

    def _init_weight(net):
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(mindspore.common.initializer.initializer(
                    mindspore.common.initializer.XavierUniform(), cell.weight.shape, cell.weight.dtype))
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(mindspore.common.initializer.initializer(
                    mindspore.common.initializer.TruncatedNormal(), cell.weight.shape, cell.weight.dtype))

    def _create_dataset(dataset_path, do_train, batch_size=128, train_image_size=224):
        ds.config.set_prefetch_size(64)
        if do_train:
            data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=1, shuffle=True, num_samples=500)
        else:
            data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=1, shuffle=True)

        # define map operations
        trans = []
        if do_train:
            trans += [
                ds.vision.RandomCrop((32, 32), (4, 4, 4, 4)),
                ds.vision.RandomHorizontalFlip(prob=0.5)
            ]

        trans += [
            ds.vision.Resize((train_image_size, train_image_size)),
            ds.vision.Rescale(1.0 / 255.0, 0.0),
            ds.vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ds.vision.HWC2CHW()
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

    train_ds = _create_dataset(dataset_path=train_ds_path, do_train=True, batch_size=16, train_image_size=224)
    step_size = train_ds.get_dataset_size()
    net = resnet18(class_num=class_num)
    _init_weight(net=net)

    # apply golden-stick algo
    algo = create_simqat()
    net = algo.apply(net)

    lr = get_lr(lr_init=lr_init, lr_end=lr_end, lr_max=lr_max, warmup_epochs=warmup_epochs, total_epochs=epoch_size,
                steps_per_epoch=step_size, lr_decay_mode=lr_decay_mode)
    lr = mindspore.Tensor(lr)
    # define opt
    group_params = _init_group_params(net)
    opt = nn.Momentum(group_params, lr, momentum, weight_decay=weight_decay, loss_scale=loss_scale)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss_scale = FixedLossScaleManager(loss_scale, drop_overflow_update=False)

    metrics = {"acc"}
    model = mindspore.Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics=metrics,
                            keep_batchnorm_fp32=False)
    # train model
    dataset_sink_mode = target != "CPU"
    print("============== Starting Training ==============")
    model.train(epoch_size, train_ds, callbacks=[algo.callback()], sink_size=train_ds.get_dataset_size(),
                dataset_sink_mode=dataset_sink_mode)
    print("============== End Training ==============")
    print("============== Starting Test ==============")
    val_ds = _create_dataset(dataset_path=test_ds_path, do_train=False, batch_size=64, train_image_size=224)
    metrics_res = model.eval(val_ds)
    acc = metrics_res['acc']
    print(f"============== acc: {acc} ==============")
    assert acc >= 0.3
    print("============== End Test ==============")
