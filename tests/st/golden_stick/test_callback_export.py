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
"""test Callback export."""

import os
import sys

import pytest
import numpy as np

import mindspore
from mindspore import context, Tensor, nn
from mindspore.train import Model
from mindspore.train.metrics import Accuracy
from mindspore_gs.comp_algo import CompAlgo

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../models/official/cv/'))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("run_mode", [context.GRAPH_MODE])
def test_lenet_export_mindir(run_mode):
    """
    Feature: test callback export MindIR train lenet.
    Description: Set save mindir True and train lenet.
    Expectation: MindIR is exported automatically.
    """

    from lenet.src.lenet import LeNet5
    from lenet.src.dataset import create_dataset as create_mnist_ds
    context.set_context(mode=run_mode)
    mnist_path = os.getenv("DATASET_PATH", "/home/workspace/mindspore_dataset/mnist")
    data_path = os.path.join(mnist_path, "train")
    ds_train = create_mnist_ds(data_path, 32, 1)
    if ds_train.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")
    network = LeNet5(10)

    algo = CompAlgo({})
    algo.set_save_mindir(save_mindir=True)
    mindir_path = os.path.join(os.path.dirname(__file__), "lenet")
    algo.set_save_mindir_path(save_mindir_path=mindir_path)

    network = algo.apply(network)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)

    context.set_context(enable_graph_kernel=True)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    model.train(1, ds_train, callbacks=algo.callbacks())

    mindir_file_path = mindir_path + ".mindir"
    assert os.path.exists(mindir_file_path)

    input_tensor = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32))
    graph = mindspore.load(mindir_file_path)
    net = nn.GraphCell(graph)
    _ = net(input_tensor)
