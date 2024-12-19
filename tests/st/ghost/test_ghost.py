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
import pytest
import mindspore
from mindspore import nn, context
from mindspore_gs.ghost.ghost import GhostAlgo, GhostModule
from tests.st import test_utils as utils

cur_path = os.path.dirname(os.path.abspath(__file__))
model_name = "ResNet"
config_name = "resnet50_cifar10_config.yaml"
ori_model_path = os.path.join(cur_path, "../../../tests/models/official/cv")
train_log_rpath = os.path.join("golden_stick", "scripts", "train_parallel", "log")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("run_mode", [context.GRAPH_MODE])
def test_resnet_apply(run_mode):
    """
    Feature: Simulated quantization algorithm.
    Description: Apply simulated_quantization on resnet.
    Expectation: Apply success.
    """

    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../models/official/cv/ResNet/'))
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
    from models.resnet import resnet50

    mindspore.context.set_context(mode=run_mode, device_target="GPU")

    network = resnet50(10)
    algo = GhostAlgo({})
    new_network = algo.apply(network)
    flag = False
    for num, (_, module) in enumerate(new_network.cells_and_names()):
        if isinstance(module, GhostModule):
            flag = True
        if num == 1:
            assert isinstance(module, nn.Conv2d)
    assert flag


@pytest.mark.skip(reason="mindspore update 2.5.0 cause NAN.")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_resnet_accuracy_graph():
    """
    Feature: Ghost algorithm.
    Description: Apply GhostAlgo on resnet and test accuracy
    Expectation: accuracy is larger than 0.08 after 2 epochs.
    """

    is_self_check = os.getenv("SELF_CHECK", "False")
    if is_self_check == "True":
        epochs = 500
        acc_thres = 0.93
    else:
        epochs = 1
        acc_thres = 0.04

    config = utils.TrainEvalConfig.run_mode_train_eval_config("GRAPH")
    config.set_config(utils.TrainEvalConfig.epoch_size_key, epochs)

    model_path = utils.train_network(ori_model_path, model_name, "test_gpu_accuracy_graph", config_name,
                                     "ghost", "run_distribute_train_gpu.sh", config, "cifar10", 740,
                                     train_log_rpath=train_log_rpath)
    acc = utils.eval_network(model_path, model_name, config_name, "ghost", "run_eval_gpu.sh",
                             "train_parallel/output/checkpoint/ckpt_0/resnet-{}_48.ckpt".format(epochs), config,
                             "cifar10", 200, "log", "'top_1_accuracy': ([0-9.]*)", train_log_rpath=train_log_rpath)
    assert acc > acc_thres
