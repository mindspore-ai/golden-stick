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
import numpy as np
import pytest
import mindspore
from mindspore import nn, context
from mindspore_gs import PrunerKfCompressAlgo, PrunerFtCompressAlgo

cur_path = os.path.dirname(os.path.abspath(__file__))
model_name = "ResNet"
config_name = "resnet50_cifar10_config.yaml"
ori_model_path = os.path.join(cur_path, "../../../../tests/models/official/cv")
train_log_rpath = os.path.join("golden_stick", "scripts", "train_parallel", "log")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("run_mode", [context.GRAPH_MODE])
def test_scop_infer(run_mode):
    """
    Feature: simulated quantization algorithm.
    Description: apply simulated_quantization on resnet50.
    Expectation: apply success and structure of network as expect.
    """

    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../models/official/cv/ResNet/'))
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
    from tests.st.models.resnet import resnet50

    mindspore.context.set_context(mode=run_mode, device_target="GPU")

    network = resnet50(10)
    new_network = PrunerKfCompressAlgo({}).apply(network)
    new_network = PrunerFtCompressAlgo({'prune_rate': 0.4}).apply(new_network)
    inputs = mindspore.Tensor(np.ones([3, 3, 224, 224]), mindspore.float32)
    mindspore.export(new_network, inputs, file_name="ResNet_SCOP", file_format='MINDIR')
    #load mindir
    graph = mindspore.load('./ResNet_SCOP.mindir')
    net = nn.GraphCell(graph)
    res = net(inputs)
    assert isinstance(res, mindspore.Tensor)
