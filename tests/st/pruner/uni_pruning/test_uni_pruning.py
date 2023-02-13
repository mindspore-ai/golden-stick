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
"""ST-Test for UniPruning algorithm."""

import os
import sys
import types
import mindspore
from mindspore import context
import pytest
from mindspore_gs.pruner.uni_pruning import UniPruner

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../models/official/cv/'))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("run_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_resnet(run_mode):
    """
    Feature: UniPruning algorithm.
    Description: Apply computational graph analyzer on resnet.
    Expectation: Apply success.
    """
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
    from models.resnet import resnet50
    mindspore.context.set_context(mode=run_mode)

    network = resnet50(10)
    config = {
        "exp_name": 'analyzer_test',
        "frequency": 1,
        "target_sparsity": 0.75,
        "pruning_step": 32,
        "filter_lower_threshold": 32,
        "input_size": [16, 3, 224, 224],
        "output_path": './',
        "prune_flag": 1,
        "rank": 0,
        "device_target": 'GPU'
    }
    algo = UniPruner(config)
    network = algo.apply(network)

    assert len(algo.graph_anaylzer.groups) == 39
    print("============== test resnet uni pruning success ==============")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("run_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_callback(run_mode):
    """
    Feature: UniPruning algorithm.
    Description: Return algorithm's callback.
    Expectation: Return not None.
    """
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
    from models.resnet import resnet50
    mindspore.context.set_context(mode=run_mode)

    network = resnet50(10)
    config = {
        "exp_name": 'callback_test',
        "frequency": 1,
        "target_sparsity": 0.75,
        "pruning_step": 32,
        "filter_lower_threshold": 32,
        "input_size": [16, 3, 224, 224],
        "output_path": './',
        "prune_flag": 1,
        "rank": 0,
        "device_target": 'GPU'
    }
    algo = UniPruner(config)
    network = algo.apply(network)
    assert algo.callbacks() is not None
    print("============== test uni pruning callback success ==============")


@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("run_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_resnet_convert(run_mode):
    """
    Feature: UniPruning algorithm.
    Description: Apply conversion.
    Expectation: Apply success.
    """
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
    from models.resnet import resnet50
    mindspore.context.set_context(mode=run_mode, device_target="GPU")

    network = resnet50(10)
    config = {
        "exp_name": 'convert_test',
        "frequency": 1,
        "target_sparsity": 0.75,
        "pruning_step": 32,
        "filter_lower_threshold": 32,
        "input_size": [16, 3, 224, 224],
        "output_path": './',
        "prune_flag": 1,
        "rank": 0,
        "device_target": 'GPU'
    }
    args = types.SimpleNamespace()
    args.epoch_size = 1
    args.save_checkpoint_path = "./"
    args.exp_name = "convert_test"
    args.device_target = "GPU"
    algo = UniPruner(config)
    network = algo.apply(network)
    for group in algo.graph_anaylzer.groups:
        start = group.ms_starts
        for layer in start.keys():
            if layer == 'conv1':
                filters = start[layer].weight.asnumpy().shape[0]
    mask = {'conv1': [0, 1, 2, 3]}
    print(mask)
    algo.convert(network)
    for group in algo.graph_anaylzer.groups:
        start = group.ms_starts
        for layer in start.keys():
            if layer == 'conv1':
                filters_pruned = start[layer].weight.asnumpy().shape[0]
    assert filters - filters_pruned == 4
    print("============== test resnet convert uni pruning success ==============")
