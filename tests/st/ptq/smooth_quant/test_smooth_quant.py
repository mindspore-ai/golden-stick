# Copyright 2024 Huawei Technologies Co., Ltd
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
"""test interfaces of smooth quant."""
import os
from collections import OrderedDict
import pytest
import mindspore.numpy as np
import mindspore.ops as P
import mindspore.communication.management as D
from mindspore import (
    Tensor, context, save_checkpoint, load_checkpoint, load_param_into_net,
    load_distributed_checkpoint, Model
)
from mindspore import nn
from mindspore.common.dtype import QuantDtype
from mindspore.context import ParallelMode
from mindspore.parallel import set_algo_parameters
from mindformers.modules import Linear

from mindspore_gs.ptq import SmoothQuant, PTQConfig
from mindspore_gs.ptq.quant_cells import SQLinearWrapper
from mindspore_gs.ptq.fake_quantizer import MinMaxPerLayer, MinMaxPerChannel


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_constructor():
    """
    Feature: smooth quant algorithm.
    Description: Call constructor of smooth quant and check config.
    Expectation: smooth_quant related is updated according to argument `config` of constructor.
    """
    sq = SmoothQuant()
    assert sq._config.approach == 'smooth_quant'
    assert sq._config.algo_args.get('alpha', None) == 0.5
    assert sq._config.algo_args.get('is_real_foward', None) == False


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_param():
    """
    Feature: set param for smooth quant
    Description: Call set param for smooth quant
    Expectation: config is updated according to input arguments.
    """
    sq = SmoothQuant()
    sq.set_param('alpha', 0.7)
    sq.set_param('act_per_channel', True)
    assert sq._config.algo_args.get('alpha', None) == 0.5
    assert sq._config.act_per_channel == True


class SimpleNet(nn.Cell):
    """
    Network with single linear to be quant
    """

    def __init__(self,
                 in_channels=5,
                 out_channels=6,
                 transpose_b=True,
                 strategy=None):
        super().__init__()
        self.linear = Linear(in_channels=in_channels,
                             out_channels=out_channels,
                             transpose_b=transpose_b)
        if strategy is not None:
            self.linear.shard(strategy)

    def construct(self, x):
        return self.linear(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_apply():
    """
    Feature: RoundToNearestPTQ algorithm set functions.
    Description: Apply RoundToNearestPTQ on SimpleNet.
    Expectation: Apply success and coordinate attributes are same as config.
    """

    network = SimpleNet()
    ptq = SmoothQuant()
    new_network = ptq.apply(network)
    cells: OrderedDict = new_network.name_cells()

    quant_cell = cells.get("linear", None)
    assert isinstance(quant_cell, SQLinearWrapper)
    weight_fake_quant = quant_cell.weight_quantizer()
    assert isinstance(weight_fake_quant, MinMaxPerChannel)
    assert weight_fake_quant.symmetric()
    assert weight_fake_quant.quant_dtype() == QuantDtype.INT8
    assert weight_fake_quant.is_per_channel()
    assert not weight_fake_quant.narrow_range()
    assert weight_fake_quant.num_bits() == 8

    act_fake_quant = quant_cell.input_quantizer()
    assert isinstance(act_fake_quant, MinMaxPerLayer)
    assert act_fake_quant.symmetric() == False
    assert act_fake_quant.quant_dtype() == QuantDtype.INT8
    assert not act_fake_quant.is_per_channel()
    assert not act_fake_quant.narrow_range()
    assert act_fake_quant.num_bits() == 8

    assert quant_cell.output_quantizer() is None


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single # 2p
def test_smooth_quant_adjust_error_2stages_2p():
    """
    Feature: test smooth quant adjust parameter in two stages
    in parallel mode using 2 cards
    Description: Feed invalid type of bn_fold to convert function.
    Expectation: adjust error is in certain range.
    """
    context.reset_auto_parallel_context()
    context.set_context(device_target='Ascend',
                        mode=context.GRAPH_MODE)
    D.init()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      gradients_mean=False)
    set_algo_parameters(elementwise_op_strategy_follow=True)
    act_in, act_out = 8, 8
    cfg = PTQConfig(approach='smooth_quant', model_parallel=2)
    strategy = ((1, 2), (1, 2))
    network = SimpleNet(in_channels=act_in,
                        out_channels=act_out,
                        strategy=strategy)
    ptq = SmoothQuant(config=cfg)
    new_network = ptq.apply(network)

    def _calibrate(net, calibrate_size):
        for _ in range(calibrate_size):
            example = Tensor(np.rand(act_in, act_out))
            _ = net(example)

    _calibrate(new_network, 2)
    example = Tensor(np.rand(act_in, act_out))
    orin_out = new_network(example)
    opt_network = ptq.convert(new_network)
    rank_id = os.getenv('RANK_ID')
    ckpt_path = f'opt_network{rank_id}.ckpt'
    save_checkpoint(opt_network, ckpt_path, integrated_save=False)

    # second stage
    strategy_filename = 'simplenet_strategy.ckpt'
    context.set_auto_parallel_context(strategy_ckpt_config={'save_file': strategy_filename})
    ptq._config.algo_args['is_real_forward'] = True
    apply_network = SimpleNet(in_channels=act_in,
                              out_channels=act_out,
                              strategy=strategy)
    deploy_net = ptq.apply(apply_network)
    opt_deploy_net = ptq.convert(deploy_net)
    model = Model(opt_deploy_net)
    predict_layout = model.infer_predict_layout(example)
    ckpt_file_list = [f'opt_network{i}.ckpt' for i in range(2)]
    load_distributed_checkpoint(opt_deploy_net,
                                ckpt_file_list,
                                predict_layout,
                                train_strategy_filename=strategy_filename)
    adjust_out = model.predict(example)
    diff = P.Abs()(orin_out - adjust_out)
    relative_diff = diff / (P.Abs()(orin_out) + 1e-5)
    assert diff[relative_diff > 1e-4].max() < 5e-2

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_smooth_quant_adjust_error_2stages_1p():
    """
    Feature: test smooth quant adjust parameter in two stages using 1 card
    Description: Feed invalid type of bn_fold to convert function.
    Expectation: adjust error is in certain range.
    """
    context.set_context(device_target='Ascend',
                        mode=context.GRAPH_MODE)
    network = SimpleNet()
    ptq_calib = SmoothQuant()
    new_network = ptq_calib.apply(network)

    act_in, act_out = 6, 5
    def _calibrate(net, calibrate_size):
        for _ in range(calibrate_size):
            example = Tensor(np.rand(act_in, act_out))
            _ = net(example)

    _calibrate(new_network, 2)
    opt_network = ptq_calib.convert(new_network)
    ckpt_path = 'opt_network.ckpt'
    save_checkpoint(opt_network, ckpt_path)

    cfg_deploy = PTQConfig(approach='SmoothQuant')
    cfg_deploy.algo_args['is_real_forward'] = True
    ptq_deploy = SmoothQuant(config=cfg_deploy)
    deploy_net = ptq_deploy.apply(network)
    opt_deploy_net = ptq_deploy.convert(deploy_net)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(opt_deploy_net, param_dict)
    example = Tensor(np.rand(act_in, act_out))
    orin_out = new_network(example)
    adjust_out = opt_deploy_net(example)
    diff = P.Abs()(orin_out - adjust_out)
    relative_diff = diff / (P.Abs()(orin_out) + 1e-5)
    assert diff[relative_diff > 1e-4].max() < 5e-2

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_smooth_quant_adjust_error_1p():
    """
    Feature: simulated quantization convert function.
    Description: Feed invalid type of bn_fold to convert function.
    Expectation: Except TypeError.
    """
    context.set_context(device_target='Ascend',
                        mode=context.GRAPH_MODE)
    network = SimpleNet()
    ptq = SmoothQuant()
    new_network = ptq.apply(network)

    act_in, act_out = 6, 5
    def _calibrate(net, calibrate_size):
        for _ in range(calibrate_size):
            example = Tensor(np.rand(act_in, act_out))
            _ = net(example)

    _calibrate(new_network, 2)
    opt_network = ptq.convert(new_network)
    example = Tensor(np.rand(act_in, act_out))
    orin_out = new_network(example)
    adjust_out = opt_network(example)
    diff = P.Abs()(orin_out - adjust_out)
    relative_diff = diff / (P.Abs()(orin_out) + 1e-5)
    assert diff[relative_diff > 1e-4].max() < 5e-2
