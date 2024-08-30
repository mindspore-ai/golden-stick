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
"""test some features of mindspore."""


import os

import pytest
import numpy as np
from mindspore import nn, Parameter, context, Tensor
from mindspore import dtype as mstype
from mindspore.ops import operations as msops
from mindspore.ops.auto_generate import QuantBatchMatmul
from mindspore.common.initializer import initializer
from mindspore.parallel import set_algo_parameters
from mindspore.communication.management import get_rank
import mindspore.communication.management as mcm


@pytest.mark.parametrize("is_row_parallel", [True, False])
def test_qbmm_biasadd_fusion_executor(is_row_parallel):
    """
    Feature: test reshape_shape_reshape pattern network in auto-parallel compiler.
    Description: build reshape_shape_reshape pattern network and try building network in parallel.
    Expectation: output shape as expect.
    """
    class Linear(nn.Cell):
        """Linear for test."""
        def __init__(self, ic, oc):
            super().__init__()
            self.ic = ic
            self.oc = oc

            weight_shape = (self.ic, self.oc)
            bias_shape = (self.oc,)
            self.weight = Parameter(initializer('ones', weight_shape, mstype.int8), name='weight')
            self.bias = Parameter(initializer('ones', bias_shape, mstype.float16), name='bias')
            self.scale = Parameter(initializer('ones', bias_shape, mstype.int64), name='scale')
            self.offset = None

            self.qbmm = QuantBatchMatmul(transpose_x1=False, transpose_x2=False, dtype=mstype.float16)
            self.bias_add = msops.Add()

        def shard(self, tp, is_row_parallel):
            if is_row_parallel:
                self.qbmm.shard(in_strategy=((1, tp), (tp, 1), (1,)))
                self.bias_add.shard(((1, 1), (1,)))
            else:
                self.qbmm.shard(in_strategy=((1, 1), (1, tp), (tp,)))
                self.bias_add.shard(((1, tp), (tp,)))

        def construct(self, x):
            x = self.qbmm(x, self.weight, self.scale, self.offset, None)
            x = self.bias_add(x, self.bias)
            return x

    os.environ['MS_INTERNAL_ENABLE_CUSTOM_KERNEL_LIST'] = 'QbmmAllReduceAdd,QbmmAdd'
    IS_INTERNAL_KERNEL = False
    if not os.environ.get('MS_ENABLE_INTERNAL_KERNELS'):
        IS_INTERNAL_KERNEL = True
        os.environ['MS_ENABLE_INTERNAL_KERNELS'] = 'on'
    IS_ASCEND_HOME_PATH = False
    if not os.environ.get('ASCEND_HOME_PATH'):
        IS_ASCEND_HOME_PATH = True
        os.environ['ASCEND_HOME_PATH'] = '/usr/local/Ascend/latest'
    save_graphs_path = "./test_qbmm_biasadd_fusion_irs"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                        jit_config={"jit_level": "O0", "infer_boost": "on"}, save_graphs=True,
                        save_graphs_path=save_graphs_path)
    mcm.init()
    context.set_auto_parallel_context(parallel_mode=context.ParallelMode.SEMI_AUTO_PARALLEL, gradients_mean=False,
                                      full_batch=True, strategy_ckpt_save_file='strategy.ckpt')
    set_algo_parameters(elementwise_op_strategy_follow=True)
    m, k, n = 2, 4, 8
    net = Linear(k, n)
    net.shard(2, is_row_parallel)
    inp = Tensor(np.ones((m, k)), dtype=mstype.int8)
    net(inp).asnumpy()
    rank_id = get_rank()
    all_files = os.listdir(os.path.join(save_graphs_path, f"rank_{rank_id}"))
    res_ok = True
    for ir_file in all_files:
        if not ir_file.startswith("trace_code_graph"):
            continue
        full_ir_path = os.path.join(save_graphs_path, f"rank_{rank_id}", ir_file)
        cmd = f"grep 'PrimFunc_Add' {full_ir_path}" + " | wc | awk -F ' ' '{print$2}'"
        with os.popen(cmd, "r") as f:
            result = f.read()
            result = result.strip().strip('\n')
            if result != '0':
                res_ok = False
                break
    os.environ.pop('MS_INTERNAL_ENABLE_CUSTOM_KERNEL_LIST')
    if IS_INTERNAL_KERNEL:
        os.environ.pop('MS_ENABLE_INTERNAL_KERNELS')
    if IS_ASCEND_HOME_PATH:
        os.environ.pop('ASCEND_HOME_PATH')
    assert res_ok


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_qbmm_biasadd_fusion():
    """
    Feature: test reshape_shape_reshape pattern network in auto-parallel compiler.
    Description: build reshape_shape_reshape pattern network and try building network in parallel.
    Expectation: output shape as expect.
    """
    return_code = os.system(
        "msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 "
        "--master_port=10926 --join=True --log_dir=./test_reshape_shape_reshape_parallel_fusion_logs "
        "pytest -s test_mindspore_features.py::test_qbmm_biasadd_fusion_executor"
    )
    if return_code != 0:
        log_file = open("./test_reshape_shape_reshape_parallel_fusion_logs/worker_0.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()

    assert return_code == 0
