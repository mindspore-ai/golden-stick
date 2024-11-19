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
"""Parallel min max op."""

from mindspore import ops as msops
from mindspore import nn
from mindformers.experimental.parallel_core.pynative.parallel_state import get_tensor_model_parallel_group


class MinFromTensorParallelRegion(nn.Cell):
    "Get argmin from tensor-parallel region"
    def __init__(self):
        super().__init__()
        self.all_reduce = msops.AllReduce(op=msops.ReduceOp.MIN, group=get_tensor_model_parallel_group())

    def construct(self, input_, axis=None, keepdims=False, *, initial=None, where=None):
        output_parallel, _ = msops.min(input_, axis, keepdims, initial=initial, where=where)
        output = self.all_reduce(output_parallel)
        return output, _


class MaxFromTensorParallelRegion(nn.Cell):
    "Get argmax from tensor-parallel region"
    def __init__(self):
        super().__init__()
        self.all_reduce = msops.AllReduce(op=msops.ReduceOp.MAX, group=get_tensor_model_parallel_group())

    def construct(self, input_, axis=None, keepdims=False, *, initial=None, where=None):
        output_parallel, _ = msops.max(input_, axis, keepdims, initial=initial, where=where)
        output = self.all_reduce(output_parallel)
        return output, _


def get_smooth_x_obs_min_max_op():
    return msops.max, msops.min


def get_smooth_w_obs_min_max_op(tensor_parallel, is_colparallel):
    if tensor_parallel and is_colparallel:
        w_obs_max = MaxFromTensorParallelRegion()
        w_obs_min = MinFromTensorParallelRegion()
    else:
        w_obs_max = msops.max
        w_obs_min = msops.min
    return w_obs_max, w_obs_min


def get_quant_min_max_op(tensor_parallel, is_rowparallel):
    if tensor_parallel and is_rowparallel:
        quant_max = MaxFromTensorParallelRegion()
        quant_min = MinFromTensorParallelRegion()
    else:
        quant_max = msops.max
        quant_min = msops.min
    return quant_max, quant_min
