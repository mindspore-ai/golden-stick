# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Quantization parameters post-processingfor MindOne in deploy phase."""

import numpy as np

from mindspore import ops as msops
from mindspore import Tensor, dtype
from mindspore.ops.operations.comm_ops import ReduceOp
from mindspore.communication.management import GlobalComm

def correction_into_bias(q_weight, input_scale, input_offset, weight_scale,
                         trans_b, compute_type, need_allreduce=False):
    """_correction_into_bias"""
    x_zp = input_offset.asnumpy()
    q_correction = -np.sum(x_zp.astype(np.int32) * q_weight.asnumpy().astype(np.int32),
                            axis=-1 if trans_b else -2).astype(np.int32)
    if need_allreduce:
        t_q_correction = Tensor(q_correction)
        t_q_correction = msops.AllReduce(op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)(t_q_correction)
        q_correction = t_q_correction.asnumpy()

    # for align precision
    deq_scale_np = (input_scale.asnumpy() * weight_scale.asnumpy()).astype(np.float64)
    q_correction = q_correction.astype(np.float64) * deq_scale_np
    q_correction_t = Tensor(q_correction, dtype=compute_type)
    deq_scale_t = input_scale.astype(np.float32) * weight_scale.astype(np.float32)
    q_correction_t = msops.round(q_correction_t / deq_scale_t).astype(dtype.int32)
    return q_correction_t
