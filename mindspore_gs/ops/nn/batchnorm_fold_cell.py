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
"""BatchNormFoldCell."""
from __future__ import absolute_import

from mindspore.ops import operations as P
import mindspore.context as context
from mindspore.nn.cell import Cell
from mindspore.ops.operations import _quant_ops as Q


class BatchNormFoldCell(Cell):
    """
    Batch Normalization folded.

    Args:
        momentum (float): Momentum value must be [0, 1]. Default: 0.9.
        epsilon (float): A small float number to avoid dividing by 0. 1e-5 if dtype in
            float32 else 1e-3. Default: 1e-5.
        freeze_bn (int): Delay in steps at which computation switches from regular batch
            norm to frozen mean and std. Default: 0.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C, H, W)`.
        - **mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **variance** (Tensor) - Tensor of shape :math:`(C,)`.
        - **global_step** (Tensor) - Tensor to record current global step.

    Outputs:
        Tuple of 4 Tensor, the normalized input and the updated parameters.

        - **batch_mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **batch_std** (Tensor) - Tensor of shape :math:`(C,)`.
        - **running_mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **running_std** (Tensor) - Tensor of shape :math:`(C,)`.
    """

    def __init__(self, momentum=0.9, epsilon=1e-5, freeze_bn=0):
        """Initialize batch norm fold layer"""
        super(BatchNormFoldCell, self).__init__()
        self.epsilon = epsilon
        self.is_gpu = context.get_context('device_target') == "GPU"
        if self.is_gpu:
            self.bn_train = Q.BatchNormFold(momentum, epsilon, is_training=True, freeze_bn=freeze_bn)
            self.bn_infer = Q.BatchNormFold(momentum, epsilon, is_training=False, freeze_bn=freeze_bn)
        else:
            self.bn_reduce = P.BNTrainingReduce()
            self.bn_update = Q.BatchNormFoldD(momentum, epsilon, is_training=True, freeze_bn=freeze_bn)

    def construct(self, x, mean, variance, global_step):
        """construct."""
        if self.is_gpu:
            if self.training:
                batch_mean, batch_std, running_mean, running_std = self.bn_train(x, mean, variance, global_step)
            else:
                batch_mean, batch_std, running_mean, running_std = self.bn_infer(x, mean, variance, global_step)
        else:
            if self.training:
                x_sum, x_square_sum = self.bn_reduce(x)
                _, batch_mean, batch_std, running_mean, running_std, mean_updated, variance_updated = \
                    self.bn_update(x, x_sum, x_square_sum, mean, variance)
                P.Assign()(mean, mean_updated)
                P.Assign()(variance, variance_updated)
            else:
                batch_mean = P.ZerosLike()(variance)
                batch_std = P.OnesLike()(variance)
                running_mean = P.Add()(mean, 0.)
                running_std = P.Sqrt()(P.Add()(variance, self.epsilon))
        return batch_mean, batch_std, running_mean, running_std
