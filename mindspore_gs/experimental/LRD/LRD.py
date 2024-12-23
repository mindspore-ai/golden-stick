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
"""LRD algorithm."""
from typing import Tuple
import time
import mindspore as ms
from mindspore import Parameter
from mindspore import ops as msops
from mindspore.nn import Cell
from mindspore.common.initializer import initializer
from mindspore import Tensor
from mindspore.ops import Svd
import mindspore.nn as nn
from mindspore_gs.common import logger
from mindspore_gs.common.utils import value_check
from mindspore_gs.ptq.processor import Processor
from mindformers.experimental.infer.core import RowParallelLinear, ColumnParallelLinear


class DoubleMM(nn.Cell):
    """DoubleMM"""
    def __init__(self, w1, w2, transpose_b=True):
        super(DoubleMM, self).__init__()
        self.matmul = msops.MatMul(False, False)
        if transpose_b:
            self.weights_1 = Parameter(Tensor(w2.T))
            self.weights_2 = Parameter(Tensor(w1.T))
        else:
            self.weights_1 = Parameter(Tensor(w1))
            self.weights_2 = Parameter(Tensor(w2))

    # pylint: disable=unused-argument
    def construct(self, x, dummy_weight):
        """construct"""
        out = self.matmul(x, self.weights_1)
        out = self.matmul(out, self.weights_2)
        return out


class LRD:
    """LRD"""
    def __init__(self, is_deploy):
        self.is_deploy = is_deploy

    def apply(self, network: Cell) -> Cell:
        """apply"""
        selected_layer_indices = {21, 22, 23, 24, 25, 26, 27, 28, 29}

        class LinearWeightProcessor(Processor):
            """LinearWeightProcessor"""
            def __init__(self, is_deploy: bool):
                self.is_deploy = is_deploy

            @staticmethod
            def apply_low_rank(weight: Tensor, is_deploy):
                """_apply_low_rank"""
                k = 2
                if is_deploy:
                    m, n = weight.shape
                    w1 = Parameter(initializer('ones', (m, k), weight.dtype))
                    w2 = Parameter(initializer('ones', (k, n), weight.dtype))
                    return w1, w2
                logger.info(f"Start LRD on weight.")
                svd_op = Svd(full_matrices=True, compute_uv=True)
                w = ms.ops.cast(weight, ms.float32)
                start_time = time.time()
                s, u, vh = svd_op(w)
                end_time = time.time()
                logger.info(f"SVD process time for split: {end_time - start_time}")

                start_time_n = time.time()
                s_k = ms.numpy.diag(s[:k])
                end_time_n = time.time()
                logger.info(f"Diag process time for split : {end_time_n - start_time_n}")
                u_k = u[:, :k]
                vh_k = vh[:k, :]
                logger.info(f"u_k.shape : {u_k.shape}")
                weights_1 = Parameter(Tensor(u_k))
                weights_2 = Parameter(Tensor(ms.numpy.matmul(s_k, vh_k)))
                return weights_1, weights_2

            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                """process_cell"""
                if not isinstance(cell, (RowParallelLinear, ColumnParallelLinear)):
                    return cell, False
                if not any(str(layer) in cell_name for layer in selected_layer_indices):
                    return cell, False
                logger.info(f"NMF linear: {cell_name}")
                w1, w2 = LinearWeightProcessor.apply_low_rank(cell.weight, self.is_deploy)
                cell.matmul = DoubleMM(w1, w2)
                logger.info(f"NMF linear from {cell.weight.shape} to {w1.shape} x {w2.shape}.")
                dummy_weight = Parameter(Tensor(ms.numpy.zeros((16, 16), dtype=cell.weight.dtype)))
                cell.weight = dummy_weight
                return cell, True

        value_check('network', network, Cell)
        processor = LinearWeightProcessor(self.is_deploy)
        processor.process(network)
        network.update_parameters_name()
        return network
