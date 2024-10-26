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
"""functional unit cells."""
import abc
import dataclasses
import enum

import numpy as np
from mindspore import Tensor, dtype
from mindspore.nn import Cell
from mindspore import Parameter
from mindspore import ops as msops
from mindspore.common.initializer import initializer
from mindspore.ops.auto_generate import WeightQuantBatchMatmul, QuantBatchMatmul, DynamicQuantExt
from mindspore_gs.common.numpy_quant_common import NumpyQuantOps


@dataclasses.dataclass
class QuantParam:
    scale: Parameter = None
    zero_point: Parameter = None


class ParallelType(enum.Enum):
    NO_PARALLEL = 0
    COL_PARALLEL = 1
    ROW_PARALLEL = 2


class QuantUnitCell(abc.ABC, Cell):

    @abc.abstractmethod
    def param_shard_state(self, tensor_parallel_num=1, **kwargs) -> dict:
        raise NotImplementedError


class MatmulCellForHook(QuantUnitCell):
    """MatmulCellForHook"""
    def __init__(self, matmul):
        super().__init__()
        self.mm = matmul

    def construct(self, *args, **kwargs):
        return self.mm(*args, **kwargs)

    def param_shard_state(self, tensor_parallel_num=1, **kwargs):
        return {}


class SmoothMatmul(QuantUnitCell):
    """SmoothMatmul"""
    def __init__(self, mm, smooth_scale_):
        super().__init__()
        self.mm = mm
        self.smooth_scale = Parameter(msops.div(1, smooth_scale_))

    @classmethod
    def _from_matmul_cell(cls, src: MatmulCellForHook, smooth_scale):
        if not isinstance(src.mm, msops.MatMul):
            raise ValueError(
                f'matmul of MatmulCellForHook should be an instance of {msops.MatMul}, but got {src.mm}.')
        return cls(src.mm, smooth_scale)

    @staticmethod
    def create(src, smooth_scale):
        if isinstance(src, MatmulCellForHook):
            return SmoothMatmul._from_matmul_cell(src, smooth_scale)
        raise ValueError(f"Not support creating SmoothMatmul from {src}.")

    def construct(self, x, weight):
        x = msops.mul(x, self.smooth_scale)
        return self.mm(x, weight)

    # pylint: disable=arguments-differ
    def param_shard_state(self, tensor_parallel_num=1, parallel_type: ParallelType = ParallelType.NO_PARALLEL):
        if parallel_type == ParallelType.COL_PARALLEL:
            smooth_scale_shard = (1,)
        elif parallel_type == ParallelType.ROW_PARALLEL:
            smooth_scale_shard = (tensor_parallel_num,)
        else:
            return {}
        return {self.smooth_scale.name: {'shape': self.smooth_scale.shape, 'shard': smooth_scale_shard}}


class SmoothMatmulForDeploy(QuantUnitCell):
    """SmoothMatmulForDeploy"""
    def __init__(self, mm, ic_, compute_dtype_):
        super().__init__()
        self.mm = mm
        self.smooth_scale = Parameter(initializer('ones', (ic_,), dtype=compute_dtype_))

    @staticmethod
    def _from_matmul_prim(src: msops.MatMul, ic, compute_dtype):
        return SmoothMatmulForDeploy(src, ic, compute_dtype)

    @staticmethod
    def _from_matmul_cell(src: MatmulCellForHook, ic, compute_dtype):
        if not isinstance(src.mm, msops.MatMul):
            raise ValueError(
                f'matmul of MatmulCellForHook should be an instance of {msops.MatMul}, but got {src.mm}.')
        return SmoothMatmulForDeploy(src.mm, ic, compute_dtype)

    @staticmethod
    def create(src, ic, compute_dtype):
        if isinstance(src, msops.MatMul):
            return SmoothMatmulForDeploy._from_matmul_prim(src, ic, compute_dtype)
        if isinstance(src, MatmulCellForHook):
            return SmoothMatmulForDeploy._from_matmul_cell(src, ic, compute_dtype)
        raise ValueError(f"Not support creating SmoothMatmulForDeploy from {src}.")

    def construct(self, x, weight):
        x = msops.mul(x, self.smooth_scale)
        return self.mm(x, weight)

    # pylint: disable=arguments-differ
    def param_shard_state(self, tensor_parallel_num=1, parallel_type: ParallelType = ParallelType.NO_PARALLEL):
        if parallel_type == ParallelType.COL_PARALLEL:
            smooth_scale_shard = (1,)
        elif parallel_type == ParallelType.ROW_PARALLEL:
            smooth_scale_shard = (tensor_parallel_num,)
        else:
            return {}
        return {self.smooth_scale.name: {'shape': self.smooth_scale.shape, 'shard': smooth_scale_shard}}


class DynamicQuantMatmul(QuantUnitCell):
    """dynamic quant"""

    def __init__(self, is_deploy, weight_scale, transpose_a=False, transpose_b=False, dst_dtype=dtype.float16,
                 smooth_scale=None):
        super().__init__()
        if is_deploy:
            self.weight_scale = Parameter(initializer("ones", weight_scale.shape, dtype.float32))
        else:
            self.weight_scale = Parameter(weight_scale.astype(dtype.float32))
        self.smooth_scale = smooth_scale
        self.dynamic_quant = DynamicQuantExt()
        self.qbmm = QuantBatchMatmul(transpose_x1=transpose_a, transpose_x2=transpose_b, dtype=dst_dtype)

    @staticmethod
    def _from_matmul_prim(is_deploy, w_qparam: QuantParam, transpose_a=False, transpose_b=False,
                          dst_dtype=dtype.float16):
        return DynamicQuantMatmul(is_deploy, w_qparam.scale, transpose_a, transpose_b, dst_dtype, None)

    @staticmethod
    def _from_matmul_cell(is_deploy, src: MatmulCellForHook, w_qparam: QuantParam, transpose_a=False, transpose_b=False,
                          dst_dtype=dtype.float16):
        if not isinstance(src.mm, msops.MatMul):
            raise ValueError(
                f'matmul of MatmulCellForHook should be an instance of {msops.MatMul}, but got {src.mm}.')
        return DynamicQuantMatmul(is_deploy, w_qparam.scale, transpose_a, transpose_b, dst_dtype, None)

    @staticmethod
    def _from_smooth_matmul(is_deploy, src: SmoothMatmul, w_qparam: QuantParam, transpose_a=False, transpose_b=False,
                            dst_dtype=dtype.float16):
        if not isinstance(src.mm, msops.MatMul):
            raise ValueError(
                f'matmul of SmoothMatmul should be an instance of {msops.MatMul}, but got {src.mm}.')
        return DynamicQuantMatmul(is_deploy, w_qparam.scale, transpose_a, transpose_b, dst_dtype, src.smooth_scale)

    @staticmethod
    def _from_smooth_matmul_for_deploy(is_deploy, src: SmoothMatmulForDeploy, w_qparam: QuantParam, transpose_a=False,
                                       transpose_b=False, dst_dtype=dtype.float16):
        if not isinstance(src.mm, msops.MatMul):
            raise ValueError(
                f'matmul of SmoothMatmulForDeploy should be an instance of {msops.MatMul}, but got {src.mm}.')
        return DynamicQuantMatmul(is_deploy, w_qparam.scale, transpose_a, transpose_b, dst_dtype, src.smooth_scale)

    @staticmethod
    def create(src, w_qparam: QuantParam, is_deploy, transpose_a=False, transpose_b=False, dst_dtype=dtype.float16):
        """create"""
        if isinstance(src, msops.MatMul):
            return DynamicQuantMatmul._from_matmul_prim(is_deploy, w_qparam, transpose_a, transpose_b, dst_dtype)
        if isinstance(src, MatmulCellForHook):
            return DynamicQuantMatmul._from_matmul_cell(is_deploy, src, w_qparam, transpose_a, transpose_b, dst_dtype)
        if isinstance(src, SmoothMatmul):
            return DynamicQuantMatmul._from_smooth_matmul(is_deploy, src, w_qparam, transpose_a, transpose_b, dst_dtype)
        if isinstance(src, SmoothMatmulForDeploy):
            return DynamicQuantMatmul._from_smooth_matmul_for_deploy(is_deploy, src, w_qparam, transpose_a, transpose_b,
                                                                     dst_dtype)
        raise ValueError(f"Not support creating DynamicQuantMatmul from {src}.")

    def construct(self, x, quant_weight):
        qx, x_scale = self.dynamic_quant(x, self.smooth_scale)
        return self.qbmm(qx, quant_weight, self.weight_scale, None, None, x_scale)

    # pylint: disable=arguments-differ
    def param_shard_state(self, tensor_parallel_num=1, parallel_type: ParallelType = ParallelType.NO_PARALLEL):
        if parallel_type == ParallelType.COL_PARALLEL:
            smooth_scale_shard = (1,)
            weight_scale_shard = (tensor_parallel_num,)
        elif parallel_type == ParallelType.ROW_PARALLEL:
            smooth_scale_shard = (tensor_parallel_num,)
            weight_scale_shard = (1,)
        else:
            return {}
        shard_state = {self.weight_scale.name: {'shape': self.weight_scale.shape, 'shard': weight_scale_shard}}
        if self.smooth_scale:
            shard_state[self.smooth_scale.name] = {'shape': self.smooth_scale.shape, 'shard': smooth_scale_shard}
        return shard_state


class WeightQuantMatmul(QuantUnitCell):
    """quant batch matmul"""

    def __init__(self, is_deploy, weight_scale: Parameter, weight_zp: Parameter, transpose_a=False, transpose_b=False,
                 dst_type=dtype.float16, smooth_scale=None):
        super().__init__()
        self.dst_dtype = dst_type
        if is_deploy:
            self.t_scale = Parameter(initializer('ones', weight_scale.shape, self.dst_dtype), name="t_scale")
            self.t_zp_neg = Parameter(initializer('zeros', weight_zp.shape, self.dst_dtype), name="t_zp_neg")
        else:
            self.t_scale = Parameter(Tensor(weight_scale.asnumpy(), dtype=self.dst_dtype), name="t_scale")
            self.t_zp_neg = Parameter(Tensor(weight_zp.asnumpy() * -1, dtype=self.dst_dtype), name="t_zp_neg")
        self.weight_qbmm = WeightQuantBatchMatmul(transpose_a, transpose_b)
        self.has_smooth = smooth_scale is not None
        self.smooth_scale = smooth_scale

    @staticmethod
    def _from_matmul_prim(w_qparam: QuantParam, is_deploy, transpose_a=False, transpose_b=False,
                          dst_dtype=dtype.float16):
        return WeightQuantMatmul(is_deploy, w_qparam.scale, w_qparam.zero_point, transpose_a, transpose_b,
                                 dst_dtype, None)

    @staticmethod
    def _from_matmul_cell(src: MatmulCellForHook, w_qparam: QuantParam, is_deploy,
                          transpose_a=False, transpose_b=False, dst_dtype=dtype.float16):
        if not isinstance(src.mm, msops.MatMul):
            raise ValueError(
                f'matmul of MatmulCellForHook should be an instance of {msops.MatMul}, but got {src.mm}.')
        return WeightQuantMatmul(is_deploy, w_qparam.scale, w_qparam.zero_point, transpose_a, transpose_b,
                                 dst_dtype, None)

    @staticmethod
    def _from_smooth_matmul(src: SmoothMatmul, w_qparam: QuantParam, is_deploy,
                            transpose_a=False, transpose_b=False, dst_dtype=dtype.float16):
        if not isinstance(src.mm, msops.MatMul):
            raise ValueError(
                f'matmul of SmoothMatmul should be an instance of {msops.MatMul}, but got {src.mm}.')
        return WeightQuantMatmul(is_deploy, w_qparam.scale, w_qparam.zero_point, transpose_a, transpose_b,
                                 dst_dtype, src.smooth_scale)

    @staticmethod
    def _from_smooth_matmul_for_deploy(src: SmoothMatmulForDeploy, w_qparam: QuantParam, is_deploy,
                                       transpose_a=False, transpose_b=False, dst_dtype=dtype.float16):
        if not isinstance(src.mm, msops.MatMul):
            raise ValueError(
                f'matmul of SmoothMatmulForDeploy should be an instance of {msops.MatMul}, but got {src.mm}.')
        return WeightQuantMatmul(is_deploy, w_qparam.scale, w_qparam.zero_point, transpose_a, transpose_b,
                                 dst_dtype, src.smooth_scale)

    @staticmethod
    def create(src, w_qparam: QuantParam, is_deploy, transpose_a=False, transpose_b=False, dst_dtype=dtype.float16):
        """create"""
        if isinstance(src, msops.MatMul):
            return WeightQuantMatmul._from_matmul_prim(w_qparam, is_deploy, transpose_a, transpose_b, dst_dtype)
        if isinstance(src, MatmulCellForHook):
            return WeightQuantMatmul._from_matmul_cell(src, w_qparam, is_deploy, transpose_a, transpose_b, dst_dtype)
        if isinstance(src, SmoothMatmul):
            return WeightQuantMatmul._from_smooth_matmul(src, w_qparam, is_deploy, transpose_a, transpose_b, dst_dtype)
        if isinstance(src, SmoothMatmulForDeploy):
            return WeightQuantMatmul._from_smooth_matmul_for_deploy(src, w_qparam, is_deploy, transpose_a, transpose_b,
                                                                    dst_dtype)
        raise ValueError(f"Not support creating WeightQuantMatmul from {src}.")

    def construct(self, x, weight):
        """forward for WeightQuantMatmul cell"""
        if self.has_smooth:
            x = msops.mul(x, self.smooth_scale)
        output = self.weight_qbmm(x, weight, self.t_scale, self.t_zp_neg, None, None, None)
        return output.astype(self.dst_dtype)

    # pylint: disable=arguments-differ
    def param_shard_state(self, tensor_parallel_num=1, parallel_type: ParallelType = ParallelType.NO_PARALLEL):
        if parallel_type == ParallelType.COL_PARALLEL:
            smooth_scale_shard = (1,)
            t_scale_shard = (self.layer.tensor_parallel_group_size,)
            t_zp_shard = {self.layer.tensor_parallel_group_size}
        elif parallel_type == ParallelType.ROW_PARALLEL:
            smooth_scale_shard = (tensor_parallel_num,)
            t_scale_shard = (1,)
            t_zp_shard = (1,)
        else:
            return {}
        shard_state = {
            self.t_scale.name: {'shape': self.t_scale.shape, 'shard': t_scale_shard},
            self.t_zp.name: {'shape': self.t_zp.shape, 'shard': t_zp_shard},
        }
        if self.smooth_scale:
            shard_state[self.smooth_scale.name] = {'shape': self.smooth_scale.shape, 'shard': smooth_scale_shard}
        return shard_state


class AllQuantMatmul(QuantUnitCell):
    """all quant mm"""

    def __init__(self, is_deploy, input_scale: Parameter, weight_scale: Parameter, offset=None,
                 transpose_a=False, transpose_b=False, dst_dtype=dtype.float16):
        super().__init__()
        self.transpose_b = transpose_b
        self.offset = offset
        if is_deploy:
            self.dequant_scale = Parameter(initializer('ones', weight_scale.shape, dtype.int64))
        else:
            self.dequant_scale = Parameter(Tensor(AllQuantMatmul._compute_dequant_scale(input_scale.asnumpy(),
                                                                                        weight_scale.asnumpy()),
                                                  dtype=dtype.int64))
        self.qbmm = QuantBatchMatmul(transpose_x1=transpose_a, transpose_x2=transpose_b, dtype=dst_dtype)
        if offset is None:
            self.offset = None
        else:
            self.offset = Parameter(Tensor(offset, dtype=dtype.float32))

    @staticmethod
    def _from_matmul_prim(x_qparam: QuantParam, w_qparam: QuantParam, is_deploy, transpose_a=False, transpose_b=False,
                          dst_dtype=dtype.float16):
        return AllQuantMatmul(is_deploy, x_qparam.scale, w_qparam.scale, w_qparam.zero_point, transpose_a, transpose_b,
                              dst_dtype), None

    @staticmethod
    def _from_matmul_cell(src: MatmulCellForHook, x_qparam: QuantParam, w_qparam: QuantParam, is_deploy,
                          transpose_a=False, transpose_b=False, dst_dtype=dtype.float16):
        if not isinstance(src.mm, msops.MatMul):
            raise ValueError(
                f'matmul of MatmulCellForHook should be an instance of {msops.MatMul}, but got {src.mm}.')
        return AllQuantMatmul(is_deploy, x_qparam.scale, w_qparam.scale, w_qparam.zero_point, transpose_a, transpose_b,
                              dst_dtype), None

    @staticmethod
    def _from_smooth_matmul(src: SmoothMatmul, x_qparam: QuantParam, w_qparam: QuantParam, is_deploy,
                            transpose_a=False, transpose_b=False, dst_dtype=dtype.float16):
        if not isinstance(src.mm, msops.MatMul):
            raise ValueError(
                f'matmul of SmoothMatmul should be an instance of {msops.MatMul}, but got {src.mm}.')
        return AllQuantMatmul(is_deploy, x_qparam.scale, w_qparam.scale, w_qparam.zero_point, transpose_a, transpose_b,
                              dst_dtype), src.smooth_scale.asnumpy()

    @staticmethod
    def _from_smooth_matmul_for_deploy(src: SmoothMatmulForDeploy, x_qparam: QuantParam, w_qparam: QuantParam,
                                       is_deploy, transpose_a=False, transpose_b=False, dst_dtype=dtype.float16):
        if not isinstance(src.mm, msops.MatMul):
            raise ValueError(
                f'matmul of SmoothMatmul should be an instance of {msops.MatMul}, but got {src.mm}.')
        return AllQuantMatmul(is_deploy, x_qparam.scale, w_qparam.scale, w_qparam.zero_point, transpose_a, transpose_b,
                              dst_dtype), src.smooth_scale.asnumpy()

    @staticmethod
    def create(src, x_qparam: QuantParam, w_qparam: QuantParam, is_deploy, transpose_a=False, transpose_b=False,
               dst_dtype=dtype.float16):
        """create"""
        if isinstance(src, msops.MatMul):
            return AllQuantMatmul._from_matmul_prim(x_qparam, w_qparam, is_deploy, transpose_a, transpose_b, dst_dtype)
        if isinstance(src, MatmulCellForHook):
            return AllQuantMatmul._from_matmul_cell(src, x_qparam, w_qparam, is_deploy, transpose_a, transpose_b,
                                                    dst_dtype)
        if isinstance(src, SmoothMatmul):
            return AllQuantMatmul._from_smooth_matmul(src, x_qparam, w_qparam, is_deploy, transpose_a, transpose_b,
                                                      dst_dtype)
        if isinstance(src, SmoothMatmulForDeploy):
            return AllQuantMatmul._from_smooth_matmul_for_deploy(src, x_qparam, w_qparam, is_deploy, transpose_a,
                                                                 transpose_b, dst_dtype)
        raise ValueError(f"Not support creating AllQuantMatmul from {src}.")

    @staticmethod
    def _compute_dequant_scale(input_scale, weight_scale):
        """compute_dequant_scale"""
        dequant_scale = input_scale.astype(np.float32) * weight_scale.astype(np.float32)
        scale_i64 = NumpyQuantOps.trans_fp32_to_i64(dequant_scale)
        return scale_i64

    def construct(self, qx, quant_weight):
        # x: fp16 quant_weight: int8
        return self.qbmm(qx, quant_weight, self.dequant_scale, self.offset, None)

    # pylint: disable=arguments-differ
    def param_shard_state(self, tensor_parallel_num=1, parallel_type: ParallelType = ParallelType.NO_PARALLEL):
        if parallel_type == ParallelType.COL_PARALLEL:
            q_shard = (self.layer.tensor_parallel_group_size,)
        elif parallel_type == ParallelType.ROW_PARALLEL:
            q_shard = (1,)
        else:
            return {}
        shard_state = {self.dequant_scale.name: {'shape': self.dequant_scale.shape, 'shard': q_shard}}
        if self.offset:
            shard_state[self.offset.name] = {'shape': self.offset.shape, 'shard': q_shard}
        return shard_state
