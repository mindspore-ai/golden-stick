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
from typing import Optional

import numpy as np
from mindspore import Tensor, dtype
from mindspore.nn import Cell
from mindspore import Parameter
from mindspore import ops as msops
from mindspore.ops.operations._infer_ops import QuantV2
from mindspore.common.initializer import initializer
from mindspore.ops.operations.comm_ops import ReduceOp
from mindspore.communication.management import GlobalComm
from mindspore.ops.auto_generate import WeightQuantBatchMatmul, QuantBatchMatmul, DynamicQuantExt
from mindspore_gs.common.numpy_quant_common import NumpyQuantOps
from mindspore_gs.common import logger
from mindspore_gs.ptq import PTQMode
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.convert_utils import AntiQuantCell
from mindspore_gs.ptq.basic_quant_func import np_int4data_pack_to_int8, convert_fp32_to_int64


class KernelType(enum.Enum):
    ASD = 0
    ACLNN = 1
    INTERNAL = 2
    HIGH_PRECISION = 100


@dataclasses.dataclass
class QuantParam:
    """QuantParam"""
    scale: Parameter = None
    zero_point: Parameter = None
    group_size: int = 0
    quant_dtype: dtype = None

    def __str__(self):
        return f"QuantParam{{scale:{{{self.scale.shape}, {self.scale.dtype}, {self.scale.asnumpy()}}}, " \
               f"zero_point:{{{self.zero_point.shape}, {self.zero_point.dtype}, {self.zero_point.asnumpy()}}}}}, " \
               f"group_size: {self.group_size}, quant_dtype: {self.quant_dtype}"

    def __repr__(self):
        return self.__str__()


class ParallelType(enum.Enum):
    """ParallelType"""
    NO_PARALLEL = 0
    COL_PARALLEL = 1
    ROW_PARALLEL = 2


class QuantUnitCell(abc.ABC, Cell):
    """QuantUnitCell"""
    def __init__(self, layer_name):
        super().__init__()
        self.layer_name = layer_name

    @abc.abstractmethod
    def param_shard_state(self, tensor_parallel_num=1, **kwargs) -> dict:
        """param_shard_state"""
        raise NotImplementedError


class QuantWithSmooth(QuantUnitCell):
    """QuantWithSmooth"""
    def __init__(self, layer_name, parallel_type: ParallelType):
        super().__init__(layer_name)
        self.parallel_type = parallel_type
        self.input_scale = None
        self.input_zp = None

    @staticmethod
    def create(layer_name, x_qparam: QuantParam, ic, dst_dtype, is_deploy, parallel_type: ParallelType,
               smooth_scale: Optional[np.ndarray] = None, kernel_type: KernelType = KernelType.ASD):
        """create"""
        if kernel_type in (KernelType.ASD, KernelType.INTERNAL):
            return QuantWithSmoothHighPerformance(layer_name, x_qparam, ic, dst_dtype, is_deploy, parallel_type,
                                                  smooth_scale, dtype.int8)
        if kernel_type is KernelType.ACLNN:
            return QuantWithSmoothHighPerformance(layer_name, x_qparam, ic, dst_dtype, is_deploy, parallel_type,
                                                  smooth_scale, dtype.float16)
        if kernel_type is KernelType.HIGH_PRECISION:
            return QuantWithSmoothHighPrecision(layer_name, x_qparam, ic, dst_dtype, is_deploy, parallel_type,
                                                smooth_scale)
        raise RuntimeError(f"Not supported kernel type: {kernel_type}")

    def param_shard_state(self, tensor_parallel_num=1, **kwargs) -> dict:
        if self.parallel_type == ParallelType.COL_PARALLEL:
            input_scale_shard = (1,)
            input_zp_shard = (1,)
        elif self.parallel_type == ParallelType.ROW_PARALLEL:
            input_scale_shard = (tensor_parallel_num,)
            input_zp_shard = (tensor_parallel_num,)
        else:
            return {}
        return {self.input_scale.name: {'shape': self.input_scale.shape, 'shard': input_scale_shard},
                self.input_zp.name: {'shape': self.input_zp.shape, 'shard': input_zp_shard}}


class QuantWithSmoothHighPrecision(QuantWithSmooth):
    """QuantWithSmoothHighPrecision"""
    def __init__(self, layer_name, x_qparam: QuantParam, ic, dst_dtype, is_deploy, parallel_type: ParallelType,
                 smooth_scale: Optional[np.ndarray] = None):
        super().__init__(layer_name, parallel_type)
        self.dst_dtype = dst_dtype
        if is_deploy:
            self.input_scale = Parameter(initializer('ones', (ic,), dtype.float64))
            self.input_zp = Parameter(initializer('zeros', (ic,), dtype.float64))
            return
        # fuse smooth.mul and quant
        input_scale_np = x_qparam.scale.asnumpy().astype(np.float64)
        if smooth_scale is not None:
            final_scale_np = input_scale_np / smooth_scale.astype(np.float64)
            logger.debug(f"QuantWithSmoothHighPrecision: input scale with smooth scale of Layer({parallel_type}:"
                         f"{layer_name}) is {{{final_scale_np.shape}, {final_scale_np.dtype}, {final_scale_np}}}")
        else:
            if input_scale_np.shape == (1,):  # aclnn quant op not support pertensor
                final_scale_np = np.tile(input_scale_np, ic)
                logger.debug(f"QuantWithSmoothHighPrecision: input scale from vector of Layer({parallel_type}:"
                             f"{layer_name}) is {{{final_scale_np.shape}, {final_scale_np.dtype}, {final_scale_np}}}")
            else:
                final_scale_np = input_scale_np
                logger.debug(f"QuantWithSmoothHighPrecision: input scale of Layer({parallel_type}:{layer_name}) is "
                             f"{{{final_scale_np.shape}, {final_scale_np.dtype}, {final_scale_np}}}")
        self.input_scale = Parameter(Tensor(final_scale_np, dtype=dtype.float64))

        if self.input_scale.shape != x_qparam.zero_point.shape:
            if isinstance(x_qparam.zero_point, np.number):
                raise RuntimeError("Shape of scale and zero point are not compatible.")
            self.input_zp = Parameter(Tensor(np.tile(x_qparam.zero_point, ic).astype(np.float64), dtype=dtype.float64))
            logger.debug(f"QuantWithSmoothHighPrecision: input zp from vector of Layer({parallel_type}:{layer_name}) is"
                         f" {{{self.input_zp.shape}, {self.input_zp.dtype}, {self.input_zp}}}")
        else:
            self.input_zp = Parameter(Tensor(x_qparam.zero_point, dtype=dtype.float64))
            logger.debug(f"QuantWithSmoothHighPrecision: input zp of Layer({parallel_type}:{layer_name}) is "
                         f"{{{self.input_zp.shape}, {self.input_zp.dtype}, {self.input_zp.asnumpy()}}}")

    def construct(self, x):
        out = x / self.input_scale
        out = msops.round(out)
        out = msops.clip(out, -128., 127.)
        return msops.cast(out, self.dst_dtype)


class QuantWithSmoothHighPerformance(QuantWithSmooth):
    """QuantWithSmoothHighPerformance"""
    def __init__(self, layer_name, x_qparam: QuantParam, ic, dst_dtype, is_deploy, parallel_type: ParallelType,
                 smooth_scale: Optional[np.ndarray] = None, zp_dtype=dtype.int8):
        super().__init__(layer_name, parallel_type)
        self.quant = QuantV2()
        if is_deploy:
            self.input_scale = Parameter(initializer('ones', (ic,), dst_dtype))
            self.input_zp = Parameter(initializer('zeros', (ic,), zp_dtype))
            return
        # fuse smooth.mul and quant
        input_scale_np = x_qparam.scale.asnumpy().astype(np.float16)
        if smooth_scale is not None:
            final_scale_np = input_scale_np / smooth_scale.astype(np.float16)
            logger.debug(f"QuantWithSmoothHighPerformance: input scale with smooth scale of Layer({parallel_type}:"
                         f"{layer_name}) is {{{final_scale_np.shape}, {final_scale_np.dtype}, {final_scale_np}}}")
        else:
            if input_scale_np.shape == (1,):  # aclnn quant op not support pertensor
                final_scale_np = np.tile(input_scale_np, ic)
                logger.debug(f"QuantWithSmoothHighPerformance: input scale from vector of Layer({parallel_type}:"
                             f"{layer_name}) is {{{final_scale_np.shape}, {final_scale_np.dtype}, {final_scale_np}}}")
            else:
                final_scale_np = input_scale_np
                logger.debug(f"QuantWithSmoothHighPerformance: input scale of Layer({parallel_type}:{layer_name}) is "
                             f"{{{final_scale_np.shape}, {final_scale_np.dtype}, {final_scale_np}}}")
        self.input_scale = Parameter(Tensor(final_scale_np, dtype=dst_dtype))

        if self.input_scale.shape != x_qparam.zero_point.shape:
            if isinstance(x_qparam.zero_point, np.number):
                raise RuntimeError("Shape of scale and zero point are not compatible.")
            self.input_zp = Parameter(Tensor(np.tile(x_qparam.zero_point, ic).astype(np.float16), dtype=zp_dtype))
            logger.debug(f"QuantWithSmoothHighPerformance: input zp from vector of Layer({parallel_type}:{layer_name})"
                         f" is {{{self.input_zp.shape}, {self.input_zp.dtype}, {self.input_zp}}}")
        else:
            self.input_zp = Parameter(Tensor(x_qparam.zero_point, dtype=zp_dtype))
            logger.debug(f"QuantWithSmoothHighPerformance: input zp of Layer({parallel_type}:{layer_name}) is "
                         f"{{{self.input_zp.shape}, {self.input_zp.dtype}, {self.input_zp.asnumpy()}}}")

    def construct(self, x):
        return self.quant(x, self.input_scale, self.input_zp, False, "ROUND", dtype.int8)


class MatmulCellForHook(QuantUnitCell):
    """MatmulCellForHook"""
    def __init__(self, layer_name, matmul):
        super().__init__(layer_name)
        self.mm = matmul

    def construct(self, *args, **kwargs):
        return self.mm(*args, **kwargs)

    def param_shard_state(self, tensor_parallel_num=1, **kwargs):
        return {}


class SmoothMatmul(QuantUnitCell):
    """SmoothMatmul"""
    def __init__(self, layer_name, mm, smooth_scale_):
        super().__init__(layer_name)
        self.mm = mm
        self.smooth_scale = Parameter(msops.div(1, smooth_scale_))
        logger.debug(f"SmoothMatmul: smooth_scale for act of Layer({layer_name}) is {{{self.smooth_scale.shape}, "
                     f"{self.smooth_scale.dtype}, {self.smooth_scale.asnumpy()}}}")

    @classmethod
    def _from_matmul_cell(cls, layer_name, src: MatmulCellForHook, smooth_scale):
        if not isinstance(src.mm, msops.MatMul):
            raise ValueError(
                f'matmul of MatmulCellForHook should be an instance of {msops.MatMul}, but got {src.mm}.')
        return cls(layer_name, src.mm, smooth_scale)

    @staticmethod
    def create(layer_name, src, smooth_scale):
        if isinstance(src, MatmulCellForHook):
            return SmoothMatmul._from_matmul_cell(layer_name, src, smooth_scale)
        raise ValueError(f"Not support creating SmoothMatmul from {src}.")

    def construct(self, x, weight):
        smooth_scale = msops.cast(self.smooth_scale, x.dtype)
        x = msops.mul(x, smooth_scale)
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
    def __init__(self, layer_name, mm, ic_, compute_dtype_):
        super().__init__(layer_name)
        self.mm = mm
        self.smooth_scale = Parameter(initializer('ones', (ic_,), dtype=compute_dtype_))

    @staticmethod
    def _from_matmul_prim(layer_name, src: msops.MatMul, ic, compute_dtype):
        return SmoothMatmulForDeploy(layer_name, src, ic, compute_dtype)

    @staticmethod
    def _from_matmul_cell(layer_name, src: MatmulCellForHook, ic, compute_dtype):
        if not isinstance(src.mm, msops.MatMul):
            raise ValueError(
                f'matmul of MatmulCellForHook should be an instance of {msops.MatMul}, but got {src.mm}.')
        return SmoothMatmulForDeploy(layer_name, src.mm, ic, compute_dtype)

    @staticmethod
    def create(layer_name, src, ic, compute_dtype):
        if isinstance(src, msops.MatMul):
            return SmoothMatmulForDeploy._from_matmul_prim(layer_name, src, ic, compute_dtype)
        if isinstance(src, MatmulCellForHook):
            return SmoothMatmulForDeploy._from_matmul_cell(layer_name, src, ic, compute_dtype)
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

    def __init__(self, layer_name, is_deploy, weight_scale, transpose_a=False, transpose_b=False,
                 dst_dtype=dtype.float16, smooth_scale=None):
        super().__init__(layer_name)
        self.dst_dtype = dst_dtype
        if is_deploy:
            self.weight_scale = Parameter(initializer("ones", weight_scale.shape, dtype.float32))
        else:
            self.weight_scale = Parameter(weight_scale.astype(dtype.float32))
            logger.debug(f"DynamicQuantMatmul: weight_scale of Layer({layer_name}) is {{{self.weight_scale.shape}, "
                         f"{self.weight_scale.dtype}, {self.weight_scale.asnumpy()}}}")
        self.smooth_scale = smooth_scale
        if not is_deploy and smooth_scale is not None:
            logger.debug(f"DynamicQuantMatmul: smooth_scale of Layer({layer_name}) is "
                         f"{{{self.smooth_scale.shape}, {self.smooth_scale.dtype}, {self.smooth_scale.asnumpy()}}}")
        self.dynamic_quant = DynamicQuantExt()
        # FIXME set dtype to dst_dtype when qbmm support bfp16 output
        self.qbmm = QuantBatchMatmul(transpose_x1=transpose_a, transpose_x2=transpose_b, dtype=dtype.float16)

    @staticmethod
    def _from_matmul_prim(layer_name, is_deploy, w_qparam: QuantParam, transpose_a=False, transpose_b=False,
                          dst_dtype=dtype.float16):
        return DynamicQuantMatmul(layer_name, is_deploy, w_qparam.scale, transpose_a, transpose_b, dst_dtype, None)

    @staticmethod
    def _from_matmul_cell(layer_name, is_deploy, src: MatmulCellForHook, w_qparam: QuantParam, transpose_a=False,
                          transpose_b=False, dst_dtype=dtype.float16):
        if not isinstance(src.mm, msops.MatMul):
            raise ValueError(
                f'matmul of MatmulCellForHook should be an instance of {msops.MatMul}, but got {src.mm}.')
        return DynamicQuantMatmul(layer_name, is_deploy, w_qparam.scale, transpose_a, transpose_b, dst_dtype, None)

    @staticmethod
    def _from_smooth_matmul(layer_name, is_deploy, src: SmoothMatmul, w_qparam: QuantParam, transpose_a=False,
                            transpose_b=False, dst_dtype=dtype.float16):
        if isinstance(src.mm, (msops.MatMul, MatmulCellForHook)):
            return DynamicQuantMatmul(layer_name, is_deploy, w_qparam.scale, transpose_a, transpose_b, dst_dtype,
                                      src.smooth_scale)
        raise ValueError(
            f'matmul of SmoothMatmul should be an instance of {msops.MatMul} or {MatmulCellForHook}, but got {src.mm}.')

    @staticmethod
    def _from_smooth_matmul_for_deploy(layer_name, is_deploy, src: SmoothMatmulForDeploy, w_qparam: QuantParam,
                                       transpose_a=False, transpose_b=False, dst_dtype=dtype.float16):
        if isinstance(src.mm, (msops.MatMul, MatmulCellForHook)):
            return DynamicQuantMatmul(layer_name, is_deploy, w_qparam.scale, transpose_a, transpose_b, dst_dtype,
                                      src.smooth_scale)
        raise ValueError(
            f'matmul of SmoothMatmul should be an instance of {msops.MatMul} or {MatmulCellForHook}, but got {src.mm}.')

    @staticmethod
    def create(layer_name, src, w_qparam: QuantParam, is_deploy, transpose_a=False, transpose_b=False,
               dst_dtype=dtype.float16):
        """create"""
        if isinstance(src, msops.MatMul):
            return DynamicQuantMatmul._from_matmul_prim(layer_name, is_deploy, w_qparam, transpose_a, transpose_b,
                                                        dst_dtype)
        if isinstance(src, MatmulCellForHook):
            return DynamicQuantMatmul._from_matmul_cell(layer_name, is_deploy, src, w_qparam, transpose_a, transpose_b,
                                                        dst_dtype)
        if isinstance(src, SmoothMatmul):
            return DynamicQuantMatmul._from_smooth_matmul(layer_name, is_deploy, src, w_qparam, transpose_a,
                                                          transpose_b, dst_dtype)
        if isinstance(src, SmoothMatmulForDeploy):
            return DynamicQuantMatmul._from_smooth_matmul_for_deploy(layer_name, is_deploy, src, w_qparam, transpose_a,
                                                                     transpose_b, dst_dtype)
        raise ValueError(f"Not support creating DynamicQuantMatmul from {src}.")

    def construct(self, x, quant_weight):
        qx, x_scale = self.dynamic_quant(x, self.smooth_scale)
        output = self.qbmm(qx, quant_weight, self.weight_scale, None, None, x_scale)
        return output.astype(self.dst_dtype)

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

    def __init__(self, layer_name, is_deploy, w_qparam: QuantParam, transpose_a=False, transpose_b=False,
                 dst_type=dtype.float16, smooth_scale=None):
        super().__init__(layer_name)
        self.dst_dtype = dst_type
        if is_deploy:
            self.weight_scale = Parameter(initializer('ones', w_qparam.scale.shape, self.dst_dtype))
            self.weight_zp = Parameter(initializer('zeros', w_qparam.zero_point.shape, self.dst_dtype))
        else:
            self.weight_scale = Parameter(Tensor(w_qparam.scale.asnumpy(), dtype=self.dst_dtype))
            self.weight_zp = Parameter(Tensor(w_qparam.zero_point.asnumpy() * -1, dtype=self.dst_dtype))
            logger.debug(f"WeightQuantMatmul {PTQMode.QUANTIZE} mode: weight_scale of Layer({layer_name}) is "
                         f"{{{self.weight_scale.shape}, {self.weight_scale.dtype}, {self.weight_scale.asnumpy()}}}")
            logger.debug(f"WeightQuantMatmul {PTQMode.QUANTIZE} mode: weight_zp of Layer({layer_name}) is "
                         f"{{{self.weight_zp.shape}, {self.weight_zp.dtype}, {self.weight_zp.asnumpy()}}}")
        self.weight_qbmm = WeightQuantBatchMatmul(transpose_a, transpose_b, w_qparam.group_size)
        self.has_smooth = smooth_scale is not None
        self.smooth_scale = smooth_scale
        if self.has_smooth:
            logger.debug(f"WeightQuantMatmul {PTQMode.QUANTIZE} mode: smooth_scale of Layer({layer_name}) is "
                         f"{{{self.smooth_scale.shape}, {self.smooth_scale.dtype}, {self.smooth_scale.asnumpy()}}}")

    @classmethod
    def _from_matmul_prim(cls, layer_name, w_qparam: QuantParam, is_deploy, transpose_a=False, transpose_b=False,
                          dst_dtype=dtype.float16):
        return cls(layer_name, is_deploy, w_qparam, transpose_a, transpose_b, dst_dtype, None)

    @classmethod
    def _from_matmul_cell(cls, layer_name, src: MatmulCellForHook, w_qparam: QuantParam, is_deploy,
                          transpose_a=False, transpose_b=False, dst_dtype=dtype.float16):
        if not isinstance(src.mm, msops.MatMul):
            raise ValueError(
                f'matmul of MatmulCellForHook should be an instance of {msops.MatMul}, but got {src.mm}.')
        return cls(layer_name, is_deploy, w_qparam, transpose_a, transpose_b, dst_dtype, None)

    @classmethod
    def _from_smooth_matmul(cls, layer_name, src: SmoothMatmul, w_qparam: QuantParam, is_deploy,
                            transpose_a=False, transpose_b=False, dst_dtype=dtype.float16):
        if isinstance(src.mm, (msops.MatMul, MatmulCellForHook)):
            return cls(layer_name, is_deploy, w_qparam, transpose_a, transpose_b, dst_dtype, src.smooth_scale)
        raise ValueError(
            f'matmul of SmoothMatmul should be an instance of {msops.MatMul} or {MatmulCellForHook}, but got {src.mm}.')

    @classmethod
    def _from_smooth_matmul_for_deploy(cls, layer_name, src: SmoothMatmulForDeploy, w_qparam: QuantParam, is_deploy,
                                       transpose_a=False, transpose_b=False, dst_dtype=dtype.float16):
        if isinstance(src.mm, (msops.MatMul, MatmulCellForHook)):
            return cls(layer_name, is_deploy, w_qparam, transpose_a, transpose_b, dst_dtype, src.smooth_scale)
        raise ValueError(
            f'matmul of SmoothMatmul should be an instance of {msops.MatMul} or {MatmulCellForHook}, but got {src.mm}.')

    @staticmethod
    def create(layer_name, linear, q_weight, w_qparam: QuantParam, is_deploy, transpose_a=False, transpose_b=False,
               dst_dtype=dtype.float16):
        """create"""
        if isinstance(linear.matmul, msops.MatMul):
            return WeightQuantMatmul._from_matmul_prim(layer_name, w_qparam, is_deploy, transpose_a, transpose_b,
                                                       dst_dtype)
        if isinstance(linear.matmul, MatmulCellForHook):
            return WeightQuantMatmul._from_matmul_cell(layer_name, linear.matmul, w_qparam, is_deploy, transpose_a,
                                                       transpose_b, dst_dtype)
        if isinstance(linear.matmul, SmoothMatmul):
            return WeightQuantMatmul._from_smooth_matmul(layer_name, linear.matmul, w_qparam, is_deploy, transpose_a,
                                                         transpose_b, dst_dtype)
        if isinstance(linear.matmul, SmoothMatmulForDeploy):
            return WeightQuantMatmul._from_smooth_matmul_for_deploy(layer_name, linear.matmul, w_qparam, is_deploy,
                                                                    transpose_a, transpose_b, dst_dtype)
        raise ValueError(f"Not support creating WeightQuantMatmul from {linear}.")

    def construct(self, x, weight):
        """forward for WeightQuantMatmul cell"""
        if self.has_smooth:
            x = msops.mul(x, self.smooth_scale)
        output = self.weight_qbmm(x, weight, self.weight_scale, self.weight_zp, None, None, None)
        return output.astype(self.dst_dtype)

    # pylint: disable=arguments-differ
    def param_shard_state(self, tensor_parallel_num=1, parallel_type: ParallelType = ParallelType.NO_PARALLEL):
        if parallel_type == ParallelType.COL_PARALLEL:
            smooth_scale_shard = (1,)
            t_scale_shard = (tensor_parallel_num,)
            t_zp_shard = (tensor_parallel_num,)
        elif parallel_type == ParallelType.ROW_PARALLEL:
            smooth_scale_shard = (tensor_parallel_num,)
            t_scale_shard = (1,)
            t_zp_shard = (1,)
        else:
            return {}
        shard_state = {
            self.weight_scale.name: {'shape': self.weight_scale.shape, 'shard': t_scale_shard},
            self.weight_zp.name: {'shape': self.weight_zp.shape, 'shard': t_zp_shard},
        }
        if self.smooth_scale:
            shard_state[self.smooth_scale.name] = {'shape': self.smooth_scale.shape, 'shard': smooth_scale_shard}
        return shard_state


class WeightQuantInt4Matmul(WeightQuantMatmul):
    """WeightQuantInt4Matmul"""

    def __init__(self, layer_name, is_deploy, w_qparam: QuantParam, transpose_a=False, transpose_b=False,
                 dst_type=dtype.float16, smooth_scale=None):
        super().__init__(layer_name, is_deploy, w_qparam, transpose_a, transpose_b, dst_type, smooth_scale)
        self.weight_qbmm = WeightQuantBatchMatmul(transpose_a, False, w_qparam.group_size)

    @staticmethod
    def create(layer_name, linear, q_weight, w_qparam: QuantParam, is_deploy, transpose_a=False, transpose_b=False,
               dst_dtype=dtype.float16):
        """create"""
        trans_b = linear.transpose_b
        rank = len(q_weight.shape)
        ic_idx, oc_idx = (rank - 1, rank - 2) if trans_b else (rank - 2, rank - 1)
        ic, oc = q_weight.shape[ic_idx], q_weight.shape[oc_idx]
        if is_deploy:
            weight_shape = (ic, oc // 2)
            q_weight = Parameter(Tensor(np.ones(weight_shape), w_qparam.quant_dtype), name=linear.weight.name)
        else:
            q_weight = q_weight.asnumpy().T if trans_b else q_weight.asnumpy()
            q_weight_pack = np_int4data_pack_to_int8(q_weight)
            logger.debug(f"WeightQuantInt4Matmul: pack q_weight of Layer({layer_name}) is "
                         f"{{{q_weight_pack.shape}, {q_weight_pack.dtype}, {q_weight_pack}}}")
            q_weight = Parameter(Tensor(q_weight_pack, dtype=w_qparam.quant_dtype), name=linear.weight.name)

        if isinstance(linear.matmul, msops.MatMul):
            wqbmm = WeightQuantInt4Matmul._from_matmul_prim(layer_name, w_qparam, is_deploy, transpose_a, transpose_b,
                                                            dst_dtype)
        elif isinstance(linear.matmul, MatmulCellForHook):
            wqbmm = WeightQuantInt4Matmul._from_matmul_cell(layer_name, linear.matmul, w_qparam, is_deploy, transpose_a,
                                                            transpose_b, dst_dtype)
        elif isinstance(linear.matmul, SmoothMatmul):
            wqbmm = WeightQuantInt4Matmul._from_smooth_matmul(layer_name, linear.matmul, w_qparam, is_deploy,
                                                              transpose_a, transpose_b, dst_dtype)
        elif isinstance(linear.matmul, SmoothMatmulForDeploy):
            wqbmm = WeightQuantInt4Matmul._from_smooth_matmul_for_deploy(layer_name, linear.matmul, w_qparam, is_deploy,
                                                                         transpose_a, transpose_b, dst_dtype)
        else:
            raise ValueError(f"Not support creating WeightQuantMatmul from {linear}.")
        return wqbmm, q_weight


class AllQuantMatmul(QuantUnitCell):
    """all quant mm"""

    def __init__(self, layer_name, transpose_b=False, dst_dtype=dtype.float16):
        super().__init__(layer_name)
        self.dst_dtype = dst_dtype
        self.transpose_b = transpose_b
        self.dequant_scale = None
        self.offset = None

    @staticmethod
    def _correction_into_bias(quant_weight: Parameter, x_qparam: QuantParam, w_qparam: QuantParam, trans_b,
                              new_bias_need_allreduce, dst_dtype, origin_bias: Optional[Parameter] = None) -> Tensor:
        """compute fused bias"""
        if quant_weight is None:
            raise ValueError("quant_weight is None.")
        x_zp = x_qparam.zero_point.asnumpy()
        q_correction = -np.sum(x_zp.astype(np.int32) * quant_weight.asnumpy().astype(np.int32),
                               axis=1 if trans_b else 0).astype(np.int32)
        if new_bias_need_allreduce:
            t_q_correction = Tensor(q_correction)
            t_q_correction = msops.AllReduce(op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)(t_q_correction)
            q_correction = t_q_correction.asnumpy()
        dequant_scale = np.squeeze(x_qparam.scale.asnumpy() * w_qparam.scale.asnumpy()).astype(np.float64)
        correction = q_correction.astype(np.float64) * dequant_scale
        if origin_bias is not None:
            return Tensor(origin_bias.asnumpy().astype(np.float64) + correction, dtype=dst_dtype)
        return Tensor(correction, dtype=dst_dtype)

    @staticmethod
    def _from_matmul_prim(layer_name, x_qparam: QuantParam, w_qparam: QuantParam, is_deploy, transpose_a=False,
                          transpose_b=False, dst_dtype=dtype.float16, kernel_type: KernelType = KernelType.ASD):
        return AllQuantMatmul.create_self(layer_name, is_deploy, x_qparam, w_qparam, transpose_a, transpose_b,
                                          dst_dtype, kernel_type), None

    @staticmethod
    def _from_matmul_cell(layer_name, src: MatmulCellForHook, x_qparam: QuantParam, w_qparam: QuantParam, is_deploy,
                          transpose_a=False, transpose_b=False, dst_dtype=dtype.float16,
                          kernel_type: KernelType = KernelType.ASD):
        if not isinstance(src.mm, msops.MatMul):
            raise ValueError(
                f'matmul of MatmulCellForHook should be an instance of {msops.MatMul}, but got {src.mm}.')
        return AllQuantMatmul.create_self(layer_name, is_deploy, x_qparam, w_qparam, transpose_a, transpose_b,
                                          dst_dtype, kernel_type), None

    @staticmethod
    def _from_smooth_matmul(layer_name, src: SmoothMatmul, x_qparam: QuantParam, w_qparam: QuantParam, is_deploy,
                            transpose_a=False, transpose_b=False, dst_dtype=dtype.float16,
                            kernel_type: KernelType = KernelType.ASD):
        """_from_smooth_matmul"""
        smooth_scale = src.smooth_scale.asnumpy()
        if isinstance(src.mm, msops.MatMul):
            qmm = AllQuantMatmul.create_self(layer_name, is_deploy, x_qparam, w_qparam, transpose_a, transpose_b,
                                             dst_dtype, kernel_type)
            return qmm, smooth_scale
        if isinstance(src.mm, MatmulCellForHook):
            qmm, _ = AllQuantMatmul._from_matmul_cell(layer_name, src.mm, x_qparam, w_qparam, is_deploy, transpose_a,
                                                      transpose_b, dst_dtype, kernel_type)
            return qmm, smooth_scale
        raise ValueError(
            f'matmul of SmoothMatmul should be an instance of {msops.MatMul} or {MatmulCellForHook}, but got {src.mm}.')

    @staticmethod
    def _from_smooth_matmul_for_deploy(layer_name, src: SmoothMatmulForDeploy, x_qparam: QuantParam,
                                       w_qparam: QuantParam, is_deploy, transpose_a=False, transpose_b=False,
                                       dst_dtype=dtype.float16, kernel_type: KernelType = KernelType.ASD):
        """_from_smooth_matmul_for_deploy"""
        smooth_scale = src.smooth_scale.asnumpy()
        if isinstance(src.mm, msops.MatMul):
            qmm = AllQuantMatmul.create_self(layer_name, is_deploy, x_qparam, w_qparam, transpose_a, transpose_b,
                                             dst_dtype, kernel_type)
            return qmm, smooth_scale
        if isinstance(src.mm, MatmulCellForHook):
            qmm, _ = AllQuantMatmul._from_matmul_cell(layer_name, src.mm, x_qparam, w_qparam, is_deploy, transpose_a,
                                                      transpose_b, dst_dtype, kernel_type)
            return qmm, smooth_scale
        raise ValueError(
            f'matmul of SmoothMatmulForDeploy should be an instance of {msops.MatMul} or {MatmulCellForHook}, '
            f'but got {src.mm}.')

    @staticmethod
    def create_self(layer_name, is_deploy, x_qparam: QuantParam, w_qparam: QuantParam, transpose_a=False,
                    transpose_b=False, dst_dtype=dtype.float16, kernel_type: KernelType = KernelType.ASD):
        if kernel_type in (KernelType.ASD, KernelType.ACLNN, KernelType.INTERNAL):
            return AllQuantMatmulHighPerformance(layer_name, is_deploy, x_qparam, w_qparam, transpose_a, transpose_b,
                                                 dst_dtype)
        if kernel_type is KernelType.HIGH_PRECISION:
            return AllQuantMatmulHighPrecision(layer_name, is_deploy, x_qparam, w_qparam, transpose_a, transpose_b,
                                               dst_dtype)
        raise RuntimeError(f"Not supported kernel type: {kernel_type}")

    @staticmethod
    def create(layer_name, linear, parallel_type: ParallelType, q_weight, x_qparam: QuantParam, w_qparam: QuantParam,
               is_deploy, tp_size, dst_dtype=dtype.float16,
               kernel_type: KernelType = KernelType.ASD) -> (QuantWithSmooth, 'AllQuantMatmul', Parameter):
        """create"""
        trans_a = False
        trans_b = linear.transpose_b
        rank = len(q_weight.shape)
        ic_idx, oc_idx = (rank - 1, rank - 2) if trans_b else (rank - 2, rank - 1)
        ic, oc = q_weight.shape[ic_idx], q_weight.shape[oc_idx]
        # create qmm
        if isinstance(linear.matmul, msops.MatMul):
            qmm, smooth_scale = AllQuantMatmul._from_matmul_prim(layer_name, x_qparam, w_qparam, is_deploy, trans_a,
                                                                 trans_b, dst_dtype, kernel_type)
        elif isinstance(linear.matmul, MatmulCellForHook):
            qmm, smooth_scale = AllQuantMatmul._from_matmul_cell(layer_name, linear.matmul, x_qparam, w_qparam,
                                                                 is_deploy, trans_a, trans_b, dst_dtype, kernel_type)
        elif isinstance(linear.matmul, SmoothMatmul):
            qmm, smooth_scale = AllQuantMatmul._from_smooth_matmul(layer_name, linear.matmul, x_qparam, w_qparam,
                                                                   is_deploy, trans_a, trans_b, dst_dtype, kernel_type)
        elif isinstance(linear.matmul, SmoothMatmulForDeploy):
            qmm, smooth_scale = AllQuantMatmul._from_smooth_matmul_for_deploy(layer_name, linear.matmul, x_qparam,
                                                                              w_qparam, is_deploy, trans_a, trans_b,
                                                                              dst_dtype, kernel_type)
        else:
            raise ValueError(f"Not support creating AllQuantMatmul from {linear.matmul}.")
        # create quant
        quant = QuantWithSmooth.create(layer_name, x_qparam, ic, dst_dtype, is_deploy, parallel_type, smooth_scale,
                                       kernel_type)
        # correction into bias
        bias_name = linear.bias.name if linear.has_bias else q_weight.name + "_bias"
        if is_deploy:
            bias = Parameter(initializer("zeros", (oc,), dst_dtype), name=bias_name)
        else:
            # fuse bias
            origin_bias = linear.bias if linear.has_bias else None
            if parallel_type is ParallelType.ROW_PARALLEL:
                t_bias = AllQuantMatmul._correction_into_bias(q_weight, x_qparam, w_qparam, trans_b, tp_size > 1,
                                                              dst_dtype, origin_bias)
            else:
                t_bias = AllQuantMatmul._correction_into_bias(q_weight, x_qparam, w_qparam, trans_b, False, dst_dtype,
                                                              origin_bias)
            bias = Parameter(t_bias, name=bias_name)
        return quant, qmm, bias

    # pylint: disable=arguments-differ
    def param_shard_state(self, tensor_parallel_num=1, parallel_type: ParallelType = ParallelType.NO_PARALLEL):
        if parallel_type == ParallelType.COL_PARALLEL:
            q_shard = (tensor_parallel_num,)
        elif parallel_type == ParallelType.ROW_PARALLEL:
            q_shard = (1,)
        else:
            return {}
        shard_state = {self.dequant_scale.name: {'shape': self.dequant_scale.shape, 'shard': q_shard}}
        if self.offset is not None:
            shard_state[self.offset.name] = {'shape': self.offset.shape, 'shard': q_shard}
        return shard_state


class AllQuantMatmulHighPrecision(AllQuantMatmul):
    """AllQuantMatmulHighPrecision"""

    def __init__(self, layer_name, is_deploy, x_qparam: QuantParam, w_qparam: QuantParam, transpose_a=False,
                 transpose_b=False, dst_dtype=dtype.float16):
        super().__init__(layer_name, transpose_b, dst_dtype)
        self.transpose_a = transpose_a
        if is_deploy:
            self.dequant_scale = Parameter(initializer('ones', w_qparam.scale.shape, dtype.float64))
        else:
            np_dequant_scale = x_qparam.scale.asnumpy().astype(np.float64) * w_qparam.scale.asnumpy().astype(np.float64)
            self.dequant_scale = Parameter(Tensor(np_dequant_scale))
            logger.debug(f"AllQuantMatmulHighPrecision: dequant_scale of Layer({layer_name}) is "
                         f"{{{self.dequant_scale.shape}, {self.dequant_scale.dtype}, {self.dequant_scale.asnumpy()}}}")

        if w_qparam.zero_point is None:
            self.offset = None
            self.has_offset = False
        else:
            self.has_offset = True
            self.offset = Parameter(Tensor(w_qparam.zero_point, dtype=dtype.float64))
            if not is_deploy:
                logger.debug(f"AllQuantMatmulHighPrecision: offset of Layer({layer_name}) is {{{self.offset.shape}, "
                             f"{self.offset.dtype}, {self.offset.asnumpy()}}}")

    def construct(self, qx, quant_weight):
        """construct"""
        qx = msops.cast(qx, dtype.float64)
        if self.transpose_a:
            qx = msops.transpose(qx, (1, 0))
        quant_weight = msops.cast(quant_weight, dtype.float64)
        if self.transpose_b:
            quant_weight = msops.transpose(quant_weight, (1, 0))
        mm = msops.matmul(qx, quant_weight)
        output = mm * self.dequant_scale
        if self.has_offset:
            output = output + self.offset
        return output.astype(self.dst_dtype)


class AllQuantMatmulHighPerformance(AllQuantMatmul):
    """AllQuantMatmulHighPerformance"""

    def __init__(self, layer_name, is_deploy, x_qparam: QuantParam, w_qparam: QuantParam, transpose_a=False,
                 transpose_b=False, dst_dtype=dtype.float16):
        super().__init__(layer_name, transpose_b, dst_dtype)
        scale_dtype = dtype.float32 if self.dst_dtype == dtype.bfloat16 else dtype.int64
        if is_deploy:
            self.dequant_scale = Parameter(initializer('ones', w_qparam.scale.shape, scale_dtype))
        else:
            self.dequant_scale = Parameter(Tensor(self._compute_dequant_scale(x_qparam.scale.asnumpy(),
                                                                              w_qparam.scale.asnumpy(),
                                                                              dst_dtype), dtype=scale_dtype))
            logger.debug(f"AllQuantMatmul: dequant_scale of Layer({layer_name}) is "
                         f"{{{self.dequant_scale.shape}, {self.dequant_scale.dtype}, {self.dequant_scale.asnumpy()}}}")

        self.qbmm = QuantBatchMatmul(transpose_x1=transpose_a, transpose_x2=transpose_b, dtype=self.dst_dtype)
        if w_qparam.zero_point is None:
            self.offset = None
        else:
            self.offset = Parameter(Tensor(w_qparam.zero_point, dtype=dtype.float32))
            if not is_deploy:
                logger.debug(f"AllQuantMatmul: offset of Layer({layer_name}) is {{{self.offset.shape}, "
                             f"{self.offset.dtype}, {self.offset.asnumpy()}}}")

    @staticmethod
    def _compute_dequant_scale(input_scale, weight_scale, dst_dtype):
        """compute_dequant_scale"""
        dequant_scale = input_scale.astype(np.float32) * weight_scale.astype(np.float32)
        # when dst_dtype is dtype.bfloat16, qbmm ops use asd ops, scale need be fp32
        # scale need be int64 when qbmm use internal ops
        if dst_dtype == dtype.bfloat16:
            return dequant_scale
        scale_i64 = NumpyQuantOps.trans_fp32_to_i64(dequant_scale)
        return scale_i64

    def construct(self, qx, quant_weight):
        # x: fp16 quant_weight: int8
        output = self.qbmm(qx, quant_weight, self.dequant_scale, self.offset, None)
        return output.astype(self.dst_dtype)


class C8PagedAttentionCell(QuantUnitCell):
    """C8PagedAttentionMgrCell"""
    def __init__(self, layer_name, cfg: InnerPTQConfig, compute_type, n, d, k_qparam: QuantParam,
                 v_qparam: QuantParam):
        super().__init__(layer_name)
        ic = k_qparam.scale.shape[0]
        is_deploy = cfg.mode == PTQMode.DEPLOY
        self.enable_deploy_fusion = cfg.enable_deploy_fusion
        kernel_type = KernelType.INTERNAL if compute_type == dtype.bfloat16 else KernelType.ASD
        if is_deploy:
            if self.enable_deploy_fusion:
                self.k_v_scale_fusion, self.k_v_zp_fusion = self.__param_init(kernel_type, ic)
            else:
                self.k_scale_no_fusion = Parameter(initializer('ones', (ic), compute_type))
                self.k_zp_no_fusion = Parameter(initializer('zeros', (ic), compute_type))
                self.v_scale_no_fusion = Parameter(initializer('ones', (ic), compute_type))
                self.v_zp_no_fusion = Parameter(initializer('zeros', (ic), compute_type))
        else:
            self.k_scale_no_fusion = Parameter(Tensor(k_qparam.scale.asnumpy(), dtype=compute_type))
            self.k_zp_no_fusion = Parameter(Tensor(k_qparam.zero_point.asnumpy(), dtype=compute_type))
            self.v_scale_no_fusion = Parameter(Tensor(v_qparam.scale.asnumpy(), dtype=compute_type))
            self.v_zp_no_fusion = Parameter(Tensor(v_qparam.zero_point.asnumpy(), dtype=compute_type))
            self.k_v_scale_fusion, self.k_v_zp_fusion = self._param_compute(kernel_type,
                                                                            k_qparam.scale.asnumpy(),
                                                                            k_qparam.zero_point.asnumpy(),
                                                                            v_qparam.scale.asnumpy(),
                                                                            v_qparam.zero_point.asnumpy())
        self._key_output_anti_quant = AntiQuantCell(n, d, compute_type)
        self._value_output_anti_quant = AntiQuantCell(n, d, compute_type)
        self.quant = QuantV2()

    def __param_init(self, kernel_type, ic):
        if kernel_type == KernelType.ASD:
            return Parameter(initializer('ones', (2, ic), dtype=dtype.int64)), \
                    Parameter(initializer('zeros', (2, ic), dtype=dtype.int32))
        if kernel_type == KernelType.INTERNAL:
            return Parameter(initializer('ones', (2, ic), dtype.float16)), \
                    Parameter(initializer('zeros', (2, ic), dtype.float16))
        raise ValueError(f"kernel_type:{kernel_type} is unsupported in C8PagedAttentionCell.")

    def _param_compute(self, kernel_type, key_t_scale, key_t_zp, value_t_scale, value_t_zp):
        if kernel_type == KernelType.ASD:
            return self._param_compute_asd(key_t_scale, key_t_zp, value_t_scale, value_t_zp)
        if kernel_type == KernelType.INTERNAL:
            return self._param_compute_internal(key_t_scale, key_t_zp, value_t_scale, value_t_zp)
        raise ValueError(f"kernel_type:{kernel_type} is unsupported in C8PagedAttentionCell.")

    def _param_compute_asd(self, key_t_scale, key_t_zp, value_t_scale, value_t_zp):
        """_param_compute_asd"""
        t_scale_len = key_t_scale.shape[0]
        key_t_scale = convert_fp32_to_int64(key_t_scale.astype(np.float32))
        value_t_scale = convert_fp32_to_int64(value_t_scale.astype(np.float32))
        key_value_t_scale = np.concatenate((key_t_scale.reshape((1, t_scale_len)),
                                            value_t_scale.reshape((1, t_scale_len))))

        t_zp_len = value_t_scale.shape[0]
        key_t_zp = (key_t_zp*-1).astype(np.int32)
        value_t_zp = (value_t_zp*-1).astype(np.int32)
        key_value_t_zp = np.concatenate((key_t_zp.reshape((1, t_zp_len)), value_t_zp.reshape((1, t_zp_len))))

        k_v_scale_fusion = Parameter((Tensor(key_value_t_scale, dtype=dtype.int64)))
        k_v_zp_fusion = Parameter((Tensor(key_value_t_zp, dtype=dtype.int32)))
        return k_v_scale_fusion, k_v_zp_fusion

    def _param_compute_internal(self, key_t_scale, key_t_zp, value_t_scale, value_t_zp):
        """_param_compute_internal"""
        t_scale_len = key_t_scale.shape[0]
        key_value_t_scale = np.concatenate((key_t_scale.reshape((1, t_scale_len)),
                                            value_t_scale.reshape((1, t_scale_len))))
        t_zp_len = key_t_zp.shape[0]
        key_t_zp = key_t_zp*-1
        value_t_zp = value_t_zp*-1
        key_value_t_zp = np.concatenate((key_t_zp.reshape((1, t_zp_len)), value_t_zp.reshape((1, t_zp_len))))
        k_v_scale_fusion = Parameter(Tensor(key_value_t_scale, dtype=dtype.float16))
        k_v_zp_fusion = Parameter(Tensor(key_value_t_zp, dtype=dtype.float16))
        return k_v_scale_fusion, k_v_zp_fusion

    # pylint: disable=W0613
    def paged_attn(self, pa_mgr, query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=None):
        """The forward compute of Paged Attention."""
        if not self.enable_deploy_fusion:
            kcache = self._key_output_anti_quant(pa_mgr.key_cache, self.k_zp_no_fusion, self.k_scale_no_fusion)
            vcache = self._value_output_anti_quant(pa_mgr.value_cache, self.v_zp_no_fusion, self.v_scale_no_fusion)
            return pa_mgr.paged_attention(query, kcache, vcache, block_tables, batch_valid_length)
        return pa_mgr.paged_attention(query, pa_mgr.key_cache, pa_mgr.value_cache, block_tables,
                                      batch_valid_length, self.k_v_scale_fusion, self.k_v_zp_fusion)

    def paged_attn_with_alibi(self, pa_mgr, query, batch_valid_length, block_tables, alibi_tensor):
        """The forward compute of Paged Attention."""
        if not self.enable_deploy_fusion:
            kcache = self._key_output_anti_quant(pa_mgr.key_cache, self.k_zp_no_fusion, self.k_scale_no_fusion)
            vcache = self._value_output_anti_quant(pa_mgr.value_cache, self.v_zp_no_fusion, self.v_scale_no_fusion)
            return pa_mgr.paged_attention_with_alibi(query, kcache, vcache, block_tables, batch_valid_length,
                                                     alibi_tensor)
        return pa_mgr.paged_attention_with_alibi(query, pa_mgr.key_cache, pa_mgr.value_cache, block_tables,
                                                 batch_valid_length, self.k_v_scale_fusion,
                                                 self.k_v_zp_fusion, alibi_tensor)

    # pylint: disable=arguments-differ
    # pylint: disable=W0613
    def param_shard_state(self, tensor_parallel_num=1, parallel_type: ParallelType = ParallelType.NO_PARALLEL):
        state_dict = {}
        if self.enable_deploy_fusion:
            key_value_t_scale_shard = (1, tensor_parallel_num)
            key_value_t_zp_shard = (1, tensor_parallel_num)
            state_dict[self.k_v_scale_fusion.name] = {'shape': self.k_v_scale_fusion.shape,
                                                      'shard': key_value_t_scale_shard}
            state_dict[self.k_v_zp_fusion.name] = {'shape': self.k_v_zp_fusion.shape,
                                                   'shard': key_value_t_zp_shard}
        else:
            key_t_scale_shard = (tensor_parallel_num,)
            key_t_zp_shard = (tensor_parallel_num,)

            value_t_scale_shard = (tensor_parallel_num,)
            value_t_zp_shard = (tensor_parallel_num,)

            state_dict[self.k_scale_no_fusion.name] = {'shape': self.k_scale_no_fusion.shape,
                                                       'shard': key_t_scale_shard}
            state_dict[self.k_zp_no_fusion.name] = {'shape': self.k_zp_no_fusion.shape,
                                                    'shard': key_t_zp_shard}
            state_dict[self.v_scale_no_fusion.name] = {'shape': self.v_scale_no_fusion.shape,
                                                       'shard': value_t_scale_shard}
            state_dict[self.v_zp_no_fusion.name] = {'shape': self.v_zp_no_fusion.shape,
                                                    'shard': value_t_zp_shard}
        return state_dict


class QuantV2Cell(QuantUnitCell):
    """QuantCellV2, warp Quant to support serialize and deserialize use QuantV2."""
    def __init__(self, layer_name, is_deploy, t_scale: Tensor, t_zp: Tensor):
        super().__init__(layer_name)
        self._is_perchannel: bool = t_scale.shape != (1,)
        if is_deploy:
            self.t_scale = Parameter(initializer('ones', t_scale.shape, t_scale.dtype))
            self.t_zp = Parameter(initializer('zeros', t_zp.shape, t_zp.dtype))
        else:
            self.t_scale = Parameter(t_scale)
            self.t_zp = Parameter(t_zp)
        self.quant = QuantV2()

    def construct(self, x):
        """construct network forward"""
        return self.quant(x, self.t_scale, self.t_zp, False, "ROUND", dtype.int8)

    @staticmethod
    def create(layer_name, dst_type, cfg: InnerPTQConfig, qparam: QuantParam):
        '''create'''
        is_deploy = cfg.mode == PTQMode.DEPLOY
        ops_priority = KernelType.INTERNAL
        t_scale = qparam.scale.asnumpy()
        t_zp = qparam.zero_point.asnumpy()
        if ops_priority == KernelType.ACLNN:
            return QuantV2Cell(layer_name, is_deploy, Tensor(t_scale, dtype=dst_type),
                               Tensor(t_zp, dtype=dst_type))
        return QuantV2Cell(layer_name, is_deploy, Tensor(t_scale, dtype=dst_type),
                           Tensor(t_zp.astype(np.int8), dtype=dtype.int8))

    # pylint: disable=arguments-differ
    # pylint: disable=W0613
    def param_shard_state(self, tensor_parallel_num=1, parallel_type: ParallelType = ParallelType.NO_PARALLEL):
        state_dict = {}
        t_scale_shard = (tensor_parallel_num,)
        t_zp_shard = (tensor_parallel_num,)
        state_dict[self.t_scale.name] = {'shape': self.t_scale.shape, 'shard': t_scale_shard}
        state_dict[self.t_zp.name] = {'shape': self.t_zp.shape, 'shard': t_zp_shard}
        return state_dict
