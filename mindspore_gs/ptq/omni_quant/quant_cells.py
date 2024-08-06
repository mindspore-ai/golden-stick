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
"""ptq quant cells."""
from typing import Union
import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, dtype
from mindspore.nn import Cell
from mindspore import ops as msops
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.operations._infer_ops import QuantV2
from mindspore.ops.auto_generate import WeightQuantBatchMatmul, QuantBatchMatmul
from mindspore.common.initializer import initializer
from mindspore.ops.operations.comm_ops import ReduceOp
from mindspore.communication.management import GlobalComm
from mindformers.modules.layers import Linear
from mindformers.experimental.distri_cores.tensor_parallel.layers import (
    ColumnParallelLinear, RowParallelLinear
)
from mindformers.experimental.distri_cores.tensor_parallel.collective_primitives import (
    MaxFromTensorParallelRegion, MinFromTensorParallelRegion
)

from mindspore_gs.ptq.ptq_config import InnerPTQConfig
from mindspore_gs.quantization.quant_utils import (
    get_quant_min_max, cal_quantization_params,
    quant_tensor_data)
from mindspore_gs.common.numpy_quant_common import NumpyQuantOps


class MinMaxLinearWrapper(Cell):
    """Linear layer wrapper with min max"""

    def __init__(self,
                 linear_name: str,
                 linear: Union[Linear, ColumnParallelLinear, RowParallelLinear],
                 cfg: InnerPTQConfig = None):
        super().__init__()
        if not isinstance(linear, (Linear, ColumnParallelLinear, RowParallelLinear)):
            raise ValueError("only Linear、ColumnParallelLinear、RowParallelLinear cell is supported,"
                             f"but {linear_name} type is {type(linear)}.")
        self.cfg = cfg
        self._handler: Linear = linear
        self.mm = MatMul(linear)
        self.float_weight = linear.weight
        rank = len(linear.weight.shape)
        ic_axis = rank - 1 if linear.matmul.transpose_b else rank - 2
        self.weight_quantizer_axis = rank - 2 if linear.matmul.transpose_b else rank - 1
        self.ic = linear.weight.shape[ic_axis]
        self.oc = linear.weight.shape[self.weight_quantizer_axis]
        self._weight_symmetric = cfg.weight_symmetric
        self._act_symmetric = cfg.act_symmetric

    def float_forward(self, inputs):
        self._handler.weight = self.float_weight
        self._handler.matmul = self.mm
        return self._handler(inputs)

    def quant_forward(self, inputs):
        self._handler.matmul = self.qmm
        return self._handler(inputs)

    def construct(self, inputs, weight=None):
        # for deploy forward
        raise NotImplementedError


class SQLinearWrapper(MinMaxLinearWrapper):
    """Linear layer wrapper with min max"""

    def __init__(self,
                 linear_name: str,
                 linear: Union[Linear, ColumnParallelLinear, RowParallelLinear],
                 cfg: InnerPTQConfig = None):
        super().__init__(linear_name, linear, cfg)
        self.pre_clip_ratio = None
        self.post_clip_ratio = None
        self.smooth_alpha = None
        self.smooth_type = None
        self.act_clip_ratio = 1.0

        self.weight_quantizer_min_max_axis = 0 if self.weight_quantizer_axis else 1
        weight_observer_axis = 1 if self._handler.matmul.transpose_b else 0
        self.observer_min_max_axis = 0 if weight_observer_axis else 1
        self._weight_signed = cfg.weight_quant_dtype == dtype.int8
        self.weight_quant_min, self.weight_quant_max = get_quant_min_max(num_bits=8,
                                                                         signed=self._weight_signed,
                                                                         narrow_range=cfg.weight_narrow_range)
        if cfg.act_quant_dtype == dtype.int8:
            self._act_signed = True
            self.act_quant_min, self.act_quant_max = get_quant_min_max(num_bits=8,
                                                                       signed=self._act_signed,
                                                                       narrow_range=cfg.act_narrow_range)
        self.observer_x = []
        self.observer_w_max = None
        self.observer_w_min = None
        self.quantizer_x_max = None
        self.quantizer_x_min = None
        self.quantizer_w_max = None
        self.quantizer_w_min = None

        self.x_obs_max = msops.max
        self.x_obs_min = msops.min
        if isinstance(self._handler, Linear):
            self.x_quant_max = msops.max
            self.x_quant_min = msops.min
            self.w_obs_max = msops.max
            self.w_obs_min = msops.min
            self.w_quant_max = msops.max
            self.w_quant_min = msops.min
        elif isinstance(self._handler, ColumnParallelLinear):
            self.x_quant_max = msops.max
            self.x_quant_min = msops.min
            self.w_obs_max = MaxFromTensorParallelRegion()
            self.w_obs_min = MinFromTensorParallelRegion()
            self.w_quant_max = msops.max
            self.w_quant_min = msops.min
        elif isinstance(self._handler, RowParallelLinear):
            self.x_quant_max = MaxFromTensorParallelRegion()
            self.x_quant_min = MinFromTensorParallelRegion()
            self.w_obs_max = msops.max
            self.w_obs_min = msops.min
            self.w_quant_max = MaxFromTensorParallelRegion()
            self.w_quant_min = MinFromTensorParallelRegion()

        if isinstance(self._handler, Linear):
            self.compute_type = self._handler.dtype
        else:
            self.compute_type = self._handler.compute_dtype

    def set_search_args(self, pre_clip_ratio, post_clip_ratio, smooth_alpha, smooth_type="smooth_quant"):
        self.pre_clip_ratio = pre_clip_ratio
        self.post_clip_ratio = post_clip_ratio
        self.smooth_alpha = smooth_alpha
        self.smooth_type = smooth_type

    def _weight_observer(self):
        self.observer_w_max = self.w_obs_max(self._handler.weight, self.observer_min_max_axis)[0]
        self.observer_w_min = self.w_obs_min(self._handler.weight, self.observer_min_max_axis)[0]
        self.quantizer_w_max = self.w_quant_max(self._handler.weight,
                                                self.weight_quantizer_min_max_axis, keepdims=True)[0]
        self.quantizer_w_min = self.w_quant_min(self._handler.weight,
                                                self.weight_quantizer_min_max_axis, keepdims=True)[0]

    def _calc_smooth_scale(self):
        """calc_smooth_scale"""
        if self.smooth_type == "smooth_quant":
            act_max = msops.maximum(msops.abs(self.x_obs_max(self.observer_x, 0)[0]),
                                    msops.abs(self.x_obs_min(self.observer_x, 0)[0]))
            input_max_pow = msops.pow(act_max, self.smooth_alpha)
            weight_max = msops.maximum(msops.abs(self.observer_w_min),
                                       msops.abs(self.observer_w_max))
            weight_max_pow = msops.pow(weight_max, 1 - self.smooth_alpha)
            smooth_scale = msops.div(input_max_pow, weight_max_pow).clamp(1e-5)
            # set 0 or nan to 1.0 to avoid quantization error
            smooth_scale[input_max_pow == 0] = 1.0
            smooth_scale[weight_max_pow == 0] = 1.0
        elif self.smooth_type == "awq":
            pass
        return smooth_scale

    def _clip_weight(self, weight, clip_ratio):
        """clip weight min max"""
        if self._weight_symmetric:
            weight_max = msops.maximum(msops.abs(self.quantizer_w_min), msops.abs(self.quantizer_w_max))
            self.quantizer_w_max = msops.mul(weight_max, clip_ratio)
            self.quantizer_w_min = msops.mul(-1, self.quantizer_w_max)
        else:
            self.quantizer_w_max = msops.mul(self.quantizer_w_max, clip_ratio)
            self.quantizer_w_min = msops.mul(self.quantizer_w_min, clip_ratio)
        return weight.clamp(self.quantizer_w_min, self.quantizer_w_max).astype(weight.dtype)

    def _fused_bias(self, quant_weight, act_offset, new_bias_need_allreduce=False):
        """compute fused bias"""
        if quant_weight is None:
            return None
        new_bias = -np.sum(act_offset.astype(np.int32) * quant_weight.asnumpy().astype(np.int32),
                           axis=1 if self._handler.transpose_b else 0).astype(np.int32)
        if new_bias_need_allreduce:
            t_new_bias = Tensor(new_bias)
            reduce_sum = msops.AllReduce(op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)
            t_new_bias = reduce_sum(t_new_bias)
            new_bias = t_new_bias.asnumpy()
        return new_bias

    #pylint: disable=protected-access
    def quant_weight(self):
        """quant weight"""
        self._weight_observer()
        weight = self._clip_weight(self._handler.weight, self.pre_clip_ratio)
        if self.cfg.act_quant_dtype == dtype.int8:
            if isinstance(self.observer_x, list):
                self.observer_x = msops.cat(tuple(self.observer_x))
            smooth_scale = self._calc_smooth_scale()
            if self._handler.transpose_b:
                weight = msops.mul(weight, smooth_scale)
            else:
                weight_scale = smooth_scale.expand_dims(1)
                weight = msops.mul(weight, weight_scale)
            self.quantizer_w_max = self.w_quant_max(weight, self.weight_quantizer_min_max_axis, keepdims=True)[0]
            self.quantizer_w_min = self.w_quant_min(weight, self.weight_quantizer_min_max_axis, keepdims=True)[0]
            weight = self._clip_weight(weight, self.post_clip_ratio)

            observer_x = msops.mul(self.observer_x, msops.div(1.0, smooth_scale))
            self.quantizer_x_max = self.x_quant_max(observer_x)[0]
            self.quantizer_x_min = self.x_quant_min(observer_x)[0]
            x_scale, x_zp = cal_quantization_params(self.quantizer_x_min.asnumpy(), self.quantizer_x_max.asnumpy(),
                                                    self.act_quant_min,
                                                    self.act_quant_max,
                                                    symmetric=self._act_symmetric)
        w_scale, w_zp = cal_quantization_params(self.quantizer_w_min.asnumpy(), self.quantizer_w_max.asnumpy(),
                                                self.weight_quant_min, self.weight_quant_max,
                                                symmetric=self._weight_symmetric)
        weight = quant_tensor_data(weight, w_scale.squeeze(), w_zp.squeeze(),
                                   self.weight_quant_min, self.weight_quant_max, self.weight_quantizer_axis)

        if self.cfg.act_quant_dtype == dtype.int8:
            dequant_scale = np.squeeze(w_scale * x_scale).astype(np.float32)
            bias = None
            if isinstance(self._handler, RowParallelLinear):
                bias = self._fused_bias(weight, x_zp, True)
            else:
                bias = self._fused_bias(weight, x_zp, False)
            bias = bias.astype(np.float64) * dequant_scale
            bias = Tensor(bias, dtype=self.compute_type) if bias is not None else None
            if self._handler.has_bias:
                bias = Tensor((bias.asnumpy() + self._handler.bias.asnumpy()), dtype=self.compute_type)
                bias_name = self._handler.bias.name
            else:
                bias_name = self._handler.weight.name + "_bias"
                self._handler.bias_add = P.Add()
            self._handler.has_bias = True
            self._handler.bias = Parameter(bias.astype(self.compute_type), name=bias_name)

            qmm = AllQuantMatmul(smooth_scale.asnumpy(), x_scale, x_zp, w_scale,
                                 transpose_b=self._handler.transpose_b)
        else:
            qmm = WeightQuantMatmul(w_scale, w_zp, bias=self._handler.bias,
                                    transpose_b=self._handler.transpose_b)

        self._handler.weight = Parameter(weight.astype(dtype.int8), name=self._handler.weight.name)
        self._handler.matmul = qmm
        self.observer_x = []
        self._handler.weight._offload()
        self.float_weight._offload()

    def quant_forward(self, inputs):
        float_weight = msops.deepcopy(self._handler.weight)
        self.quant_weight()
        output = self._handler(inputs)
        self._handler.weight = Parameter(float_weight.astype(dtype.float32), name=self._handler.weight.name)
        self._handler.matmul = self.mm
        return output

    def linear_act_observer(self, x):
        """act observer forward"""
        out_shape = P.Shape()(x)[:-1] + (self._handler.out_channels,)
        x = P.Reshape()(x, (-1, self._handler.in_channels))
        if hasattr(self._handler, "expert_flag") and self._handler.expert_flag:
            if self._handler.use_expert_group_size is True:
                x = P.Reshape()(x, (-1, self._handler.expert_num, self._handler.expert_group_size,
                                    self._handler.in_channels))
            else:
                x = P.Reshape()(x, (self._handler.outer_batch, self._handler.expert_num, -1, self._handler.in_channels))
        ori_dtype = F.dtype(x)
        weight = self._handler.cast(self._handler.weight, self._handler.dtype)
        x = self._handler.cast(x, self._handler.dtype)
        self.observer_x.append(x)
        x = self._handler.matmul(x, weight)
        if self._handler.has_bias:
            x = self._handler.bias_add(x, self._handler.cast(self._handler.bias, self._handler.dtype))
        if self._handler.activation_flag:
            x = self._handler.activation(x)
        x = F.cast(x, ori_dtype)
        output = self._handler.reshape(x, out_shape)
        return output

    def col_parallel_linear_act_observer(self, input_, weight=None):
        """act observer forward"""
        if weight is None and self._handler.skip_weight_param_allocation:
            raise ValueError("For ColumnParallelLinear, when skip_weight_param_allocation=True,"
                             " weight should be passed to construct(), but got None.")

        if self._handler.sequence_parallel or self._handler.explicit_expert_comm:
            input_parallel = input_
        else:
            input_parallel = self._handler.copy_to_mp_region(input_)

        origin_dtype = F.dtype(input_parallel)
        if self._handler.skip_weight_param_allocation:
            weight = self._handler.cast(weight, self._handler.compute_dtype)
        else:
            weight = self._handler.cast(self._handler.weight, self._handler.compute_dtype)
        input_parallel = self._handler.cast(input_parallel, self._handler.compute_dtype)

        if self._handler.sequence_parallel:
            input_parallel = input_parallel.swapaxes(0, 1).contiguous()
            input_parallel = self._handler.gather_from_sp_region(input_parallel)
            input_parallel = input_parallel.swapaxes(0, 1).contiguous()
        self.observer_x.append(msops.squeeze(input_parallel))
        output_parallel = self._handler.matmul(input_parallel, weight)
        if self._handler.has_bias:
            output_parallel = self._handler.bias_add(
                output_parallel, self._handler.cast(self._handler.bias, self._handler.compute_dtype)
            )
        output_parallel = self._handler.cast(output_parallel, origin_dtype)

        if self._handler.gather_output:
            output = self._handler.gather_from_mp_region(output_parallel)
        else:
            output = output_parallel
        return output

    def row_parallel_linear_act_observer(self, input_):
        """act observer forward"""
        if self._handler.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = self._handler.scatter_to_mp_region(input_)

        origin_dtype = F.dtype(input_parallel)
        weight = self._handler.cast(self._handler.weight, self._handler.compute_dtype)
        input_parallel = self._handler.cast(input_parallel, self._handler.compute_dtype)
        self.observer_x.append(msops.squeeze(input_parallel))
        output_parallel = self._handler.matmul(input_parallel, weight)
        if self._handler.explicit_expert_comm:
            output = output_parallel
        elif self._handler.sequence_parallel:
            output_parallel = output_parallel.swapaxes(0, 1).contiguous()
            output = self._handler.reduce_scatter_to_sp_region(output_parallel)
            output = output.swapaxes(0, 1).contiguous()
        else:
            output = self._handler.reduce_from_mp_region(output_parallel)

        if self._handler.has_bias:
            output = self._handler.bias_add(output, self._handler.cast(self._handler.bias, self._handler.compute_dtype))
        output = self._handler.cast(output, origin_dtype)
        return output

    def construct(self, inputs, weight=None):
        """
        Defines the computation of LinearWithMinMax to be performed.

        Returns:
            Tensor, returns the computed result.
        """
        if self._handler.infer_mode == "observer_x":
            if isinstance(self._handler, Linear):
                return self.linear_act_observer(inputs)
            if isinstance(self._handler, ColumnParallelLinear):
                return self.col_parallel_linear_act_observer(inputs, weight)
            return self.row_parallel_linear_act_observer(inputs)
        if self._handler.infer_mode == "float":
            return self.float_forward(inputs)
        return self.quant_forward(inputs)


class MatMul(Cell):
    """linear matmul"""

    def __init__(self, linear: Linear):
        super().__init__()
        self.matmul = linear.matmul

    def construct(self, x, weight):
        return self.matmul(x, weight)


class AllQuantMatmul(Cell):
    """quant act and weight"""

    def __init__(self, smooth_scale, input_scale, input_zp, weight_scale,
                 offset=None, transpose_a=False,
                 transpose_b=False, dst_dtype=dtype.float16):
        super().__init__()
        self.smooth_scale = smooth_scale
        self.transpose_b = transpose_b
        self.offset = offset
        dequant_scale = input_scale.astype(np.float32) * weight_scale.astype(np.float32)
        scale_i64 = NumpyQuantOps.trans_fp32_to_i64(np.squeeze(dequant_scale))
        self.dequant_scale = Parameter(Tensor(scale_i64, dtype=dtype.int64))

        t_scale = input_scale * smooth_scale
        self.input_scale = Parameter(Tensor(t_scale, dtype=dtype.float16))
        t_zp = np.array([input_zp] * len(t_scale)).astype(np.int8)
        self.input_zp = Parameter(Tensor(t_zp, dtype=dtype.int8))

        self.quant = QuantV2()
        self.qbmm = QuantBatchMatmul(transpose_x1=transpose_a, transpose_x2=transpose_b, dtype=dst_dtype)
        if offset is None:
            self.offset = None
        else:
            self.offset = Parameter(Tensor(offset, dtype=dtype.float32))

    def construct(self, x, quant_weight):
        # x: fp16 quant_weight: int8
        quant_weight = quant_weight.astype(dtype.int8)
        qx = self.quant(x, self.input_scale, self.input_zp, False, "ROUND", ms.int8)
        return self.qbmm(qx, quant_weight, self.dequant_scale, self.offset, None)


class WeightQuantMatmul(Cell):
    """quant batch matmul"""

    def __init__(self, t_scale, t_zp, bias=None, transpose_a=False, transpose_b=False, dst_type=dtype.float16):
        super().__init__()
        self.t_scale = t_scale
        self.t_zp = t_zp
        self.bias = bias
        self.dst_dtype = dst_type
        self.weight_qbmm = WeightQuantBatchMatmul(transpose_a, transpose_b)

    def construct(self, x, weight):
        """forward for antiquant bmm cell"""
        output = self.weight_qbmm(x, weight, self.t_scale, self.t_zp, None, None, self.bias)
        return output.astype(self.dst_dtype)

class SQLinearDeploy(Cell):
    """Linear deploy phase"""

    def __init__(self,
                 linear_name: str,
                 linear: Union[Linear, ColumnParallelLinear, RowParallelLinear],
                 cfg: InnerPTQConfig = None):
        super().__init__()
        if not isinstance(linear, (Linear, ColumnParallelLinear, RowParallelLinear)):
            raise ValueError("only Linear、ColumnParallelLinear、RowParallelLinear cell is supported,"
                             f"but {linear_name} type is {type(linear)}.")
        self.cfg = cfg
        self._handler = linear
        rank = len(linear.weight.shape)
        ic_axis = rank - 1 if linear.matmul.transpose_b else rank - 2
        self.weight_quantizer_axis = rank - 2 if linear.matmul.transpose_b else rank - 1
        self.ic = linear.weight.shape[ic_axis]
        self.oc = linear.weight.shape[self.weight_quantizer_axis]
        if isinstance(self._handler, Linear):
            self.compute_type = self._handler.dtype
        else:
            self.compute_type = self._handler.compute_dtype

        self._handler.weight = Parameter(initializer("ones", linear.weight.shape, dtype.int8), name=linear.weight.name)
        if linear.has_bias:
            self._handler.bias = Parameter(initializer("ones", linear.bias.shape, linear.bias.dtype),
                                           name=linear.bias.name)
        else:
            self._handler.has_bias = True
            bias_shape = [linear.weight.shape[0] if linear.transpose_b else linear.weight.shape[1]]
            bias_name = linear.weight.name + "_bias"
            self._handler.bias = Parameter(initializer("ones", bias_shape, self.compute_type), name=bias_name)
            self._handler.bias_add = P.Add()

        smooth_scale = np.array([1] * self.ic)
        x_scale = np.array(1.0)
        x_zp = np.array(1.0)
        w_scale = np.array([1] * self.oc)
        w_zp = np.array([0] * self.oc)

        if self.cfg.act_quant_dtype == dtype.int8:
            qmm = AllQuantMatmul(smooth_scale, x_scale, x_zp, w_scale, transpose_b=self._handler.transpose_b)
        else:
            qmm = WeightQuantMatmul(w_scale, w_zp, bias=self._handler.bias,
                                    transpose_b=self._handler.transpose_b)
        self._handler.matmul = qmm
        self.is_colparallel = (isinstance(self._handler, ColumnParallelLinear))
        self.is_linear = isinstance(self._handler, Linear)

    def construct(self, x, weight=None):
        """linear deploy construct"""
        if self.is_linear:
            return self._handler(x)
        out_shape = x.shape[:-1] + (self.oc,)
        x = msops.reshape(x, (-1, self.ic))
        if self.is_colparallel:
            x = self._handler(x, weight)
        else:
            x = self._handler(x)
        x = msops.reshape(x, out_shape)
        return x
