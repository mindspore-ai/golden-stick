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
from mindformers.modules.layers import Linear

from mindspore_gs.ptq.ptq_config import InnerPTQConfig
from mindspore_gs.quantization.quant_utils import (
    get_quant_min_max, cal_tensor_quantization_params,
    quant_tensor_data, quant_bias_data)
from mindspore_gs.common.numpy_quant_common import NumpyQuantOps


class MinMaxLinearWrapper(Cell):
    """Linear layer wrapper with min max"""

    def __init__(self, linear_name: str, linear: Linear, cfg: InnerPTQConfig = None):
        super().__init__()
        if not isinstance(linear, Linear):
            raise ValueError(f'only Linear cell is supported, but {linear_name} type is {type(linear)}.')
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

    def construct(self, inputs):
        # for deploy forward
        raise NotImplementedError


class SQLinearWrapper(MinMaxLinearWrapper):
    """Linear layer wrapper with min max"""

    def __init__(self, linear_name: str, linear: Linear, cfg: InnerPTQConfig = None):
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

    def set_search_args(self, pre_clip_ratio, post_clip_ratio, smooth_alpha, smooth_type="smooth_quant"):
        self.pre_clip_ratio = pre_clip_ratio
        self.post_clip_ratio = post_clip_ratio
        self.smooth_alpha = smooth_alpha
        self.smooth_type = smooth_type

    def _weight_observer(self):
        self.observer_w_max = msops.max(self._handler.weight, self.observer_min_max_axis)[0]
        self.observer_w_min = msops.min(self._handler.weight, self.observer_min_max_axis)[0]
        self.quantizer_w_max = msops.max(self._handler.weight, self.weight_quantizer_min_max_axis, keepdims=True)[0]
        self.quantizer_w_min = msops.min(self._handler.weight, self.weight_quantizer_min_max_axis, keepdims=True)[0]

    def _calc_smooth_scale(self):
        """calc_smooth_scale"""
        if self.smooth_type == "smooth_quant":
            act_max = msops.maximum(msops.abs(msops.min(self.observer_x, 0)[0]),
                                    msops.abs(msops.max(self.observer_x, 0)[0]))
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
            self.quantizer_w_max = msops.max(weight, self.weight_quantizer_min_max_axis, keepdims=True)[0]
            self.quantizer_w_min = msops.min(weight, self.weight_quantizer_min_max_axis, keepdims=True)[0]
            weight = self._clip_weight(weight, self.post_clip_ratio)

            observer_x = msops.mul(self.observer_x, msops.div(1.0, smooth_scale))
            self.quantizer_x_max = msops.max(observer_x)[0]
            self.quantizer_x_min = msops.min(observer_x)[0]
            x_scale, x_zp = cal_tensor_quantization_params(self.quantizer_x_min, self.quantizer_x_max,
                                                           self.act_quant_min,
                                                           self.act_quant_max,
                                                           symmetric=self._act_symmetric)
        w_scale, w_zp = cal_tensor_quantization_params(self.quantizer_w_min, self.quantizer_w_max,
                                                       self.weight_quant_min, self.weight_quant_max,
                                                       symmetric=self._weight_symmetric)
        weight = quant_tensor_data(weight, w_scale.asnumpy().squeeze(), w_zp.asnumpy().squeeze(),
                                   self.weight_quant_min, self.weight_quant_max, self.weight_quantizer_axis)
        if self.cfg.act_quant_dtype == dtype.int8:
            quant_bias = None
            if self._handler.has_bias:
                quant_bias = quant_bias_data(self._handler.bias, msops.mul(w_scale, x_scale).asnumpy())
            qmm = AllQuantMatmul(smooth_scale, x_scale, x_zp, w_scale,
                                 weight, quant_bias, transpose_b=self._handler.transpose_b)
        else:
            qmm = WeightQuantMatmul(w_scale, w_zp, bias=self._handler.bias,
                                    transpose_b=self._handler.transpose_b)
        self._handler.has_bias = False
        self._handler.bias = None
        self._handler.weight = Parameter(weight.astype(dtype.int8), name=self._handler.weight.name)
        self._handler.matmul = qmm

    def quant_forward(self, inputs):
        float_weight = msops.deepcopy(self._handler.weight)
        self.quant_weight()
        output = self._handler(inputs)
        self._handler.weight = Parameter(float_weight.astype(dtype.float32), name=self._handler.weight.name)
        self._handler.matmul = self.mm
        return output

    def act_observer(self, x):
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

    def construct(self, inputs):
        """
        Defines the computation of LinearWithMinMax to be performed.

        Returns:
            Tensor, returns the computed result.
        """
        if self._handler.infer_mode == "observer_x":
            self.observer_x = []
            return self.act_observer(inputs)
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
                 quant_weight, quant_bias, offset=None, transpose_a=False,
                 transpose_b=False, dst_dtype=dtype.float16):
        super().__init__()
        self.smooth_scale = smooth_scale
        self.transpose_b = transpose_b
        self.offset = offset
        dequant_scale = np.array(input_scale, dtype=np.float32) * np.array(weight_scale, dtype=np.float32)
        scale_i64 = NumpyQuantOps.trans_fp32_to_i64(dequant_scale)
        self.dequant_scale = Parameter(Tensor(np.squeeze(scale_i64), dtype=dtype.int64))
        self.input_scale = Parameter(msops.mul(input_scale, smooth_scale).astype(dtype.float16))
        self.input_zp = Parameter(msops.mul(msops.ones_like(self.input_scale), input_zp).astype(dtype.int8))
        self.new_bias = Parameter(self._fused_bias(quant_weight, quant_bias, input_zp))
        self.quant = QuantV2()
        self.qbmm = QuantBatchMatmul(transpose_x1=transpose_a, transpose_x2=transpose_b, dtype=dst_dtype)
        if offset is None:
            self.offset = None
        else:
            self.offset = Parameter(Tensor(offset, dtype=dtype.float32))

    def _fused_bias(self, quant_weight, quant_bias, act_offset):
        if quant_weight is None:
            return None
        new_bias = -np.sum(act_offset.asnumpy().astype(np.float32) * quant_weight.asnumpy(),
                           axis=1 if self.transpose_b else 0).astype(np.int32)
        if quant_bias is not None:
            new_bias = quant_bias.asnumpy().astype(np.int32) + new_bias
        new_bias = Tensor(new_bias, dtype=dtype.int32) if new_bias is not None else None
        return new_bias

    def construct(self, x, quant_weight):
        # x: fp16 quant_weight: int8
        quant_weight = quant_weight.astype(dtype.int8)
        qx = self.quant(x, self.input_scale, self.input_zp, False, "ROUND", ms.int8)
        return self.qbmm(qx, quant_weight, self.dequant_scale, self.offset, self.new_bias)


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

    def __init__(self, linear_name: str, linear: Linear, cfg: InnerPTQConfig = None):
        super().__init__()
        if not isinstance(linear, Linear):
            raise ValueError(f'only Linear cell is supported, but {linear_name} type is {type(linear)}.')
        self.cfg = cfg
        self._handler = linear
        rank = len(linear.weight.shape)
        ic_axis = rank - 1 if linear.matmul.transpose_b else rank - 2
        self.weight_quantizer_axis = rank - 2 if linear.matmul.transpose_b else rank - 1
        self.ic = linear.weight.shape[ic_axis]
        self.oc = linear.weight.shape[self.weight_quantizer_axis]

        self._handler.weight = Parameter(initializer("ones", linear.weight.shape, dtype.int8), name=linear.weight.name)

        smooth_scale = Tensor([1] * self.ic)
        x_scale = Tensor(1.0)
        x_zp = Tensor(1.0)
        w_scale = [1] * self.oc
        w_zp = [0] * self.oc
        weight = msops.ones(linear.weight.shape)
        quant_bias = None

        if self.cfg.act_quant_dtype == dtype.int8:
            qmm = AllQuantMatmul(smooth_scale, x_scale, x_zp, w_scale,
                                 weight, quant_bias, transpose_b=self._handler.transpose_b)
            self._handler.has_bias = False
            self._handler.bias = None
        else:
            qmm = WeightQuantMatmul(w_scale, w_zp, bias=self._handler.bias,
                                    transpose_b=self._handler.transpose_b)
        self._handler.matmul = qmm

    def construct(self, x):
        return self._handler(x)
