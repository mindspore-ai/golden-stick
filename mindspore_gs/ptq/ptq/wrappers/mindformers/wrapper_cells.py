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
"""ptq wrapper cells for mindformers."""
import abc

import numpy as np

from mindspore import Parameter, Tensor, dtype
from mindspore import ops as msops
from mindspore.nn import Cell
from mindspore.ops.operations.comm_ops import ReduceOp
from mindspore.communication.management import GlobalComm
from mindspore.ops.operations._infer_ops import QuantV2
from mindspore.ops.auto_generate import WeightQuantBatchMatmul, QuantBatchMatmul
from mindspore.common.initializer import initializer
from mindformers.modules.layers import Linear
from mindformers.experimental.distri_cores.tensor_parallel.layers import (
    ColumnParallelLinear, RowParallelLinear
)
from mindformers.experimental.distri_cores.tensor_parallel.collective_primitives import (
    MaxFromTensorParallelRegion, MinFromTensorParallelRegion
)
from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq_config import InnerPTQConfig, PTQMode
from mindspore_gs.ptq.ptq.wrapper_cell import WrapperCell
from mindspore_gs.ptq.network_helpers import LayerType, NetworkHelper
from mindspore_gs.quantization.quant_utils import get_quant_min_max, cal_quantization_params, quant_tensor_data
from mindspore_gs.common.numpy_quant_common import NumpyQuantOps


class WrapperLinearCell(WrapperCell, abc.ABC):
    """WrapperCell"""

    class MatmulCell(Cell):
        def __init__(self, matmul):
            super().__init__()
            self.mm = matmul

        def construct(self, *args, **kwargs):
            return self.mm(*args, **kwargs)

    def __init__(self, layer_name: str, layer, cfg: InnerPTQConfig, network_helper: NetworkHelper):
        super().__init__(layer_name, layer, cfg, network_helper)
        if self.cfg.mode == PTQMode.QUANTIZE:
            self._layer.matmul = WrapperLinearCell.MatmulCell(self._layer.matmul)

    def add_hook(self):
        def hook_fn(_, inps):
            x = inps[0]
            self.samples.append(msops.squeeze(x))
        self._layer.matmul.register_forward_pre_hook(hook_fn)

    def remove_hook(self):
        self._layer.matmul = WrapperLinearCell.MatmulCell(self._layer.matmul.mm)

    @abc.abstractmethod
    def deploy(self):
        raise NotImplementedError


class SmoothLinearCell(WrapperLinearCell):
    """SmoothLinearCell"""
    def __init__(self, linear_name, linear, cfg, network_helper):
        super().__init__(linear_name, linear, cfg, network_helper)
        if not isinstance(linear, (Linear, ColumnParallelLinear, RowParallelLinear)):
            raise ValueError("only Linear、ColumnParallelLinear、RowParallelLinear cell is supported,"
                             f"but {linear_name} type is {type(linear)}.")
        self.x_obs_max = msops.max
        self.x_obs_min = msops.min
        if isinstance(self.layer, (Linear, RowParallelLinear)):
            self.w_obs_max = msops.max
            self.w_obs_min = msops.min
        elif isinstance(self.layer, ColumnParallelLinear):
            self.w_obs_max = MaxFromTensorParallelRegion()
            self.w_obs_min = MinFromTensorParallelRegion()

    def _calc_smooth_scale(self, alpha):
        """_calc_smooth_scale"""
        act_max = msops.maximum(msops.abs(self.x_obs_max(self.cat_samples, 0)[0]),
                                msops.abs(self.x_obs_min(self.cat_samples, 0)[0]))
        input_max_pow = msops.pow(act_max, alpha)
        weight_smooth_minmax_axis = -2 if self.layer.transpose_b else -1
        weight_max = msops.maximum(msops.abs(self.w_obs_max(self.layer.weight, weight_smooth_minmax_axis)[0]),
                                   msops.abs(self.w_obs_min(self.layer.weight, weight_smooth_minmax_axis)[0]))
        weight_max_pow = msops.pow(weight_max, 1 - alpha)
        smooth_scale = msops.div(input_max_pow, weight_max_pow).clamp(1e-5)
        # set 0 or nan to 1.0 to avoid quantization error
        smooth_scale[input_max_pow == 0] = 1.0
        smooth_scale[weight_max_pow == 0] = 1.0
        return smooth_scale

    def _apply_weight_smooth(self, smooth_scale: Tensor):
        """_apply_weight_smooth"""
        # weight * scale
        weight_scale = msops.expand_dims(smooth_scale, 0)
        if not self._layer.transpose_b:
            weight_scale = weight_scale.transpose()
        orin_dtype = self._layer.weight.dtype
        weight = msops.mul(self._layer.weight, weight_scale)
        weight = self._layer.cast(weight, orin_dtype)
        msops.assign(self._layer.weight, weight)

    def _apply_act_smooth_to_pre_layer(self, smooth_scale: Tensor):
        """_apply_act_smooth_to_pre_layer"""
        pre_layer = self.net_helper.get_pre_layer(self._layer_name)
        # pre-weight / scale
        if not pre_layer:
            raise ValueError("Not support inserting mul in x for smooth now, please enable qkv_concat and "
                             "ffn_concat.")
        if pre_layer.type_ == LayerType.NORM_LAYER:
            orin_dtype = pre_layer.layer.weight.dtype
            norm_weight = msops.div(pre_layer.layer.weight, smooth_scale)
            norm_weight = msops.cast(norm_weight, orin_dtype)
            msops.assign(pre_layer.layer.weight, norm_weight)
        if pre_layer.type_ == LayerType.LINEAR_LAYER:
            if isinstance(pre_layer.layer, (Linear, ColumnParallelLinear, RowParallelLinear)):
                linear: Linear = pre_layer.layer
            elif isinstance(pre_layer.layer, SmoothLinearCell):
                sqlinear: SmoothLinearCell = pre_layer.layer
                linear: Linear = sqlinear.linear
            else:
                raise RuntimeError(f"Got unexpected linear layer, name: {pre_layer.name} {pre_layer.layer}.")
            if linear.transpose_b:
                # oc * ic
                pre_scale = msops.expand_dims(smooth_scale, 1)
            else:
                # ic * oc
                pre_scale = msops.expand_dims(smooth_scale, 0)
            orin_dtype = linear.weight.dtype
            weight = msops.div(linear.weight, pre_scale)
            weight = msops.cast(weight, orin_dtype)
            msops.assign(linear.weight, weight)
        if pre_layer.type_ == LayerType.CONCAT_LINEAR_LAYER:
            if isinstance(pre_layer.layer, (Linear, ColumnParallelLinear, RowParallelLinear)):
                linear: Linear = pre_layer.layer
            elif isinstance(pre_layer.layer, SmoothLinearCell):
                sqlinear: SmoothLinearCell = pre_layer.layer
                linear: Linear = sqlinear.linear
            else:
                raise RuntimeError(f"Got unexpected linear layer, name: {pre_layer.name} {pre_layer.layer}.")
            if linear.transpose_b:
                # oc * ic
                oc = linear.weight.shape[0]
                pre_scale = msops.pad(smooth_scale, [oc - smooth_scale.shape[0], 0], value=1)
                pre_scale = msops.expand_dims(pre_scale, 1)
            else:
                # ic * oc
                oc = linear.weight.shape[1]
                pre_scale = msops.pad(smooth_scale, [oc - smooth_scale.shape[0], 0], value=1)
                pre_scale = msops.expand_dims(pre_scale, 0)
            orin_dtype = linear.weight.dtype
            weight = msops.div(linear.weight, pre_scale)
            weight = msops.cast(weight, orin_dtype)
            msops.assign(linear.weight, weight)

    def _apply_act_smooth_by_insert_op(self, smooth_scale: Tensor):
        """_apply_act_smooth_by_insert_op"""
        class SmoothMatmul(Cell):
            def __init__(self, mm, smooth_scale_):
                super().__init__()
                self.mm = mm
                self.mul_scale = Parameter(smooth_scale_, name="mul_scale")

            def construct(self, x, weight):
                x = msops.div(x, self.mul_scale)
                return self.mm(x, weight)

        self._layer.matmul = SmoothMatmul(self._layer.matmul, smooth_scale)

    def _apply_act_smooth_by_insert_op_for_deploy(self, ic, compute_dtype):
        """_apply_act_smooth_by_insert_op_for_deploy"""
        class SmoothMatmul(Cell):
            def __init__(self, mm, ic_, compute_dtype_):
                super().__init__()
                self.mm = mm
                self.mul_scale = Parameter(initializer('ones', (ic_,), dtype=compute_dtype_), name="mul_scale")

            def construct(self, x, weight):
                x = msops.mul(x, self.mul_scale)
                return self.mm(x, weight)

        self._layer.matmul = SmoothMatmul(self._layer.matmul, ic, compute_dtype)

    def _apply_smooth(self, smooth_scale):
        """_apply_smooth"""
        if self.cfg.smooth_to_pre_layer:
            self._apply_act_smooth_to_pre_layer(smooth_scale)
        else:
            self._apply_act_smooth_by_insert_op(smooth_scale)
        self._apply_weight_smooth(smooth_scale)

    def smooth(self, alpha=0.5):
        """smooth"""
        smooth_scale = self._calc_smooth_scale(alpha)
        self._apply_smooth(smooth_scale)

    def process(self):
        if self.cfg.mode == PTQMode.QUANTIZE:
            super(SmoothLinearCell, self).process()
            self.smooth(self.cfg.algo_args.get('alpha', 0.5))
            return
        if not self.cfg.smooth_to_pre_layer:
            ic = self._layer.weight.shape[1] if self._layer.transpose_b else self._layer.weight.shape[1]
            self._apply_act_smooth_by_insert_op_for_deploy(ic, self._layer.dtype)


    def deploy(self):
        logger.info("Take back Linear from SmoothQuantLinearCell.")
        return self.layer


class QuantLinearCell(WrapperLinearCell):
    """QuantLinearCell"""
    def __init__(self, linear_name, linear, cfg, network_helper):
        super().__init__(linear_name, linear, cfg, network_helper)
        if not isinstance(linear, (Linear, ColumnParallelLinear, RowParallelLinear)):
            raise ValueError("only Linear、ColumnParallelLinear、RowParallelLinear cell is supported,"
                             f"but {linear_name} type is {type(linear)}.")
        self.post_clip_ratio = cfg.algo_args.get("post_clip_ratio", 1.0)
        rank = len(linear.weight.shape)
        ic_axis = rank - 1 if linear.transpose_b else rank - 2
        self.weight_quantizer_axis = rank - 2 if linear.transpose_b else rank - 1
        self.weight_quantizer_min_max_axis = 0 if self.weight_quantizer_axis else 1
        self.ic = linear.weight.shape[ic_axis]
        self.oc = linear.weight.shape[self.weight_quantizer_axis]

        self._weight_symmetric = cfg.weight_symmetric
        self._act_symmetric = cfg.act_symmetric
        if cfg.weight_quant_dtype == dtype.int8:
            self.weight_quant_min, self.weight_quant_max = get_quant_min_max(num_bits=8,
                                                                             signed=True,
                                                                             narrow_range=cfg.weight_narrow_range)
        if cfg.act_quant_dtype == dtype.int8:
            self.act_quant_min, self.act_quant_max = get_quant_min_max(num_bits=8,
                                                                       signed=True,
                                                                       narrow_range=cfg.act_narrow_range)

        self.quantizer_x_max = None
        self.quantizer_x_min = None
        self.quantizer_w_max = None
        self.quantizer_w_min = None
        if isinstance(self.layer, (Linear, ColumnParallelLinear)):
            self.x_quant_max = msops.max
            self.x_quant_min = msops.min
            self.w_quant_max = msops.max
            self.w_quant_min = msops.min
        elif isinstance(self.layer, RowParallelLinear):
            self.x_quant_max = MaxFromTensorParallelRegion()
            self.x_quant_min = MinFromTensorParallelRegion()
            self.w_quant_max = MaxFromTensorParallelRegion()
            self.w_quant_min = MinFromTensorParallelRegion()
        if isinstance(self.layer, Linear):
            self.compute_type = self.layer.dtype
        else:
            self.compute_type = self.layer.compute_dtype

        self.a16w8 = self.cfg.act_quant_dtype != dtype.int8 and self.cfg.weight_quant_dtype == dtype.int8
        self.a8w8 = self.cfg.act_quant_dtype == dtype.int8 and self.cfg.weight_quant_dtype == dtype.int8

    def _fused_bias(self, quant_weight, act_offset, new_bias_need_allreduce=False):
        """compute fused bias"""
        if quant_weight is None:
            return None
        new_bias = -np.sum(act_offset.astype(np.int32) * quant_weight.asnumpy().astype(np.int32),
                           axis=1 if self.layer.transpose_b else 0).astype(np.int32)
        if new_bias_need_allreduce:
            t_new_bias = Tensor(new_bias)
            reduce_sum = msops.AllReduce(op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)
            t_new_bias = reduce_sum(t_new_bias)
            new_bias = t_new_bias.asnumpy()
        return new_bias

    #pylint: disable=protected-access
    def quant_weight(self):
        """quant weight"""
        if self.a8w8:
            self.quantizer_x_max = self.x_quant_max(self.cat_samples)[0]
            self.quantizer_x_min = self.x_quant_min(self.cat_samples)[0]
            x_scale, x_zp = cal_quantization_params(self.quantizer_x_min.asnumpy(), self.quantizer_x_max.asnumpy(),
                                                    self.act_quant_min,
                                                    self.act_quant_max,
                                                    symmetric=self._act_symmetric)
        if self.a16w8 or self.a8w8:
            self.quantizer_w_max = self.w_quant_max(self.layer.weight, self.weight_quantizer_min_max_axis,
                                                    keepdims=True)[0]
            self.quantizer_w_min = self.w_quant_min(self.layer.weight, self.weight_quantizer_min_max_axis,
                                                    keepdims=True)[0]
            w_scale, w_zp = cal_quantization_params(self.quantizer_w_min.asnumpy(), self.quantizer_w_max.asnumpy(),
                                                    self.weight_quant_min, self.weight_quant_max,
                                                    symmetric=self._weight_symmetric)
            weight = quant_tensor_data(self.layer.weight, w_scale.squeeze(), w_zp.squeeze(),
                                       self.weight_quant_min, self.weight_quant_max, self.weight_quantizer_axis)
        if self.a8w8:
            dequant_scale = np.squeeze(w_scale * x_scale).astype(np.float32)
            bias = None
            if isinstance(self.layer, RowParallelLinear):
                bias = self._fused_bias(weight, x_zp, True)
            else:
                bias = self._fused_bias(weight, x_zp, False)
            bias = bias.astype(np.float64) * dequant_scale
            bias = Tensor(bias, dtype=self.compute_type) if bias is not None else None
            if self.layer.has_bias:
                bias = Tensor((bias.asnumpy() + self.layer.bias.asnumpy()), dtype=self.compute_type)
                bias_name = self.layer.bias.name
            else:
                bias_name = self.layer.weight.name + "_bias"
                self.layer.bias_add = msops.Add()
            self.layer.has_bias = True
            self.layer.bias = Parameter(bias.astype(self.compute_type), name=bias_name)
            # FIXME hangangqiang, decouple with smooth
            if self.cfg.outliers_suppression == "smooth" and not self.cfg.smooth_to_pre_layer:
                smooth_scale = self.layer.matmul.mm.mul_scale.asnumpy()
            else:
                smooth_scale = None
            qmm = AllQuantMatmul(x_scale, x_zp, w_scale, in_channels=self.ic, transpose_b=self.layer.transpose_b,
                                 smooth_scale=smooth_scale)
        if self.a16w8:
            qmm = WeightQuantMatmul(w_scale, w_zp, transpose_b=self.layer.transpose_b)
        if self.a8w8 or self.a16w8:
            self.layer.weight = Parameter(weight.astype(dtype.int8), name=self.layer.weight.name)
            self.layer.matmul = qmm

    def process(self):
        super(QuantLinearCell, self).process()
        self.quant_weight()
        self.layer.weight._offload()
        self.cat_samples = None

    def deploy(self):
        return self

    def add_hook(self):
        def hook_fn(_, inps):
            x = inps[0]
            self.samples.append(msops.squeeze(x))
        if self.cfg.outliers_suppression == "smooth" and not self.cfg.smooth_to_pre_layer:
            self._layer.matmul.mm.mm.register_forward_pre_hook(hook_fn)
        else:
            self._layer.matmul.register_forward_pre_hook(hook_fn)


class DeployLinearCell(WrapperLinearCell):
    """DeployLinearCell"""
    def __init__(self, linear_name, linear, cfg, network_helper=None):
        super().__init__(linear_name, linear, cfg, network_helper)
        if not isinstance(linear, (Linear, ColumnParallelLinear, RowParallelLinear)):
            raise ValueError("only Linear、ColumnParallelLinear、RowParallelLinear cell is supported,"
                             f"but {linear_name} type is {type(linear)}.")
        self.cfg = cfg
        rank = len(linear.weight.shape)
        ic_axis = rank - 1 if linear.matmul.transpose_b else rank - 2
        self.weight_quantizer_axis = rank - 2 if linear.matmul.transpose_b else rank - 1
        self.ic = linear.weight.shape[ic_axis]
        self.oc = linear.weight.shape[self.weight_quantizer_axis]
        if isinstance(self.layer, Linear):
            self.compute_type = self.layer.dtype
        else:
            self.compute_type = self.layer.compute_dtype
        self.layer.weight = Parameter(initializer("ones", linear.weight.shape, dtype.int8), name=linear.weight.name)
        w_scale = np.array([1] * self.oc)
        w_zp = np.array([0] * self.oc)
        if self.cfg.act_quant_dtype == dtype.int8 and self.cfg.weight_quant_dtype == dtype.int8:
            if linear.has_bias:
                self.layer.bias = Parameter(initializer("ones", linear.bias.shape, linear.bias.dtype),
                                            name=linear.bias.name)
            else:
                self.layer.has_bias = True
                bias_shape = [linear.weight.shape[0] if linear.transpose_b else linear.weight.shape[1]]
                bias_name = linear.weight.name + "_bias"
                self.layer.bias = Parameter(initializer("ones", bias_shape, self.compute_type), name=bias_name)
                self.layer.bias_add = msops.Add()
            x_scale = np.array(1.0)
            x_zp = np.array(1.0)
            qmm = AllQuantMatmul(x_scale, x_zp, w_scale, in_channels=self.ic, transpose_b=self.layer.transpose_b)
        elif self.cfg.act_quant_dtype != dtype.int8 and self.cfg.weight_quant_dtype == dtype.int8:
            qmm = WeightQuantMatmul(w_scale, w_zp, transpose_b=self.layer.transpose_b)
        self.layer.matmul = qmm
        self.is_rowparallel = (isinstance(self.layer, RowParallelLinear))
        self.is_colparallel = (isinstance(self.layer, ColumnParallelLinear))
        self.is_linear = isinstance(self.layer, Linear)

    def deploy(self):
        return self.layer

    def construct(self, x, **kwargs):
        """linear deploy construct"""
        if self.is_linear:
            return self.layer(x)
        out_shape = x.shape[:-1] + (self.oc,)
        x = msops.reshape(x, (-1, self.ic))
        if self.is_colparallel:
            x = self.layer(x, **kwargs)
        if self.is_rowparallel:
            x = self.layer(x)
        x = msops.reshape(x, out_shape)
        return x


class AllQuantMatmul(Cell):
    """quant act and weight"""

    def __init__(self, input_scale, input_zp, weight_scale, in_channels=None, out_channels=None, offset=None,
                 transpose_a=False, transpose_b=False, dst_dtype=dtype.float16, smooth_scale=None):
        super().__init__()
        self.transpose_b = transpose_b
        self.offset = offset
        self.ic = in_channels
        self.oc = out_channels
        dequant_scale = input_scale.astype(np.float32) * weight_scale.astype(np.float32)
        scale_i64 = NumpyQuantOps.trans_fp32_to_i64(np.squeeze(dequant_scale))
        self.dequant_scale = Parameter(Tensor(scale_i64, dtype=dtype.int64))
        if smooth_scale is not None:
            input_scale = (input_scale * smooth_scale).astype(np.float16)
        else:
            input_scale = np.array([input_scale] * in_channels).astype(np.float16)
        self.input_scale = Parameter(Tensor(input_scale, dtype=dtype.float16))
        input_zp = np.array([input_zp] * len(input_scale)).astype(np.int8)
        self.input_zp = Parameter(Tensor(input_zp, dtype=dtype.int8))

        self.quant = QuantV2()
        self.qbmm = QuantBatchMatmul(transpose_x1=transpose_a, transpose_x2=transpose_b, dtype=dst_dtype)
        if offset is None:
            self.offset = None
        else:
            self.offset = Parameter(Tensor(offset, dtype=dtype.float32))

    def construct(self, x, quant_weight):
        # x: fp16 quant_weight: int8
        quant_weight = quant_weight.astype(dtype.int8)
        qx = self.quant(x, self.input_scale, self.input_zp, False, "ROUND", dtype.int8)
        return self.qbmm(qx, quant_weight, self.dequant_scale, self.offset, None)


class WeightQuantMatmul(Cell):
    """quant batch matmul"""

    def __init__(self, t_scale, t_zp, transpose_a=False, transpose_b=False, dst_type=dtype.float16):
        super().__init__()
        self.dst_dtype = dst_type
        self.t_scale = Parameter(Tensor(np.squeeze(t_scale), dtype=self.dst_dtype))
        self.t_zp_neg = Parameter(Tensor(np.squeeze(t_zp) * -1, dtype=self.dst_dtype))
        self.weight_qbmm = WeightQuantBatchMatmul(transpose_a, transpose_b)

    def construct(self, x, weight):
        """forward for antiquant bmm cell"""
        weight = weight.astype(dtype.int8)
        output = self.weight_qbmm(x, weight, self.t_scale, self.t_zp_neg, None, None, None)
        return output.astype(self.dst_dtype)


class QuantPageAttentionMgrCell(WrapperCell):
    """QuantPageAttentionMgrCell"""

    def process(self):
        raise NotImplementedError

    def deploy(self):
        raise NotImplementedError
