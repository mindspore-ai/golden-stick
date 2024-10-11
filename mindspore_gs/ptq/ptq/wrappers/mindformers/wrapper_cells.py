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

from mindspore import nn, Parameter, Tensor, dtype
from mindspore import ops as msops
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer
from mindspore.nn import Cell
from mindspore.ops.operations.comm_ops import ReduceOp
from mindspore.communication.management import GlobalComm
from mindspore.ops.operations._infer_ops import QuantV2
from mindspore.ops.auto_generate import WeightQuantBatchMatmul, QuantBatchMatmul
from mindformers.modules.layers import Linear
from mindformers.modules.paged_attention_mgr import PagedAttentionMgr
from mindformers.experimental.infer.core.layers import ColumnParallelLinear, RowParallelLinear
from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_tensor_model_parallel_group, get_tensor_model_parallel_world_size)
from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq_config import InnerPTQConfig, PTQMode, OutliersSuppressionType
from mindspore_gs.ptq.convert_utils import QuantCellV2, AntiQuantCell
from mindspore_gs.ptq.ptq.wrapper_cell import WrapperCell
from mindspore_gs.ptq.network_helpers import LayerType, NetworkHelper
from mindspore_gs.quantization.quant_utils import (
    get_quant_min_max, cal_quantization_params,
    quant_tensor_data,
    convert_fp32_to_int64
)
from mindspore_gs.common.numpy_quant_common import NumpyQuantOps


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

    @staticmethod
    def is_enable(cfg: InnerPTQConfig):
        return cfg.outliers_suppression == OutliersSuppressionType.SMOOTH

    def __init__(self, linear_name, linear, cfg, network_helper):
        super().__init__(linear_name, linear, cfg, network_helper)
        if not isinstance(linear, (Linear, ColumnParallelLinear, RowParallelLinear)):
            raise ValueError("only Linear、ColumnParallelLinear、RowParallelLinear cell is supported,"
                             f"but {linear_name} type is {type(linear)}.")

        if self.cfg.mode == PTQMode.QUANTIZE and self.net_helper.get_spec("qkv_concat") is False:
            logger.info(f"qkv_concat is False, set smooth_to_pre_layer to False.")
            self.cfg.smooth_to_pre_layer = False

        self.x_obs_max = msops.max
        self.x_obs_min = msops.min
        if isinstance(self.layer, Linear):
            self.compute_type = self.layer.dtype
        else:
            self.compute_type = self.layer.compute_dtype
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
                x = msops.div(x, self.mul_scale)
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
        super(SmoothLinearCell, self).process()
        self.smooth(self.cfg.algo_args.get('alpha', 0.5))

    def deploy(self):
        logger.info("Take back Linear from SmoothQuantLinearCell.")
        if self.cfg.mode == PTQMode.QUANTIZE or self.cfg.smooth_to_pre_layer or \
            (self.cfg.act_quant_dtype == dtype.int8 and self.cfg.weight_quant_dtype == dtype.int8):
            return self.layer
        ic = self._layer.weight.shape[1] if self._layer.transpose_b else self._layer.weight.shape[1]
        self._apply_act_smooth_by_insert_op_for_deploy(ic, self.compute_type)
        return self.layer


class QuantLinearCell(WrapperLinearCell):
    """QuantLinearCell"""

    @staticmethod
    def is_enable(cfg: InnerPTQConfig):
        return cfg.weight_quant_dtype == dtype.int8

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
        self.input_scale = Parameter(initializer('ones', (self.ic), dtype.float16), name="input_scale")
        self.input_zp = Parameter(initializer('zeros', (self.ic), dtype.int8), name="input_zp")
        self.weight = Parameter(initializer("ones", linear.weight.shape, dtype.int8), name=linear.weight.name)
        self.w_scale = np.array([1] * self.oc)
        self.x_scale = np.array(1.0)
        self.w_zp = np.array([0] * self.oc)

        self.a16w8 = self.cfg.act_quant_dtype != dtype.int8 and self.cfg.weight_quant_dtype == dtype.int8
        self.a8w8 = self.cfg.act_quant_dtype == dtype.int8 and self.cfg.weight_quant_dtype == dtype.int8
        self.bias = None
        if self.a8w8:
            self.bias_name = self.layer.bias.name if self.layer.has_bias else linear.weight.name + "_bias"
            self.bias = Parameter(initializer("ones", (self.oc), self.compute_type), name=self.bias_name)

    def _fused_bias(self, quant_weight, act_offset, new_bias_need_allreduce=False):
        """compute fused bias"""
        if quant_weight is None:
            return None
        new_bias = -np.sum(act_offset.astype(np.int32) * quant_weight.asnumpy().astype(np.int32),
                           axis=1 if self.layer.transpose_b else 0).astype(np.int32)
        if new_bias_need_allreduce:
            t_new_bias = Tensor(new_bias)
            t_new_bias = msops.AllReduce(op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)(t_new_bias)
            new_bias = t_new_bias.asnumpy()
        return new_bias

    #pylint: disable=protected-access
    def quant_weight(self):
        """quant weight"""
        if not self.a16w8 and not self.a8w8:
            return
        self.quantizer_w_max = self.w_quant_max(self.layer.weight, self.weight_quantizer_min_max_axis,
                                                keepdims=True)[0]
        self.quantizer_w_min = self.w_quant_min(self.layer.weight, self.weight_quantizer_min_max_axis,
                                                keepdims=True)[0]
        w_scale, w_zp = cal_quantization_params(self.quantizer_w_min.asnumpy(), self.quantizer_w_max.asnumpy(),
                                                self.weight_quant_min, self.weight_quant_max,
                                                symmetric=self._weight_symmetric)
        weight = quant_tensor_data(self.layer.weight, w_scale.squeeze(), w_zp.squeeze(),
                                   self.weight_quant_min, self.weight_quant_max, self.weight_quantizer_axis)
        self.weight.set_data(Tensor(weight.asnumpy(), dtype=dtype.int8))
        self.w_scale = w_scale
        self.w_zp = w_zp
        if self.a8w8:
            self.quantizer_x_max = self.x_quant_max(self.cat_samples)[0]
            self.quantizer_x_min = self.x_quant_min(self.cat_samples)[0]
            x_scale, x_zp = cal_quantization_params(self.quantizer_x_min.asnumpy(), self.quantizer_x_max.asnumpy(),
                                                    self.act_quant_min,
                                                    self.act_quant_max,
                                                    symmetric=self._act_symmetric)
            self.x_scale = x_scale
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
            self.bias.set_data(bias.astype(self.compute_type))
            # FIXME hangangqiang, decouple with smooth
            if self.cfg.outliers_suppression == OutliersSuppressionType.SMOOTH and not self.cfg.smooth_to_pre_layer:
                smooth_scale = self.layer.matmul.mm.mul_scale.asnumpy()
            else:
                smooth_scale = None
            if smooth_scale is not None:
                input_scale = (x_scale * smooth_scale).astype(np.float16)
            else:
                input_scale = np.array([x_scale] * self.ic).astype(np.float16)
            self.input_scale.set_data(Tensor(input_scale, dtype=dtype.float16))
            input_zp = np.array([x_zp] * len(input_scale)).astype(np.int8)
            self.input_zp.set_data(Tensor(input_zp, dtype=dtype.int8))
    def process(self):
        super(QuantLinearCell, self).process()
        self.quant_weight()
        self.layer.weight._offload()
        self.cat_samples = None

    def deploy(self):
        return DeployLinearCell(self.layer, self.cfg, self.weight, self.bias, self.w_scale, self.w_zp,
                                self.x_scale, self.input_zp, self.input_scale)

    def add_hook(self):
        def hook_fn(_, inps):
            x = inps[0]
            self.samples.append(msops.squeeze(x))
        if self.cfg.outliers_suppression == OutliersSuppressionType.SMOOTH and not self.cfg.smooth_to_pre_layer:
            self._layer.matmul.mm.mm.register_forward_pre_hook(hook_fn)
        else:
            self._layer.matmul.register_forward_pre_hook(hook_fn)


class DeployLinearCell(Cell):
    """DeployLinearCell"""

    def __init__(self, linear, cfg, weight, bias, w_scale, w_zp, x_scale, input_zp, input_scale):
        super().__init__()
        self._layer = linear
        self.cfg = cfg
        self.quant = QuantV2()
        self.a8w8 = self.cfg.act_quant_dtype == dtype.int8 and self.cfg.weight_quant_dtype == dtype.int8
        self.a16w8 = self.cfg.act_quant_dtype != dtype.int8 and self.cfg.weight_quant_dtype == dtype.int8
        if self.a8w8:
            if linear.has_bias is False:
                self.layer.has_bias = True
                self.layer.bias_add = msops.Add()
            self.layer.bias = bias
            self.input_scale = input_scale
            self.input_zp = input_zp
            qmm = AllQuantMatmulDeploy(x_scale, w_scale, transpose_b=self.layer.transpose_b)
        elif self.a16w8:
            qmm = WeightQuantMatmul(w_scale, w_zp, transpose_b=self.layer.transpose_b)
        self.layer.weight = weight
        self.layer.matmul = qmm
        self.is_rowparallel = (isinstance(self.layer, RowParallelLinear))
        self.is_colparallel = (isinstance(self.layer, ColumnParallelLinear))
        self.is_linear = isinstance(self.layer, Linear)

    @property
    def layer(self):
        """layer"""
        return self._layer

    def linear_forward(self, x):
        """Forward process, x should be a tensor"""
        ori_dtype = F.dtype(x)
        x = self.layer.cast(x, self.layer.dtype)
        if self.a8w8:
            x = self.quant(x, self.input_scale, self.input_zp, False, "ROUND", dtype.int8)
        out_shape = self.layer.shape(x)[:-1] + (self.layer.out_channels,)
        if self.layer.expert_flag and not self.layer.use_gmm:
            if self.layer.use_expert_group_size is True:
                x = self.layer.reshape(x, (-1, self.layer.expert_num, self.layer.expert_group_size,
                                           self.layer.in_channels))
            else:
                x = self.layer.reshape(x, (self.layer.outer_batch, self.layer.expert_num, -1, self.layer.in_channels))
        x = self.layer.matmul(x, self.layer.weight)
        if self.layer.has_bias:
            x = self.layer.bias_add(x, self.layer.cast(self.layer.bias, self.layer.dtype))
        if self.layer.activation_flag:
            x = self.layer.activation(x)
        x = F.cast(x, ori_dtype)
        output = self.layer.reshape(x, out_shape)
        return output

    def col_linear_forward(self, input_parallel, weight=None):
        """
        Forward of ColumnParallelLinear.
        Performs a linear transformation considering various parallel modes and data type conversions.
        """

        if weight is None and self.layer.skip_weight_param_allocation:
            raise ValueError("For ColumnParallelLinear, when skip_weight_param_allocation=True,"
                             " weight should be passed to construct(), but got None.")

        origin_dtype = F.dtype(input_parallel)
        if not self.layer.skip_weight_param_allocation:
            weight = self.layer.weight
        input_parallel = self.layer.cast(input_parallel, self.layer.compute_dtype)

        if self.layer.sequence_parallel:
            input_parallel = input_parallel.swapaxes(0, 1).contiguous()
            input_parallel = self.layer.gather_from_sp_region(input_parallel)
            input_parallel = input_parallel.swapaxes(0, 1).contiguous()
        if self.a8w8:
            input_parallel = self.quant(input_parallel, self.input_scale, self.input_zp, False, "ROUND", dtype.int8)
        output_shape = self.layer.shape(input_parallel)[:-1] + (self.layer.output_size_per_partition,)
        input_parallel = self.layer.reshape(input_parallel, (-1, self.layer.input_size))
        output_parallel = self.layer.matmul(input_parallel, weight)
        if self.layer.has_bias:
            output_parallel = self.layer.bias_add(
                output_parallel, self.layer.cast(self.layer.bias, self.layer.compute_dtype)
            )
        output_parallel = self.layer.cast(output_parallel, origin_dtype)
        output_parallel = self.layer.reshape(output_parallel, output_shape)

        if self.layer.gather_output:
            output = self.layer.gather_from_mp_region(output_parallel)
        else:
            output = output_parallel
        return output

    def row_linear_forward(self, input_):
        """
        Forward of RowParallelLinear.
        Performs a linear transformation considering various parallel modes and data type conversions.
        """

        if self.layer.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = self.layer.scatter_to_mp_region(input_)

        origin_dtype = F.dtype(input_parallel)
        input_parallel = self.layer.cast(input_parallel, self.layer.compute_dtype)
        if self.a8w8:
            input_parallel = self.quant(input_parallel, self.input_scale, self.input_zp, False, "ROUND", dtype.int8)
        output_shape = self.layer.shape(input_parallel)[:-1] + (self.layer.output_size,)
        input_parallel = self.layer.reshape(input_parallel, (-1, self.layer.input_size_per_partition))
        output_parallel = self.layer.matmul(input_parallel, self.layer.weight)

        if self.layer.sequence_parallel:
            output_parallel = output_parallel.swapaxes(0, 1).contiguous()
            output = self.layer.reduce_scatter_to_sp_region(output_parallel)
            output = output.swapaxes(0, 1).contiguous()
        else:
            output = self.layer.reduce_from_mp_region(output_parallel)

        if self.layer.has_bias:
            output = self.layer.bias_add(output, self.layer.cast(self.layer.bias, self.layer.compute_dtype))
        output = self.layer.cast(output, origin_dtype)
        output = self.layer.reshape(output, output_shape)
        return output

    def construct(self, x, *args, **kwargs):
        """linear deploy construct"""
        if self.is_linear:
            return self.linear_forward(x)
        if self.is_colparallel:
            x = self.col_linear_forward(x, *args, **kwargs)
        if self.is_rowparallel:
            x = self.row_linear_forward(x)
        return x

    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        state_dict = {}
        if self.is_colparallel:
            w_shard = (self.layer.tensor_parallel_group_size, 1) if self.layer.transpose_b \
                  else (1, self.layer.tensor_parallel_group_size)
            if self.layer.has_bias:
                state_dict[self.layer.bias.name] = {'shape': self.layer.bias.shape,
                                                    'shard': (self.layer.tensor_parallel_group_size,)}
        elif self.is_rowparallel:
            w_shard = (1, self.layer.tensor_parallel_group_size) if self.layer.transpose_b \
                  else (self.layer.tensor_parallel_group_size, 1)
            if self.layer.has_bias:
                state_dict[self.layer.bias.name] = {'shape': self.layer.bias.shape, 'shard': (1,)}
        state_dict[self.layer.weight.name] = {'shape': self.layer.weight.shape, 'shard': w_shard}

        if self.a8w8:
            if self.is_colparallel:
                input_scale_shard = (1,)
                input_zp_shard = (1,)
                dequant_scale_shard = (self.layer.tensor_parallel_group_size,)
            elif self.is_rowparallel:
                input_scale_shard = (self.layer.tensor_parallel_group_size,)
                input_zp_shard = (self.layer.tensor_parallel_group_size,)
                dequant_scale_shard = (1,)
            state_dict[self.input_scale.name] = {'shape': self.input_scale.shape, 'shard': input_scale_shard}
            state_dict[self.input_zp.name] = {'shape': self.input_zp.shape, 'shard': input_zp_shard}
            state_dict[self.layer.matmul.dequant_scale.name] = {'shape': self.layer.matmul.dequant_scale.shape,
                                                                'shard': dequant_scale_shard}
        else:
            if self.is_colparallel:
                t_scale_shard = (self.layer.tensor_parallel_group_size,)
                t_zp_shard = {self.layer.tensor_parallel_group_size}
            elif self.is_rowparallel:
                t_scale_shard = (1,)
                t_zp_shard = (1,)
            state_dict[self.layer.matmul.t_scale.name] = {'shape': self.layer.matmul.t_scale.shape,
                                                          'shard': t_scale_shard}
            state_dict[self.layer.matmul.t_zp_neg.name] = {'shape': self.layer.matmul.t_zp_neg.shape,
                                                           'shard': t_zp_shard}
        return state_dict


class WeightQuantMatmul(Cell):
    """quant batch matmul"""

    def __init__(self, t_scale, t_zp, transpose_a=False, transpose_b=False, dst_type=dtype.float16):
        super().__init__()
        self.dst_dtype = dst_type
        self.t_scale = Parameter(Tensor(np.squeeze(t_scale), dtype=self.dst_dtype), name="t_scale")
        self.t_zp_neg = Parameter(Tensor(np.squeeze(t_zp) * -1, dtype=self.dst_dtype), name="t_zp_neg")
        self.weight_qbmm = WeightQuantBatchMatmul(transpose_a, transpose_b)

    def construct(self, x, weight):
        """forward for antiquant bmm cell"""
        output = self.weight_qbmm(x, weight, self.t_scale, self.t_zp_neg, None, None, None)
        return output.astype(self.dst_dtype)


class AllQuantMatmulDeploy(Cell):
    """quant weight"""

    def __init__(self, input_scale, weight_scale, offset=None,
                 transpose_a=False, transpose_b=False, dst_dtype=dtype.float16):
        super().__init__()
        self.transpose_b = transpose_b
        self.offset = offset
        dequant_scale = input_scale.astype(np.float32) * weight_scale.astype(np.float32)
        scale_i64 = NumpyQuantOps.trans_fp32_to_i64(np.squeeze(dequant_scale))
        self.dequant_scale = Parameter(Tensor(scale_i64, dtype=dtype.int64), name='dequant_scale')
        self.qbmm = QuantBatchMatmul(transpose_x1=transpose_a, transpose_x2=transpose_b, dtype=dst_dtype)
        if offset is None:
            self.offset = None
        else:
            self.offset = Parameter(Tensor(offset, dtype=dtype.float32))

    def construct(self, qx, quant_weight):
        # x: fp16 quant_weight: int8
        return self.qbmm(qx, quant_weight, self.dequant_scale, self.offset, None)


class QuantPageAttentionMgrCell(WrapperCell):
    """QuantPageAttentionMgrCell"""

    @staticmethod
    def is_enable(cfg: InnerPTQConfig):
        return cfg.kvcache_quant_dtype == dtype.int8

    def __init__(self, linear_name, layer, cfg, network_helper):
        super().__init__(linear_name, layer, cfg, network_helper)
        self.key_samples = []
        self.value_samples = []
        self.quantizer_key_max = None
        self.quantizer_key_min = None
        self.quantizer_value_max = None
        self.quantizer_value_min = None
        self.kvcache_symmetric = cfg.kvcache_symmetric
        n = layer.n_kv_heads
        d = layer.head_dim
        ic = n * d
        self.key_t_scale = Parameter(initializer('ones', (ic), dtype.float16), name="key_t_scale")
        self.key_t_zp = Parameter(initializer('ones', (ic), dtype.float16), name="key_t_zp")
        self.value_t_scale = Parameter(initializer('ones', (ic), dtype.float16), name="value_t_scale")
        self.value_t_zp = Parameter(initializer('ones', (ic), dtype.float16), name="value_t_zp")
        self.key_value_t_scale = Parameter(initializer('ones', (2, ic), dtype.int64), name="key_value_t_scale")
        self.key_value_t_zp = Parameter(initializer('ones', (2, ic), dtype.int32), name="key_value_t_zp")

        self.kvcache_quant_min, self.kvcache_quant_max = get_quant_min_max(num_bits=8,
                                                                           signed=True,
                                                                           narrow_range=cfg.kvcache_narrow_range)
    def process(self):
        if not self.key_samples or not self.value_samples:
            raise RuntimeError("Please catch ReshapeAndCache inputs before quantization.")
        key_cat_samples = msops.cat(tuple(self.key_samples), axis=0)
        self.quantizer_key_max = msops.max(key_cat_samples, 0)[0]
        self.quantizer_key_min = msops.min(key_cat_samples, 0)[0]
        key_t_scale, key_t_zp = cal_quantization_params(self.quantizer_key_min.asnumpy(),
                                                        self.quantizer_key_max.asnumpy(),
                                                        self.kvcache_quant_min,
                                                        self.kvcache_quant_max,
                                                        symmetric=self.kvcache_symmetric)
        value_cat_samples = msops.cat(tuple(self.value_samples), axis=0)
        self.quantizer_value_max = msops.max(value_cat_samples, 0)[0]
        self.quantizer_value_min = msops.min(value_cat_samples, 0)[0]
        value_t_scale, value_t_zp = cal_quantization_params(self.quantizer_value_min.asnumpy(),
                                                            self.quantizer_value_max.asnumpy(),
                                                            self.kvcache_quant_min,
                                                            self.kvcache_quant_max,
                                                            symmetric=self.kvcache_symmetric)
        key_t_scale = np.squeeze(key_t_scale).astype(np.float16)
        key_t_zp = np.squeeze(key_t_zp).astype(np.float16)
        value_t_scale = np.squeeze(value_t_scale).astype(np.float16)
        value_t_zp = np.squeeze(value_t_zp).astype(np.float16)
        self.key_t_scale.set_data(Tensor(key_t_scale, dtype=dtype.float16))
        self.key_t_zp.set_data(Tensor(key_t_zp, dtype=dtype.float16))
        self.value_t_scale.set_data(Tensor(value_t_scale, dtype=dtype.float16))
        self.value_t_zp.set_data(Tensor(value_t_zp, dtype=dtype.float16))

        t_scale_len = self.key_t_scale.shape[0]
        key_t_scale = convert_fp32_to_int64(key_t_scale.astype(np.float32))
        value_t_scale = convert_fp32_to_int64(value_t_scale.astype(np.float32))
        key_value_t_scale = np.concatenate((key_t_scale.reshape((1, t_scale_len)),
                                            value_t_scale.reshape((1, t_scale_len))))

        t_zp_len = self.value_t_zp.shape[0]
        key_t_zp = (key_t_zp*-1).astype(np.int32)
        value_t_zp = (value_t_zp*-1).astype(np.int32)
        key_value_t_zp = np.concatenate((key_t_zp.reshape((1, t_zp_len)), value_t_zp.reshape((1, t_zp_len))))

        self.key_value_t_scale.set_data(Tensor(key_value_t_scale, dtype=dtype.int64))
        self.key_value_t_zp.set_data(Tensor(key_value_t_zp, dtype=dtype.int32))
        self.key_samples.clear()
        self.value_samples.clear()

    def deploy(self):
        return DeployPageAttentionMgrCell(self.layer, self.value_t_scale, self.value_t_zp, self.key_t_scale,
                                          self.key_t_zp, self.key_value_t_scale, self.key_value_t_zp, self.cfg)

    def add_hook(self):
        pass

    def remove_hook(self):
        pass

    def construct(self, x, *args, **kwargs):
        value = args[0]
        self.key_samples.append(msops.squeeze(x))
        self.value_samples.append(msops.squeeze(value))
        slot_mapping = args[1]
        self.layer.reshape_and_cache(x, value, self.layer.key_cache, self.layer.value_cache, slot_mapping)

    # pylint: disable=W0613
    def paged_attn(self, query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=None):
        """The forward compute of Paged Attention."""
        return self.layer.paged_attention(query, self.layer.key_cache, self.layer.value_cache, block_tables,
                                          batch_valid_length)

    def paged_attn_with_alibi(self, query, batch_valid_length, block_tables, alibi_tensor):
        """The forward compute of KVCache for Paged Attention with alibi tensor."""
        return self.layer.paged_attention_with_alibi(query, self.layer.key_cache, self.layer.value_cache,
                                                     block_tables, batch_valid_length, alibi_tensor)


class DeployPageAttentionMgrCell(Cell):
    """DeployPageAttentionMgrCell"""

    def __init__(self, kvcache: PagedAttentionMgr, value_t_scale, value_t_zp, key_t_scale, key_t_zp,
                 key_value_t_scale, key_value_t_zp, cfg: InnerPTQConfig):
        super().__init__()
        self.layer = kvcache
        self.enable_deploy_fusion = cfg.enable_deploy_fusion
        self._key_input_quantizer = QuantCellV2(Tensor(key_t_scale.asnumpy(), dtype=dtype.float16),
                                                Tensor(key_t_zp.asnumpy().astype(np.int8), dtype=dtype.int8))
        self._value_input_quantizer = QuantCellV2(Tensor(value_t_scale.asnumpy(), dtype=dtype.float16),
                                                  Tensor(value_t_zp.asnumpy().astype(np.int8), dtype=dtype.int8))
        dst_type = self.layer.key_cache.dtype
        n = kvcache.n_kv_heads
        d = kvcache.head_dim
        self._key_output_quantizer = AntiQuantCell(n, d, dst_type)
        self._value_output_quantizer = AntiQuantCell(n, d, dst_type)
        if cfg.mode == PTQMode.QUANTIZE or not self.enable_deploy_fusion:
            self.key_t_zp = key_t_zp
            self.value_t_zp = value_t_zp
            self.key_t_scale = key_t_scale
            self.value_t_scale = value_t_scale
        if cfg.mode == PTQMode.QUANTIZE or self.enable_deploy_fusion:
            self.key_value_t_scale = key_value_t_scale
            self.key_value_t_zp = key_value_t_zp

        self.layer.key_cache = Parameter(initializer('ones', self.layer.key_cache.shape, dtype.int8),
                                         name=self.layer.key_cache.name, requires_grad=False)
        self.layer.value_cache = Parameter(initializer('ones', self.layer.value_cache.shape, dtype.int8),
                                           name=self.layer.value_cache.name, requires_grad=False)
        self.tensor_parallel_group_size = get_tensor_model_parallel_world_size()

    def construct(self, key, value, slot_mapping):
        """The forward compute of KVCache for Paged Attention."""
        key = self._key_input_quantizer(key)
        value = self._value_input_quantizer(value)
        self.layer.reshape_and_cache(key, value, self.layer.key_cache, self.layer.value_cache, slot_mapping)

    # pylint: disable=W0613
    def paged_attn(self, query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=None):
        """The forward compute of Paged Attention."""
        if not self.enable_deploy_fusion:
            kcache = self._key_output_quantizer(self.layer.key_cache, self.key_t_zp, self.key_t_scale)
            vcache = self._value_output_quantizer(self.layer.value_cache, self.value_t_zp, self.value_t_scale)
            return self.layer.paged_attention(query, kcache, vcache, block_tables, batch_valid_length)
        return self.layer.paged_attention(query, self.layer.key_cache, self.layer.value_cache, block_tables,
                                          batch_valid_length, self.key_value_t_scale, self.key_value_t_zp)

    def paged_attn_with_alibi(self, query, batch_valid_length, block_tables, alibi_tensor):
        """The forward compute of Paged Attention."""
        if not self.enable_deploy_fusion:
            kcache = self._key_output_quantizer(self.layer.key_cache, self.key_t_zp, self.key_t_scale)
            vcache = self._value_output_quantizer(self.layer.value_cache, self.value_t_zp, self.value_t_scale)
            return self.layer.paged_attention_with_alibi(query, kcache, vcache, block_tables, batch_valid_length,
                                                         alibi_tensor)
        return self.layer.paged_attention_with_alibi(query, self.layer.key_cache, self.layer.value_cache,
                                                     block_tables, batch_valid_length, self.key_value_t_scale,
                                                     self.key_value_t_zp, alibi_tensor)

    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        state_dict = {}
        if self.enable_deploy_fusion:
            key_value_t_scale_shard = (1, self.tensor_parallel_group_size)
            key_value_t_zp_shard = (1, self.tensor_parallel_group_size)
            state_dict[self.key_value_t_scale.name] = {'shape': self.key_value_t_scale.shape,
                                                       'shard': key_value_t_scale_shard}
            state_dict[self.key_value_t_zp.name] = {'shape': self.key_value_t_zp.shape,
                                                    'shard': key_value_t_zp_shard}
        else:
            key_t_scale_shard = (self.tensor_parallel_group_size,)
            key_t_zp_shard = (self.tensor_parallel_group_size,)

            value_t_scale_shard = (self.tensor_parallel_group_size,)
            value_t_zp_shard = (self.tensor_parallel_group_size,)

            state_dict[self.key_t_scale.name] = {'shape': self.key_t_scale.shape,
                                                 'shard': key_t_scale_shard}
            state_dict[self.key_t_zp.name] = {'shape': self.key_t_zp.shape,
                                              'shard': key_t_zp_shard}
            state_dict[self.value_t_scale.name] = {'shape': self.value_t_scale.shape,
                                                   'shard': value_t_scale_shard}
            state_dict[self.value_t_zp.name] = {'shape': self.value_t_zp.shape,
                                                'shard': value_t_zp_shard}
        state_dict = self.sharded_input_quantizer_state_dict(state_dict)
        return state_dict

    def sharded_input_quantizer_state_dict(self, state_dict):
        """provide the sharded state dict based on the config"""

        key_input_quantizer_t_scale_shard = (self.tensor_parallel_group_size,)
        key_input_quantizer_t_zp_shard = (self.tensor_parallel_group_size,)
        value_input_quantizer_t_scale_shard = (self.tensor_parallel_group_size,)
        value_input_quantizer_t_zp_shard = (self.tensor_parallel_group_size,)

        state_dict[self._key_input_quantizer.t_scale.name] = {'shape': self._key_input_quantizer.t_scale.shape,
                                                              'shard': key_input_quantizer_t_scale_shard}
        state_dict[self._key_input_quantizer.t_zp.name] = {'shape': self._key_input_quantizer.t_zp.shape,
                                                           'shard': key_input_quantizer_t_zp_shard}
        state_dict[self._value_input_quantizer.t_scale.name] = {'shape': self._value_input_quantizer.t_scale.shape,
                                                                'shard': value_input_quantizer_t_scale_shard}
        state_dict[self._value_input_quantizer.t_zp.name] = {'shape': self._value_input_quantizer.t_zp.shape,
                                                             'shard': value_input_quantizer_t_zp_shard}
        return state_dict
