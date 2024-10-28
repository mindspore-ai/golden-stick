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

import numpy as np

from mindspore import Parameter, Tensor, dtype
from mindspore import ops as msops
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer
from mindspore.nn import Cell
from mindspore.ops.operations.comm_ops import ReduceOp
from mindspore.communication.management import GlobalComm
from mindspore.ops.operations._infer_ops import QuantV2
from mindspore.ops.auto_generate import WeightQuantBatchMatmul, QuantBatchMatmul, DynamicQuantExt

from mindformers.modules.layers import Linear
from mindformers.modules.paged_attention_mgr import PagedAttentionMgr
from mindformers.experimental.infer.core.layers import ColumnParallelLinear, RowParallelLinear
from mindformers.experimental.parallel_core.pynative.parallel_state import get_tensor_model_parallel_world_size

from mindspore_gs.ptq.ptq_config import InnerPTQConfig, PTQMode, OutliersSuppressionType, QuantType, DeviceType, \
    OpsPriority
from mindspore_gs.ptq.convert_utils import QuantCellV2, AntiQuantCell
from mindspore_gs.ptq.ptq.wrapper_cell import WrapperCell
from mindspore_gs.quantization.quant_utils import (
    get_quant_min_max, cal_quantization_params,
    quant_tensor_data,
    convert_fp32_to_int64
)
from mindspore_gs.common.numpy_quant_common import NumpyQuantOps
from .parallel_minmax import MaxFromTensorParallelRegion, MinFromTensorParallelRegion
from .linear_wrapper import WrapperLinearCell


def need_insert_ops_for_smooth(cfg):
    """need_insert_ops_for_smooth"""
    if cfg.outliers_suppression == OutliersSuppressionType.NONE:
        return False
    # when set no smooth_to_pre_layer, w8a8 fusion the smooth_scale with quantv2 ops and
    # w8a8_dynamic use smooth_scale in dynamic_quant ops, not need insert ops
    if cfg.act_quant_dtype == dtype.int8 and cfg.weight_quant_dtype == dtype.int8:
        return False
    return True


def need_smooth_params_for_a8w8_dynamic(cfg):
    """need_smooth_params_for_a8w8_dynamic"""
    qtype = cfg.act_weight_quant_type()
    return qtype == QuantType.A8W8_DYNAMIC and cfg.outliers_suppression == OutliersSuppressionType.SMOOTH


class QuantLinearCell(WrapperLinearCell):
    """QuantLinearCell"""

    @staticmethod
    def is_enable(cfg: InnerPTQConfig):
        return cfg.weight_quant_dtype == dtype.int8

    def __init__(self, linear_name, linear, cfg: InnerPTQConfig, network_helper):
        super().__init__(linear_name, linear, cfg, network_helper)
        self.is_rowparallel = isinstance(self.layer, RowParallelLinear)
        self.is_colparallel = isinstance(self.layer, ColumnParallelLinear)
        self.is_linear = isinstance(self.layer, Linear)
        if not self.is_rowparallel and not self.is_colparallel and not self.is_linear:
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
            self.weight_quant_min, self.weight_quant_max = get_quant_min_max(num_bits=8, signed=True,
                                                                             narrow_range=cfg.weight_narrow_range)
        if cfg.act_quant_dtype == dtype.int8:
            self.act_quant_min, self.act_quant_max = get_quant_min_max(num_bits=8, signed=True,
                                                                       narrow_range=cfg.act_narrow_range)

        self.compute_type = self.layer.dtype if self.is_linear else self.layer.compute_dtype

        self.quantizer_x_max = None
        self.quantizer_x_min = None
        self.quantizer_w_max = None
        self.quantizer_w_min = None
        if self.is_rowparallel:
            self.x_quant_max = MaxFromTensorParallelRegion()
            self.x_quant_min = MinFromTensorParallelRegion()
            self.w_quant_max = MaxFromTensorParallelRegion()
            self.w_quant_min = MinFromTensorParallelRegion()
        else:
            self.x_quant_max = msops.max
            self.x_quant_min = msops.min
            self.w_quant_max = msops.max
            self.w_quant_min = msops.min

        self.quant_type = self.cfg.act_weight_quant_type()
        if self.quant_type is QuantType.UNDEFINED:
            raise ValueError("config quant type is undefined in QuantLinearCell, config is {cfg}.")
        self.input_scale = None
        self.input_zp = None
        self.weight = None
        self.smooth_scale = None
        self.w_scale = None
        self.x_scale = None
        self.w_zp = None
        self.bias = None
        self.bias_name = None
        param_init_func = QuantLinearCell.param_init_map.get((cfg.device_type, cfg.ops_priority))
        if param_init_func is None:
            raise ValueError("key ({cfg.device_type}, {cfg.ops_priority}) is not in QuantLinearCell.param_init_map.")
        param_init_func(self)

    def _param_init(self):
        """_param_init"""
        self.input_scale = Parameter(initializer('ones', (self.ic), dtype.float16), name="input_scale")
        self.input_zp = Parameter(initializer('zeros', (self.ic), dtype.int8), name="input_zp")
        self.weight = Parameter(initializer("ones", self.layer.weight.shape, dtype.int8), name=self.layer.weight.name)
        self.smooth_scale = Parameter(initializer('ones', (self.ic), dtype=self.compute_type))
        self.w_scale = Parameter(initializer('ones', (self.oc), dtype=self.compute_type))
        self.x_scale = Parameter(initializer('ones', (1,), dtype=self.compute_type))
        self.w_zp = Parameter(initializer('ones', (self.oc), dtype=dtype.int32))
        if self.quant_type is QuantType.A8W8:
            self.bias_name = self.layer.bias.name if self.layer.has_bias else self.layer.weight.name + "_bias"
            self.bias = Parameter(initializer("ones", (self.oc), self.compute_type), name=self.bias_name)

    param_init_map = {
        (DeviceType.ASCEND910B, OpsPriority.ACLNN): _param_init,
        (DeviceType.ASCEND910B, OpsPriority.INTERNAL): _param_init,
        (DeviceType.ASCEND910B, OpsPriority.ASD): _param_init,
        (DeviceType.ASCEND310, OpsPriority.ACLNN): _param_init,
        (DeviceType.ASCEND310, OpsPriority.INTERNAL): _param_init,
        (DeviceType.ASCEND310, OpsPriority.ASD): _param_init,
    }

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

    def _param_compute(self):
        """param compute"""
        w_scale, w_zp = cal_quantization_params(self.quantizer_w_min.asnumpy(), self.quantizer_w_max.asnumpy(),
                                                self.weight_quant_min, self.weight_quant_max,
                                                symmetric=self._weight_symmetric)
        weight = quant_tensor_data(self.layer.weight, w_scale.squeeze(), w_zp.squeeze(),
                                   self.weight_quant_min, self.weight_quant_max, self.weight_quantizer_axis)
        self.weight.set_data(Tensor(weight.asnumpy(), dtype=dtype.int8))
        self.w_scale.set_data(Tensor(np.squeeze(w_scale), dtype=self.compute_type))
        self.w_zp.set_data(Tensor(np.squeeze(w_zp), dtype=dtype.int32))
        if self.quant_type is QuantType.A8W8:
            self.quantizer_x_max = self.x_quant_max(self.cat_samples)[0]
            self.quantizer_x_min = self.x_quant_min(self.cat_samples)[0]
            x_scale, x_zp = cal_quantization_params(self.quantizer_x_min.asnumpy(), self.quantizer_x_max.asnumpy(),
                                                    self.act_quant_min,
                                                    self.act_quant_max,
                                                    symmetric=self._act_symmetric)
            self.x_scale.set_data(Tensor([x_scale], dtype=self.compute_type))
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
                smooth_scale = 1.0 / self.layer.matmul.smooth_scale.asnumpy()
            else:
                smooth_scale = None
            if smooth_scale is not None:
                input_scale = (x_scale * smooth_scale).astype(np.float16)
            else:
                input_scale = np.array([x_scale] * self.ic).astype(np.float16)
            self.input_scale.set_data(Tensor(input_scale, dtype=dtype.float16))
            input_zp = np.array([x_zp] * len(input_scale)).astype(np.int8)
            self.input_zp.set_data(Tensor(input_zp, dtype=dtype.int8))
        if need_insert_ops_for_smooth(self.cfg) or need_smooth_params_for_a8w8_dynamic(self.cfg):
            self.smooth_scale.set_data(Tensor(self.layer.matmul.smooth_scale.asnumpy(), dtype=self.compute_type))

    param_compute_map = {
        (DeviceType.ASCEND910B, OpsPriority.ACLNN): _param_compute,
        (DeviceType.ASCEND910B, OpsPriority.INTERNAL): _param_compute,
        (DeviceType.ASCEND910B, OpsPriority.ASD): _param_compute,
        (DeviceType.ASCEND310, OpsPriority.ACLNN): _param_compute,
        (DeviceType.ASCEND310, OpsPriority.INTERNAL): _param_compute,
        (DeviceType.ASCEND310, OpsPriority.ASD): _param_compute,
    }

    # pylint: disable=protected-access
    def quant_weight(self):
        """quant weight"""
        self.quantizer_w_max = self.w_quant_max(self.layer.weight, self.weight_quantizer_min_max_axis,
                                                keepdims=True)[0]
        self.quantizer_w_min = self.w_quant_min(self.layer.weight, self.weight_quantizer_min_max_axis,
                                                keepdims=True)[0]
        param_compute_func = QuantLinearCell.param_compute_map.get((self.cfg.device_type, self.cfg.ops_priority))
        if param_compute_func is None:
            raise ValueError("key ({self.cfg.device_type}, {self.cfg.ops_priority}) is \
                                    not in QuantLinearCell.param_compute_map.")
        param_compute_func(self)

    def process(self):
        super(QuantLinearCell, self).process()
        self.quant_weight()
        self.layer.weight._offload()
        self.cat_samples = None

    def deploy(self):
        return DeployLinearCell(self.layer, self.cfg, self.weight, self.bias, self.w_scale, self.w_zp,
                                self.x_scale, self.input_zp, self.input_scale, self.smooth_scale)


class DeployLinearCell(Cell):
    """DeployLinearCell"""

    def __init__(self, linear, cfg, weight, bias, w_scale, w_zp, x_scale, input_zp, input_scale, smooth_scale):
        super().__init__()
        self._layer = linear
        self.cfg = cfg
        self.quant = QuantV2()
        self.quant_type = cfg.act_weight_quant_type()
        if self.quant_type is QuantType.UNDEFINED:
            raise ValueError("config quant type is undefined in DeployLinearCell, config is {cfg}.")
        is_deploy = cfg.mode == PTQMode.DEPLOY
        self.need_insert_ops_for_smooth = need_insert_ops_for_smooth(cfg)
        if self.quant_type is QuantType.A8W8:
            if linear.has_bias is False:
                self.layer.has_bias = True
                self.layer.bias_add = msops.Add()
            self.layer.bias = bias
            self.input_scale = input_scale
            self.input_zp = input_zp
            qmm = AllQuantMatmul(is_deploy, x_scale.asnumpy(), w_scale.asnumpy(), transpose_b=self.layer.transpose_b)
        elif self.quant_type is QuantType.A16W8:
            qmm = WeightQuantMatmul(is_deploy, w_scale.asnumpy(), w_zp.asnumpy(), transpose_b=self.layer.transpose_b)
        elif self.quant_type is QuantType.A8W8_DYNAMIC:
            need_smooth = need_smooth_params_for_a8w8_dynamic(self.cfg)
            qmm = DynamicQuantMatmul(is_deploy, need_smooth, w_scale.asnumpy(),
                                     transpose_b=self.layer.transpose_b, smooth_scale=smooth_scale)
        if self.need_insert_ops_for_smooth:
            self.smooth_scale = smooth_scale
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
        if self.quant_type == QuantType.A8W8:
            x = self.quant(x, self.input_scale, self.input_zp, False, "ROUND", dtype.int8)
        if self.need_insert_ops_for_smooth:
            x = msops.mul(x, self.smooth_scale)
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
        if self.quant_type == QuantType.A8W8:
            input_parallel = self.quant(input_parallel, self.input_scale, self.input_zp, False, "ROUND", dtype.int8)
        if self.need_insert_ops_for_smooth:
            input_parallel = msops.mul(input_parallel, self.smooth_scale)
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
        if self.quant_type == QuantType.A8W8:
            input_parallel = self.quant(input_parallel, self.input_scale, self.input_zp, False, "ROUND", dtype.int8)
        if self.need_insert_ops_for_smooth:
            input_parallel = msops.mul(input_parallel, self.smooth_scale)
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

    def compute_a8w8_dynamic_sharded_state_dict(self, state_dict):
        '''compute_a8w8_dynamic_sharded_state_dict'''
        if self.is_colparallel:
            weight_scale_shard = (self.layer.tensor_parallel_group_size,)
        elif self.is_rowparallel:
            weight_scale_shard = (1,)
        state_dict[self.layer.matmul.weight_scale.name] = {'shape': self.layer.matmul.weight_scale.shape,
                                                           'shard': weight_scale_shard}
        if need_smooth_params_for_a8w8_dynamic(self.cfg):
            if self.is_colparallel:
                smooth_scale_shard = (1,)
            elif self.is_rowparallel:
                smooth_scale_shard = (self.layer.tensor_parallel_group_size,)
            state_dict[self.layer.matmul.smooth_scale.name] = {'shape': self.layer.matmul.smooth_scale.shape,
                                                               'shard': smooth_scale_shard}

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

        if self.quant_type is QuantType.A8W8:
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
        elif self.quant_type is QuantType.A16W8:
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
        elif self.quant_type is QuantType.A8W8_DYNAMIC:
            self.compute_a8w8_dynamic_sharded_state_dict(state_dict)
        if not self.need_insert_ops_for_smooth:
            return state_dict
        if self.is_colparallel:
            smooth_scale_shard = (1,)
        elif self.is_rowparallel:
            smooth_scale_shard = (self.layer.tensor_parallel_group_size,)
        state_dict[self.smooth_scale.name] = {'shape': self.smooth_scale.shape, 'shard': smooth_scale_shard}
        return state_dict


class DynamicQuantMatmul(Cell):
    """dynamic quant"""

    def __init__(self, is_deploy, need_smooth, weight_scale,
                 transpose_a=False, transpose_b=False, dst_dtype=dtype.float16, smooth_scale=None):
        super().__init__()
        if is_deploy:
            self.weight_scale = Parameter(initializer('ones', weight_scale.shape, dtype.float32))
        else:
            self.weight_scale = Parameter(Tensor(weight_scale, dtype=dtype.float32))
        self.dynamic_quant = DynamicQuantExt()
        self.smooth_scale = smooth_scale if need_smooth else None
        self.qbmm = QuantBatchMatmul(transpose_x1=transpose_a, transpose_x2=transpose_b, dtype=dst_dtype)

    def construct(self, x, quant_weight):
        qx, x_scale = self.dynamic_quant(x, self.smooth_scale)
        return self.qbmm(qx, quant_weight, self.weight_scale, None, None, x_scale)


class WeightQuantMatmul(Cell):
    """quant batch matmul"""

    def __init__(self, is_deploy, t_scale, t_zp, transpose_a=False, transpose_b=False, dst_type=dtype.float16):
        super().__init__()
        self.dst_dtype = dst_type
        if is_deploy:
            self.t_scale = Parameter(initializer('ones', t_scale.shape, dtype.float16), name="t_scale")
            self.t_zp_neg = Parameter(initializer('ones', t_zp.shape, dtype.float16), name="t_zp_neg")
        else:
            self.t_scale = Parameter(Tensor(t_scale, dtype=self.dst_dtype), name="t_scale")
            self.t_zp_neg = Parameter(Tensor(t_zp * -1, dtype=self.dst_dtype), name="t_zp_neg")
        self.weight_qbmm = WeightQuantBatchMatmul(transpose_a, transpose_b)

    def construct(self, x, weight):
        """forward for antiquant bmm cell"""
        output = self.weight_qbmm(x, weight, self.t_scale, self.t_zp_neg, None, None, None)
        return output.astype(self.dst_dtype)


class AllQuantMatmul(Cell):
    """quant weight"""

    def __init__(self, is_deploy, input_scale, weight_scale, offset=None,
                 transpose_a=False, transpose_b=False, dst_dtype=dtype.float16):
        super().__init__()
        self.transpose_b = transpose_b
        self.offset = offset
        if is_deploy:
            self.dequant_scale = Parameter(initializer('ones', weight_scale.shape, dtype.int64))
        else:
            self.dequant_scale = Parameter(Tensor(self.compute_dequant_scale(input_scale, weight_scale),
                                                  dtype=dtype.int64))
        self.qbmm = QuantBatchMatmul(transpose_x1=transpose_a, transpose_x2=transpose_b, dtype=dst_dtype)
        if offset is None:
            self.offset = None
        else:
            self.offset = Parameter(Tensor(offset, dtype=dtype.float32))

    def compute_dequant_scale(self, input_scale, weight_scale):
        '''compute_dequant_scale'''
        dequant_scale = input_scale.astype(np.float32) * weight_scale.astype(np.float32)
        scale_i64 = NumpyQuantOps.trans_fp32_to_i64(dequant_scale)
        return scale_i64

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
        self.ic = n * d
        self.k_scale_no_fusion = None
        self.k_zp_no_fusion = None
        self.v_scale_no_fusion = None
        self.v_zp_no_fusion = None
        self.k_v_scale_fusion = None
        self.k_v_zp_fusion = None

        self.kvcache_quant_min, self.kvcache_quant_max = get_quant_min_max(num_bits=8,
                                                                           signed=True,
                                                                           narrow_range=cfg.kvcache_narrow_range)
        if not self.cfg.kvcache_dynamic_quant:
            param_init_func = QuantPageAttentionMgrCell.param_init_map.get((cfg.device_type, cfg.ops_priority))
            if param_init_func is None:
                raise ValueError("key ({cfg.device_type}, {cfg.ops_priority}) is not in \
                                QuantPageAttentionMgrCell.param_init_map.")
            param_init_func(self)

    def _param_init_asd(self):
        """_param_init_asd"""
        self.k_scale_no_fusion = Parameter(initializer('ones', (self.ic), dtype.float16))
        self.k_zp_no_fusion = Parameter(initializer('ones', (self.ic), dtype.float16))
        self.v_scale_no_fusion = Parameter(initializer('ones', (self.ic), dtype.float16))
        self.v_zp_no_fusion = Parameter(initializer('ones', (self.ic), dtype.float16))
        self.k_v_scale_fusion = Parameter(initializer('ones', (2, self.ic), dtype.int64))
        self.k_v_zp_fusion = Parameter(initializer('ones', (2, self.ic), dtype.int32))

    def _param_init_internal(self):
        """_param_init_internal"""
        self.k_scale_no_fusion = Parameter(initializer('ones', (self.ic), dtype.float16))
        self.k_zp_no_fusion = Parameter(initializer('ones', (self.ic), dtype.float16))
        self.v_scale_no_fusion = Parameter(initializer('ones', (self.ic), dtype.float16))
        self.v_zp_no_fusion = Parameter(initializer('ones', (self.ic), dtype.float16))
        self.k_v_scale_fusion = Parameter(initializer('ones', (2, self.ic), dtype.float16))
        self.k_v_zp_fusion = Parameter(initializer('ones', (2, self.ic), dtype.float16))

    param_init_map = {
        (DeviceType.ASCEND910B, OpsPriority.ACLNN): _param_init_asd,
        (DeviceType.ASCEND910B, OpsPriority.INTERNAL): _param_init_internal,
        (DeviceType.ASCEND910B, OpsPriority.ASD): _param_init_asd,
        (DeviceType.ASCEND310, OpsPriority.ACLNN): _param_init_asd,
        (DeviceType.ASCEND310, OpsPriority.INTERNAL): _param_init_internal,
        (DeviceType.ASCEND310, OpsPriority.ASD): _param_init_asd,
    }

    def process(self):
        if self.cfg.kvcache_dynamic_quant:
            return
        if not self.key_samples or not self.value_samples:
            raise RuntimeError("Please catch ReshapeAndCache inputs before quantization.")
        key_cat_samples = msops.cat(tuple(self.key_samples), axis=0)
        self.quantizer_key_max = msops.max(key_cat_samples, 0)[0]
        self.quantizer_key_min = msops.min(key_cat_samples, 0)[0]

        value_cat_samples = msops.cat(tuple(self.value_samples), axis=0)
        self.quantizer_value_max = msops.max(value_cat_samples, 0)[0]
        self.quantizer_value_min = msops.min(value_cat_samples, 0)[0]

        key_t_scale, key_t_zp = cal_quantization_params(self.quantizer_key_min.asnumpy(),
                                                        self.quantizer_key_max.asnumpy(),
                                                        self.kvcache_quant_min,
                                                        self.kvcache_quant_max,
                                                        symmetric=self.kvcache_symmetric)
        value_t_scale, value_t_zp = cal_quantization_params(self.quantizer_value_min.asnumpy(),
                                                            self.quantizer_value_max.asnumpy(),
                                                            self.kvcache_quant_min,
                                                            self.kvcache_quant_max,
                                                            symmetric=self.kvcache_symmetric)
        key_t_scale = np.squeeze(key_t_scale).astype(np.float16)
        key_t_zp = np.squeeze(key_t_zp).astype(np.float16)
        value_t_scale = np.squeeze(value_t_scale).astype(np.float16)
        value_t_zp = np.squeeze(value_t_zp).astype(np.float16)
        self.k_scale_no_fusion.set_data(Tensor(key_t_scale, dtype=dtype.float16))
        self.k_zp_no_fusion.set_data(Tensor(key_t_zp, dtype=dtype.float16))
        self.v_scale_no_fusion.set_data(Tensor(value_t_scale, dtype=dtype.float16))
        self.v_zp_no_fusion.set_data(Tensor(value_t_zp, dtype=dtype.float16))
        param_compute_func = QuantPageAttentionMgrCell.param_compute_map[(self.cfg.device_type, self.cfg.ops_priority)]
        if param_compute_func is None:
            raise ValueError("key ({self.cfg.device_type}, {self.cfg.ops_priority}) is \
                                    not in QuantPageAttentionMgrCell.param_compute_map.")
        param_compute_func(self, key_t_scale, value_t_scale, key_t_zp, value_t_zp)

        self.key_samples.clear()
        self.value_samples.clear()

    def _param_compute_asd(self, key_t_scale, value_t_scale, key_t_zp, value_t_zp):
        """_param_compute_asd"""
        t_scale_len = self.k_scale_no_fusion.shape[0]
        key_t_scale = convert_fp32_to_int64(key_t_scale.astype(np.float32))
        value_t_scale = convert_fp32_to_int64(value_t_scale.astype(np.float32))
        key_value_t_scale = np.concatenate((key_t_scale.reshape((1, t_scale_len)),
                                            value_t_scale.reshape((1, t_scale_len))))

        t_zp_len = self.v_zp_no_fusion.shape[0]
        key_t_zp = (key_t_zp*-1).astype(np.int32)
        value_t_zp = (value_t_zp*-1).astype(np.int32)
        key_value_t_zp = np.concatenate((key_t_zp.reshape((1, t_zp_len)), value_t_zp.reshape((1, t_zp_len))))

        self.k_v_scale_fusion.set_data(Tensor(key_value_t_scale, dtype=dtype.int64))
        self.k_v_zp_fusion.set_data(Tensor(key_value_t_zp, dtype=dtype.int32))

    def _param_compute_internal(self, key_t_scale, value_t_scale, key_t_zp, value_t_zp):
        """_param_compute_internal"""
        t_scale_len = self.k_scale_no_fusion.shape[0]
        key_value_t_scale = np.concatenate((key_t_scale.reshape((1, t_scale_len)),
                                            value_t_scale.reshape((1, t_scale_len))))
        t_zp_len = self.v_zp_no_fusion.shape[0]
        key_t_zp = key_t_zp*-1
        value_t_zp = value_t_zp*-1
        key_value_t_zp = np.concatenate((key_t_zp.reshape((1, t_zp_len)), value_t_zp.reshape((1, t_zp_len))))
        self.k_v_scale_fusion.set_data(Tensor(key_value_t_scale, dtype=dtype.float16))
        self.k_v_zp_fusion.set_data(Tensor(key_value_t_zp, dtype=dtype.float16))

    param_compute_map = {
        (DeviceType.ASCEND910B, OpsPriority.ACLNN): _param_compute_asd,
        (DeviceType.ASCEND910B, OpsPriority.INTERNAL): _param_compute_internal,
        (DeviceType.ASCEND910B, OpsPriority.ASD): _param_compute_asd,
        (DeviceType.ASCEND310, OpsPriority.ACLNN): _param_compute_asd,
        (DeviceType.ASCEND310, OpsPriority.INTERNAL): _param_compute_internal,
        (DeviceType.ASCEND310, OpsPriority.ASD): _param_compute_asd,
    }

    def deploy(self):
        if self.cfg.kvcache_dynamic_quant:
            return DeployDynamicQuantPagedAttentionCell(self.layer)
        return DeployPageAttentionMgrCell(self.layer, self.v_scale_no_fusion, self.v_zp_no_fusion,
                                          self.k_scale_no_fusion, self.k_zp_no_fusion, self.k_v_scale_fusion,
                                          self.k_v_zp_fusion, self.cfg)

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

    def __init__(self, kvcache: PagedAttentionMgr, v_scale_no_fusion, v_zp_no_fusion, k_scale_no_fusion, k_zp_no_fusion,
                 k_v_scale_fusion, k_v_zp_fusion, cfg: InnerPTQConfig):
        super().__init__()
        self.layer = kvcache
        self.enable_deploy_fusion = cfg.enable_deploy_fusion
        self._key_input_quantizer = QuantCellV2(Tensor(k_scale_no_fusion.asnumpy(), dtype=dtype.float16),
                                                Tensor(k_zp_no_fusion.asnumpy().astype(np.int8), dtype=dtype.int8))
        self._value_input_quantizer = QuantCellV2(Tensor(v_scale_no_fusion.asnumpy(), dtype=dtype.float16),
                                                  Tensor(v_zp_no_fusion.asnumpy().astype(np.int8), dtype=dtype.int8))
        dst_type = self.layer.key_cache.dtype
        n = kvcache.n_kv_heads
        d = kvcache.head_dim
        self._key_output_quantizer = AntiQuantCell(n, d, dst_type)
        self._value_output_quantizer = AntiQuantCell(n, d, dst_type)
        if cfg.mode == PTQMode.QUANTIZE or not self.enable_deploy_fusion:
            self.k_zp_no_fusion = k_zp_no_fusion
            self.v_zp_no_fusion = v_zp_no_fusion
            self.k_scale_no_fusion = k_scale_no_fusion
            self.v_scale_no_fusion = v_scale_no_fusion
        if cfg.mode == PTQMode.QUANTIZE or self.enable_deploy_fusion:
            self.k_v_scale_fusion = k_v_scale_fusion
            self.k_v_zp_fusion = k_v_zp_fusion

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
            kcache = self._key_output_quantizer(self.layer.key_cache, self.k_zp_no_fusion, self.k_scale_no_fusion)
            vcache = self._value_output_quantizer(self.layer.value_cache, self.v_zp_no_fusion, self.v_scale_no_fusion)
            return self.layer.paged_attention(query, kcache, vcache, block_tables, batch_valid_length)
        return self.layer.paged_attention(query, self.layer.key_cache, self.layer.value_cache, block_tables,
                                          batch_valid_length, self.k_v_scale_fusion, self.k_v_zp_fusion)

    def paged_attn_with_alibi(self, query, batch_valid_length, block_tables, alibi_tensor):
        """The forward compute of Paged Attention."""
        if not self.enable_deploy_fusion:
            kcache = self._key_output_quantizer(self.layer.key_cache, self.k_zp_no_fusion, self.k_scale_no_fusion)
            vcache = self._value_output_quantizer(self.layer.value_cache, self.v_zp_no_fusion, self.v_scale_no_fusion)
            return self.layer.paged_attention_with_alibi(query, kcache, vcache, block_tables, batch_valid_length,
                                                         alibi_tensor)
        return self.layer.paged_attention_with_alibi(query, self.layer.key_cache, self.layer.value_cache,
                                                     block_tables, batch_valid_length, self.k_v_scale_fusion,
                                                     self.k_v_zp_fusion, alibi_tensor)

    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        state_dict = {}
        if self.enable_deploy_fusion:
            key_value_t_scale_shard = (1, self.tensor_parallel_group_size)
            key_value_t_zp_shard = (1, self.tensor_parallel_group_size)
            state_dict[self.k_v_scale_fusion.name] = {'shape': self.k_v_scale_fusion.shape,
                                                      'shard': key_value_t_scale_shard}
            state_dict[self.k_v_zp_fusion.name] = {'shape': self.k_v_zp_fusion.shape,
                                                   'shard': key_value_t_zp_shard}
        else:
            key_t_scale_shard = (self.tensor_parallel_group_size,)
            key_t_zp_shard = (self.tensor_parallel_group_size,)

            value_t_scale_shard = (self.tensor_parallel_group_size,)
            value_t_zp_shard = (self.tensor_parallel_group_size,)

            state_dict[self.k_scale_no_fusion.name] = {'shape': self.k_scale_no_fusion.shape,
                                                       'shard': key_t_scale_shard}
            state_dict[self.k_zp_no_fusion.name] = {'shape': self.k_zp_no_fusion.shape,
                                                    'shard': key_t_zp_shard}
            state_dict[self.v_scale_no_fusion.name] = {'shape': self.v_scale_no_fusion.shape,
                                                       'shard': value_t_scale_shard}
            state_dict[self.v_zp_no_fusion.name] = {'shape': self.v_zp_no_fusion.shape,
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


class DeployDynamicQuantPagedAttentionCell(Cell):
    """DeployDynamicQuantPagedAttentionCell"""

    def __init__(self, kvcache: PagedAttentionMgr):
        super().__init__()
        self._kvcache = kvcache
        self.paged_attention = msops.auto_generate.PagedAttention(self._kvcache.n_heads,
                                                                  self._kvcache.scale_value,
                                                                  self._kvcache.n_kv_heads,
                                                                  "PERTOKEN")
        self.paged_attention_with_alibi = msops.auto_generate.PagedAttentionMask(self._kvcache.n_heads,
                                                                                 self._kvcache.scale_value,
                                                                                 self._kvcache.n_kv_heads,
                                                                                 "PERTOKEN")
        if "in_strategy" in kvcache.paged_attention.get_attr_dict():
            pa_strategy = kvcache.paged_attention.in_strategy
            self.paged_attention.shard(pa_strategy)

        if "in_strategy" in kvcache.paged_attention_with_alibi.get_attr_dict():
            pa_strategy = kvcache.paged_attention_with_alibi.in_strategy
            self.paged_attention_with_alibi.shard(pa_strategy)

    # pylint: disable=W0221
    def construct(self, key, value, slot_mapping):
        """The forward compute of KVCache for Paged Attention."""
        return self._kvcache.reshape_and_cache(key, value, self._kvcache.key_cache, self._kvcache.value_cache,
                                               slot_mapping)

    # pylint: disable=W0613
    def paged_attn(self, query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=None):
        """The forward compute of Paged Attention."""
        return self.paged_attention(query, self._kvcache.key_cache, self._kvcache.value_cache,
                                    block_tables, batch_valid_length)

    def paged_attn_with_alibi(self, query, batch_valid_length, block_tables, alibi_tensor):
        """The forward compute of KVCache for Paged Attention with alibi tensor."""
        return self.paged_attention_with_alibi(query, self._kvcache.key_cache, self._kvcache.value_cache,
                                               block_tables, batch_valid_length, None, None, alibi_tensor)
