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
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore import Parameter, Tensor, dtype
from mindspore.common.initializer import initializer
from mindformers.modules.layers import Linear

from mindspore_gs.common.gs_enum import BackendTarget
from mindspore_gs.quantization.fake_quantizer import LinearFakeQuantizer
from mindspore_gs.quantization.quant_utils import get_quant_min_max, quant_tensor_data, quant_bias_data
from mindspore_gs.quantization.layer_policy import PerChannelArgs
from mindspore_gs.ptq.ptq_config import PTQMode
from mindspore_gs.ptq.quant_cells import PTQCell
from mindspore_gs.ptq.convert_utils import (
    convert_to_dequant_bmm_for_deploy, convert_to_dequant_bmm, convert_to_smooth_quant_for_deploy,
    convert_to_smooth_quant
)


class SQLinearActObserver(PTQCell):
    """Linear layer wrapper with min max"""

    def __init__(self, linear: Linear, policy=None, cfg=None):
        super().__init__(linear, policy)
        if not isinstance(linear, Linear):
            raise ValueError(f'only Linear cell is supported, but got {type(linear)}')
        self.cfg = cfg
        input_fq_args = {}
        if "in_strategy" in linear.matmul.get_attr_dict():
            input_fq_args["strategy"] = (linear.matmul.in_strategy[0],)
        act_rank = 2
        self.act_observer = policy.create_observer_perchannel(
            perchannel_args=PerChannelArgs(linear.in_channels, -1, act_rank), **input_fq_args)
        prex = ""
        for _, param in linear.parameters_and_names():
            prex = param.name.rsplit(".", 1)[0]
        self.act_observer.float_min.data.name = prex + "_act_observer_float_min"
        self.act_observer.float_max.data.name = prex + "_act_observer_float_max"

    def weight_quantizer(self):
        raise NotImplementedError("Inner Error: should not call SQLinearActObserver.weight_quantizer().")

    def convert(self, backend: BackendTarget = BackendTarget.NONE, is_deploy=False):
        raise NotImplementedError("Inner Error: should not call SQLinearActObserver.convert().")

    def calibrate(self):
        pass

    def core_construct(self, *args):
        pass

    # pylint: disable=W0221
    def construct(self, x):
        """
        Defines the computation of LinearWithMinMax to be performed.

        Returns:
            Tensor, returns the computed result.
        """
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
        x = self.act_observer(x)
        x = self._handler.matmul(x, weight)
        if self._handler.has_bias:
            x = self._handler.bias_add(x, self._handler.cast(self._handler.bias, self._handler.dtype))
        if self._handler.activation_flag:
            x = self._handler.activation(x)
        x = F.cast(x, ori_dtype)
        output = self._handler.reshape(x, out_shape)
        return output


class SQLinearWeightObserver(PTQCell):
    """Linear layer wrapper with min max"""

    def __init__(self, act_observer: SQLinearActObserver):
        if not isinstance(act_observer, SQLinearActObserver):
            raise ValueError(f'only SQLinearActObserver cell is supported, but got {type(act_observer)}')
        policy = act_observer.policy()
        linear = act_observer.handler()
        self.cfg = act_observer.cfg
        super().__init__(linear, policy)
        rank = len(linear.weight.shape)
        self._weight_axis = rank - 2 if linear.matmul.transpose_b else rank - 1
        weight_perchannel_args = PerChannelArgs(linear.out_channels, self._weight_axis, rank)
        weight_fq_args = {}
        self._act_strategy = None
        self._weight_strategy = None
        if "in_strategy" in linear.matmul.get_attr_dict():
            weight_fq_args["strategy"] = (linear.matmul.in_strategy[1],)
        self.weight_quantizer_ = policy.get_weight_quantizer(linear.weight.name, weight_perchannel_args,
                                                             **weight_fq_args)

        weight_observer_axis = 1 if linear.matmul.transpose_b else 0
        self.act_observer = act_observer.act_observer
        self.weight_observer = policy.create_observer_perchannel(
            perchannel_args=PerChannelArgs(linear.in_channels, weight_observer_axis, rank), **weight_fq_args)
        prex = ""
        for _, param in linear.parameters_and_names():
            prex = param.name.rsplit(".", 1)[0]
        self.weight_quantizer_.float_min.data.name = prex + "_weight_float_min"
        self.weight_quantizer_.float_max.data.name = prex + "_weight_float_max"
        self.weight_observer.float_min.data.name = prex + "_weight_observer_float_min"
        self.weight_observer.float_max.data.name = prex + "_weight_observer_float_max"
        self.scale_store = Parameter(Tensor([1.0] * linear.in_channels), name=f'{prex}_scale_store')

        self._alpha = self.cfg.algo_args.get('alpha', None)
        if self._alpha is None:
            self._alpha = 0.5

        self._expand = P.ExpandDims()
        self._weight_mul = P.Mul()
        self._weight_div = P.Div()
        self._smooth_act_maximum = P.Maximum()
        self._smooth_act_abs = P.Abs()
        self._act_pow = P.Pow()
        self._smooth_weight_maximum = P.Maximum()
        self._smooth_weight_abs = P.Abs()
        self._weight_pow = P.Pow()
        self._pow_div = P.Div()
        self._assign = P.Assign()
        self._weight_assign = P.Assign()
        if "in_strategy" in linear.matmul.get_attr_dict():
            self.shard()

    def shard(self):
        """
        shard.
        should consider out_strategy.
        """
        self._act_strategy = self._handler.matmul.in_strategy[0]
        self._weight_strategy = self._handler.matmul.in_strategy[1]
        mul_strategy = (self._act_strategy[1],)
        weight_in_strategy = self._weight_in_strategy(self._weight_strategy, self._handler.transpose_b)
        weight_out_strategy = self._weight_out_strategy(self._weight_strategy, self._handler.transpose_b)
        # weight * smooth_scale(weight_channel_in)
        if self._handler.transpose_b:
            self._weight_mul.shard((self._weight_strategy, weight_in_strategy))
            self._weight_div.shard((self._weight_strategy, weight_in_strategy))
        else:
            self._weight_mul.shard((self._weight_strategy, (weight_in_strategy[0], 1)))
            self._weight_div.shard((self._weight_strategy, (weight_in_strategy[0], 1)))
        # act observer pow, activation in
        self._smooth_act_maximum.shard((mul_strategy, mul_strategy))
        self._smooth_act_abs.shard((mul_strategy,))
        self._act_pow.shard((mul_strategy, ()))
        # weight observer pow, weight channel in
        self._smooth_weight_maximum.shard((weight_in_strategy, weight_in_strategy))
        self._smooth_weight_abs.shard((weight_in_strategy,))
        self._weight_pow.shard((weight_in_strategy, ()))
        # act_max_pow / weight_max_pow
        self._pow_div.shard((mul_strategy, weight_in_strategy))
        # store_scale assign to smooth scale
        self._assign.shard((mul_strategy, mul_strategy))
        # new weight assign to linear weight
        self._weight_assign.shard((self._weight_strategy, self._weight_strategy))
        # bias add strategy: activation index 0 to weight channel out, bias: weight channel out
        if self._handler.has_bias:
            self._handler.bias_add.shard(((self._act_strategy[0], weight_out_strategy[0]), weight_out_strategy))

    @staticmethod
    def _weight_in_strategy(strategy, is_transpose):
        if is_transpose:
            return (strategy[1],)
        return (strategy[0],)

    @staticmethod
    def _weight_out_strategy(strategy, is_transpose):
        if is_transpose:
            return (strategy[0],)
        return (strategy[1],)

    def _calc_smooth_scale(self):
        """calc_smooth_scale"""
        act_max = self._smooth_act_maximum(self._smooth_act_abs(self.act_observer.float_max),
                                           self._smooth_act_abs(self.act_observer.float_min))
        input_max_pow = self._act_pow(act_max, self._alpha)
        weight_max = self._smooth_weight_maximum(self._smooth_weight_abs(self.weight_observer.float_max),
                                                 self._smooth_weight_abs(self.weight_observer.float_min))
        weight_max_pow = self._weight_pow(weight_max, 1 - self._alpha)
        smooth_scale = self._pow_div(input_max_pow, weight_max_pow).clip(1e-5)

        # set 0 or nan to 1.0 to avoid quantization error
        smooth_scale[input_max_pow == 0] = 1.0
        smooth_scale[weight_max_pow == 0] = 1.0
        return smooth_scale

    def weight_quantizer(self):
        return self.weight_quantizer_

    def convert(self, backend: BackendTarget = BackendTarget.NONE, is_deploy=False):
        raise NotImplementedError("Inner Error: should not call SQLinearWeightObserver.convert().")

    def calibrate(self):
        weight = self._handler.cast(self._handler.weight, self._handler.dtype)
        self.weight_quantizer_(weight)

    def core_construct(self, *args):
        pass

    # pylint: disable=W0221
    def construct(self, x):
        """
        Defines the computation of LinearWithMinMax to be performed.

        Returns:
            Tensor, returns the computed result.
        """
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

        weight = self.weight_observer(weight)
        smooth_scale = self._calc_smooth_scale()
        self._assign(self.scale_store, smooth_scale)
        if self._handler.transpose_b:
            weight = self._weight_mul(weight, smooth_scale)
            weight = self.weight_quantizer_(weight)
            weight = self._weight_div(weight, smooth_scale)
        else:
            # now only Matmul is supported, shall generalize to bmm
            weight_scale = self._expand(smooth_scale, 1)
            weight = self._weight_mul(weight, weight_scale)
            weight = self.weight_quantizer_(weight)
            weight = self._weight_div(weight, weight_scale)
        weight = self._handler.cast(weight, self._handler.dtype)
        x = self._handler.matmul(x, weight)
        if self._handler.has_bias:
            x = self._handler.bias_add(x, self._handler.cast(self._handler.bias, self._handler.dtype))
        if self._handler.activation_flag:
            x = self._handler.activation(x)
        x = F.cast(x, ori_dtype)
        output = P.Reshape()(x, out_shape)
        return output


class SQLinearWrapper(PTQCell):
    """Linear layer wrapper with min max"""

    def __init__(self, weight_observer_cell: SQLinearWeightObserver):
        if not isinstance(weight_observer_cell, SQLinearWeightObserver):
            raise ValueError(f'only SQLinearWeightObserver cell is supported, but got {type(weight_observer_cell)}')
        policy = weight_observer_cell.policy()
        linear = weight_observer_cell.handler()
        self.cfg = weight_observer_cell.cfg
        super().__init__(linear, policy)
        self._weight_observer_cell = weight_observer_cell
        rank = len(linear.weight.shape)
        self._weight_axis = rank - 2 if linear.matmul.transpose_b else rank - 1
        input_fq_args = {}
        self._act_strategy = None
        self._weight_strategy = None
        if "in_strategy" in linear.matmul.get_attr_dict():
            input_fq_args["strategy"] = (linear.matmul.in_strategy[0],)

        self._input_quantizer = policy.get_input_quantizer(**input_fq_args)
        self._weight_quantizer = weight_observer_cell.weight_quantizer_
        self.act_observer = weight_observer_cell.act_observer
        self.weight_observer = weight_observer_cell.weight_observer
        self._output_quantizer = None
        prex = ""
        for _, param in linear.parameters_and_names():
            prex = param.name.rsplit(".", 1)[0]
        if self._input_quantizer:
            self._input_quantizer.float_min.data.name = prex + "_input_float_min"
            self._input_quantizer.float_max.data.name = prex + "_input_float_max"

        mode = self.cfg.mode
        self._is_deploy = mode == PTQMode.DEPLOY
        self._alpha = self.cfg.algo_args.get('alpha', None)
        if self._alpha is None:
            self._alpha = 0.5
        if self._is_deploy:
            self._handler.weight = Parameter(initializer("ones", linear.weight.shape, dtype.int8),
                                             name=linear.weight.name)
            if linear.has_bias:
                self._handler.bias = Parameter(initializer("ones", linear.bias.shape, dtype.int32),
                                               name=linear.bias.name)
        self._expand = P.ExpandDims()
        self._act_mul = P.Mul()
        self._weight_mul = P.Mul()
        self._div = P.Div()
        self._assign = P.Assign()
        self._weight_assign = P.Assign()
        if "in_strategy" in linear.matmul.get_attr_dict():
            self.shard()

    def _get_bias_reduce_num(self):
        """
        1. matmul may have four kind of in_strategy: (1,m)(m,1);(m,1)(1,1);(1,1)(1,m);(1,1)(1,1). We can find that
         (1,m)(m,1) will add allreduce after matmul, (m,1)(1,1) and (1,1)(1,m) will add allgather after matmul.
         (1,1)(1,1) will not add any operation after matmul.
        2. We can simplify Linear construct to matmul + bias + act, and quant-linear construct to matmulint8 + bias +
         dequant + act. In allreduce-parallel mode, allreduce should insert as matmulint8 + allreduce + bias + dequant +
         act, while in allgather-parallel mode: matmulint8 + allgather + bias + dequant + act or
         matmulint8 + bias + allgather + dequant + act or matmulint8 + bias + dequant + allgather + act.
        3. If matmulint8 + bias + dequant use fused kernel QuantBatchMatmul, in allreduce-paralle mode, allreduce can
         not be inserted between matmul and bias anymore, so bias will act on more than one time. To correct this issue,
         we can move bias out of fused kernel or divide value in bias by x. This function is designed to find out the x.
        """

        if "in_strategy" not in self._handler.matmul.get_attr_dict():
            return 1
        if self._handler.matmul.in_strategy is None:
            return 1
        act_strategy = self._handler.matmul.in_strategy[0]
        weight_strategy = self._handler.matmul.in_strategy[1]
        weight_strategy_0 = weight_strategy[1] if self._handler.transpose_b else weight_strategy[0]
        weight_strategy_1 = weight_strategy[0] if self._handler.transpose_b else weight_strategy[1]
        # allreduce
        if act_strategy[0] == 1 and act_strategy[1] != 1 and weight_strategy_0 != 1 and weight_strategy_1 == 1:
            if act_strategy[1] != weight_strategy_0:
                raise RuntimeError(f"Invalid in_strategy for matmul: {self._handler.matmul.in_strategy}.")
            return act_strategy[1]
        # allgather or no-parallel
        if act_strategy[1] == 1 and weight_strategy_0 == 1:
            return 1
        raise RuntimeError(f"Invalid in_strategy for matmul: {self._handler.matmul.in_strategy}.")

    def shard(self):
        """
        shard.
        should consider out_strategy.
        """
        self._act_strategy = self._handler.matmul.in_strategy[0]
        self._weight_strategy = self._handler.matmul.in_strategy[1]
        mul_strategy = (self._act_strategy[1],)
        weight_in_strategy = self._weight_in_strategy(self._weight_strategy, self._handler.transpose_b)
        weight_out_strategy = self._weight_out_strategy(self._weight_strategy, self._handler.transpose_b)
        # activation * smooth_scale(channel_in)
        self._act_mul.shard((self._act_strategy, mul_strategy))
        # weight * smooth_scale(weight_channel_in)
        if self._handler.transpose_b:
            self._weight_mul.shard((self._weight_strategy, weight_in_strategy))
        else:
            self._weight_mul.shard((self._weight_strategy, (weight_in_strategy[0], 1)))
        # 1 / smooth_scale
        self._div.shard(((), weight_in_strategy))
        # store_scale assign to smooth scale
        self._assign.shard((mul_strategy, mul_strategy))
        # new weight assign to linear weight
        self._weight_assign.shard((self._weight_strategy, self._weight_strategy))
        # bias add strategy: activation index 0 to weight channel out, bias: weight channel out
        if self._handler.has_bias:
            self._handler.bias_add.shard(((self._act_strategy[0], weight_out_strategy[0]), weight_out_strategy))

    @staticmethod
    def _weight_in_strategy(strategy, is_transpose):
        if is_transpose:
            return (strategy[1],)
        return (strategy[0],)

    @staticmethod
    def _weight_out_strategy(strategy, is_transpose):
        if is_transpose:
            return (strategy[0],)
        return (strategy[1],)

    def _adjust_parameter(self):
        weight_scale = self._expand(self._weight_observer_cell.scale_store, 0)
        if not self._handler.transpose_b:
            weight_scale = weight_scale.transpose()
        orin_dtype = self._handler.weight.dtype
        weight = self._weight_mul(self._handler.weight, weight_scale)
        weight = self._handler.cast(weight, orin_dtype)
        self._weight_assign(self._handler.weight, weight)

    def weight_quantizer(self):
        return self._weight_quantizer

    def convert(self, backend: BackendTarget = BackendTarget.NONE, is_deploy=False):
        if not self._is_deploy:
            self._adjust_parameter()

        if self.cfg.backend == BackendTarget.ASCEND:
            # quant weight to int8, bias to int32
            self._input_quantizer = self._input_quantizer.convert_to_fakequantparam()
            self._weight_quantizer = self._weight_quantizer.convert_to_fakequantparam()
            weight_quant = None
            bias_quant = None
            bias_name = self._handler.weight.name + "_bias"
            if not self._is_deploy:
                weight_quantizer: P.FakeQuantParam = self._weight_quantizer.fq
                if hasattr(self._handler, "dtype"):
                    weight = self._handler.cast(self._handler.weight, self._handler.dtype)
                else:
                    weight = self._handler.weight
                quant_min, quant_max = get_quant_min_max(
                    num_bits=weight_quantizer.attrs[LinearFakeQuantizer.attr_key_num_bits],
                    signed=weight_quantizer.attrs[LinearFakeQuantizer.attr_key_signed],
                    narrow_range=weight_quantizer.attrs[LinearFakeQuantizer.attr_key_narrow_range])
                scale = np.array(weight_quantizer.attrs[P.FakeQuantParam.attr_key_linear_quant_scale])
                zp = np.array(weight_quantizer.attrs[P.FakeQuantParam.attr_key_linear_quant_zero_point])
                if not self._handler.transpose_b:
                    scale = scale.transpose()
                    zp = zp.transpose()
                weight_quant = quant_tensor_data(weight, scale, zp, quant_min, quant_max,
                                                 self._weight_axis)
                self._handler.weight = Parameter(Tensor(weight_quant, dtype=dtype.int8), name=self._handler.weight.name)
                input_quantizer = self._input_quantizer.fq
                act_scale = np.array(input_quantizer.attrs[P.FakeQuantParam.attr_key_linear_quant_scale])
                dequant_scale = scale * act_scale
                if self._handler.has_bias:
                    bias_quant = quant_bias_data(self._handler.bias, dequant_scale)
                    bias_name = self._handler.bias.name
                    self._handler.bias = Parameter(bias_quant, name=bias_name)
            param_bias_quant = bias_quant.asnumpy() if bias_quant is not None else None
            if param_bias_quant is not None:
                bn = self._get_bias_reduce_num()
                # refer to docstring of _get_bias_reduce_num for the reason of this divide operation.
                param_bias_quant = param_bias_quant / bn
            self._output_quantizer, bias = convert_to_dequant_bmm(self._input_quantizer,
                                                                  self._weight_quantizer,
                                                                  weight_quant,
                                                                  param_bias_quant,
                                                                  dst_dtype=self._handler.dtype,
                                                                  transpose_a=False,
                                                                  transpose_b=self._handler.transpose_b,
                                                                  strategy=self.antiquant_bmm_strategy(
                                                                      act_strategy=self._act_strategy,
                                                                      weight_strategy=self._weight_strategy,
                                                                      has_bias=True,  # offset correct by bias
                                                                      has_offset=False,
                                                                      is_transpose=self._handler.transpose_b))
            self._handler.has_bias = True
            if bias is not None:
                self._handler.bias = Parameter(Tensor(bias, dtype=dtype.int32), name=bias_name)
            else:
                bias_shape = [self._handler.weight.shape[0] if self._handler.transpose_b else
                              self._handler.weight.shape[1]]
                self._handler.bias = Parameter(initializer('ones', bias_shape, dtype=dtype.int32), name=bias_name)
            iq_strategy = (self._act_strategy,) if self._act_strategy else None
            self._input_quantizer = convert_to_smooth_quant(self._input_quantizer,
                                                            self._weight_observer_cell.scale_store,
                                                            strategy=iq_strategy)

    def calibrate(self):
        if hasattr(self._handler, "dtype"):
            weight = self._handler.cast(self._handler.weight, self._handler.dtype)
        else:
            weight = self._handler.weight
        self._weight_quantizer(weight)

    def core_construct(self, *args):
        pass

    # pylint: disable=W0221
    def construct(self, x):
        """
        Defines the computation of LinearWithMinMax to be performed.

        Returns:
            Tensor, returns the computed result.
        """
        out_shape = P.Shape()(x)[:-1] + (self._handler.out_channels,)
        x = P.Reshape()(x, (-1, self._handler.in_channels))
        if hasattr(self._handler, "expert_flag") and self._handler.expert_flag:
            if self._handler.use_expert_group_size is True:
                x = P.Reshape()(x, (-1, self._handler.expert_num, self._handler.expert_group_size,
                                    self._handler.in_channels))
            else:
                x = P.Reshape()(x, (self._handler.outer_batch, self._handler.expert_num, -1, self._handler.in_channels))
        ori_dtype = F.dtype(x)

        if self._is_deploy:
            x = self._input_quantizer(x)
            # (matmul(x, int8_weight) + int32_bias) * dequant_scale
            bias = None
            if self._handler.has_bias:
                bias = self._handler.bias
            x = self._output_quantizer(x, self._handler.weight, bias)
        else:
            weight = self._handler.cast(self._handler.weight, self._handler.dtype)
            x = self._handler.cast(x, self._handler.dtype)
            x = self._act_mul(x, self._div(1.0, self._weight_observer_cell.scale_store))
            x = self._input_quantizer(x)
            x = self._act_mul(x, self._weight_observer_cell.scale_store)
            x = self._handler.cast(x, self._handler.dtype)
            x = self._handler.matmul(x, weight)
            if self._handler.has_bias:
                x = self._handler.bias_add(x, self._handler.cast(self._handler.bias, self._handler.dtype))
        if self._handler.activation_flag:
            x = self._handler.activation(x)
        x = F.cast(x, ori_dtype)
        output = P.Reshape()(x, out_shape)
        return output


class SQLinearDeploy(PTQCell):
    """Linear layer wrapper with min max"""

    def __init__(self, linear: Linear, policy=None, cfg=None):
        super().__init__(linear, policy)
        if not isinstance(linear, Linear):
            raise ValueError(f'only Linear cell is supported, but got {type(linear)}')
        self.cfg = cfg

        rank = len(linear.weight.shape)
        ic_axis = rank - 1 if linear.matmul.transpose_b else rank - 2
        self._weight_axis = rank - 2 if linear.matmul.transpose_b else rank - 1
        ic = linear.weight.shape[ic_axis]
        oc = linear.weight.shape[self._weight_axis]

        self._act_strategy = None
        self._weight_strategy = None
        if "in_strategy" in linear.matmul.get_attr_dict():
            self._act_strategy = linear.matmul.in_strategy[0]
            self._weight_strategy = linear.matmul.in_strategy[1]

        self._handler.weight = Parameter(initializer("ones", linear.weight.shape, dtype.int8), name=linear.weight.name)
        if linear.has_bias:
            self._handler.bias = Parameter(initializer("ones", linear.bias.shape, dtype.int32), name=linear.bias.name)
        else:
            linear.has_bias = True
            bias_shape = [linear.weight.shape[0] if linear.transpose_b else linear.weight.shape[1]]
            bias_name = linear.weight.name + "_bias"
            self._handler.bias = Parameter(initializer('ones', bias_shape, dtype=dtype.int32), name=bias_name)
        iq_strategy = (self._act_strategy,) if self._act_strategy else None
        self._input_quantizer = convert_to_smooth_quant_for_deploy(ic, strategy=iq_strategy)
        oq_strategy = self.antiquant_bmm_strategy(act_strategy=self._act_strategy,
                                                  weight_strategy=self._weight_strategy,
                                                  has_bias=True,  # offset correct by bias
                                                  has_offset=False, is_transpose=linear.transpose_b)
        self._output_quantizer = convert_to_dequant_bmm_for_deploy(oc, dst_dtype=linear.dtype,
                                                                   transpose_b=linear.transpose_b,
                                                                   strategy=oq_strategy)

    def weight_quantizer(self):
        raise RuntimeError("Inner error, should not invoke SQLinearDeploy.weight_quantizer().")

    def calibrate(self):
        raise RuntimeError("Inner error, should not invoke SQLinearDeploy.calibrate().")

    def core_construct(self, *args):
        raise RuntimeError("Inner error, should not invoke SQLinearDeploy.core_construct().")

    # pylint: disable=W0221
    def construct(self, x):
        """
        Defines the computation of LinearWithMinMax to be performed.

        Returns:
            Tensor, returns the computed result.
        """
        out_shape = P.Shape()(x)[:-1] + (self._handler.out_channels,)
        x = P.Reshape()(x, (-1, self._handler.in_channels))
        if hasattr(self._handler, "expert_flag") and self._handler.expert_flag:
            if self._handler.use_expert_group_size is True:
                x = P.Reshape()(x, (-1, self._handler.expert_num, self._handler.expert_group_size,
                                    self._handler.in_channels))
            else:
                x = P.Reshape()(x, (self._handler.outer_batch, self._handler.expert_num, -1, self._handler.in_channels))
        ori_dtype = F.dtype(x)

        x = self._input_quantizer(x)
        bias = None
        if self._handler.has_bias:
            bias = self._handler.bias
        x = self._output_quantizer(x, self._handler.weight, bias)
        if self._handler.activation_flag:
            x = self._handler.activation(x)
        x = F.cast(x, ori_dtype)
        output = P.Reshape()(x, out_shape)
        return output
