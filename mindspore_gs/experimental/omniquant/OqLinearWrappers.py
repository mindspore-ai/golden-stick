# Copyright 2025 Huawei Technologies Co., Ltd
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
import abc
import numpy as np
from mindspore.ops import operations as P
from mindspore import Parameter, Tensor, dtype
from mindspore import nn
from mindspore import ops
from mindformers.modules import Linear
from mindspore_gs.quantization.quant_cell import QuantCell
from mindspore_gs.quantization.quant_utils import get_quant_min_max, quant_bias_data, quant_tensor_data
from mindspore_gs.quantization.layer_policy import PerChannelArgs
from mindspore_gs.quantization.fake_quantizer import FakeQuantizer
from mindspore_gs.ptq.fake_quantizer import MinMaxPerChannel


def create_observer_perchannel(perchannel_args: PerChannelArgs = PerChannelArgs(), **kwargs) -> FakeQuantizer:
    """create_observer_perchannel."""
    strategy = kwargs.get('strategy', None)
    channel_axis = perchannel_args.channel_axis
    num_channels = perchannel_args.num_channels
    rank = perchannel_args.rank
    if num_channels == -1:
        raise RuntimeError("Please provide channel number for observer.")
    perchannel_observer = MinMaxPerChannel(axis=channel_axis,
                                           output_channel=num_channels,
                                           data_rank=rank,
                                           strategy=strategy)
    return perchannel_observer


class PTQCell(QuantCell):
    """Wrapper Cell to PTQCell with FakeQuantizer"""

    @abc.abstractmethod
    def calibrate(self):
        """calibrate"""
        raise NotImplementedError

    @staticmethod
    def antiquant_strategy(weight_strategy=None):
        """antiquant strategy for w8a16"""
        if weight_strategy is None:
            return None
        strategy_len = len(weight_strategy)
        if strategy_len != 2:
            raise RuntimeError(f'strategy length shall be 2, but got {strategy_len}')
        x_strategy = weight_strategy

        anti_strategy = (x_strategy, (), ())
        return anti_strategy

    @staticmethod
    def antiquant_bmm_strategy(act_strategy, weight_strategy, has_bias=False, is_transpose=False):
        """parallel strategy for antiquant bmm"""
        if act_strategy is None or weight_strategy is None:
            return None
        if is_transpose:
            scale_strategy = (weight_strategy[0],)
        else:
            scale_strategy = (weight_strategy[1],)
        offset_strategy = scale_strategy
        if not has_bias:
            return act_strategy, weight_strategy, scale_strategy, offset_strategy
        bias_strategy = scale_strategy
        return act_strategy, weight_strategy, scale_strategy, offset_strategy, bias_strategy


class OqLinearWrapper(PTQCell):
    """Linear layer wrapper with min max"""

    def __init__(self, linear: Linear, policy=None, cfg=None):
        super().__init__(linear, policy)
        if not isinstance(linear, Linear):
            raise ValueError(f'only Linear cell is supported, but got {type(linear)}')
        self._linear = linear
        self._cfg = cfg
        rank = len(linear.weight.shape)
        self._weight_axis = rank - 2 if linear.matmul.transpose_b else rank - 1
        input_fq_args = {}
        weight_fq_args = {}
        self._act_strategy = None
        self._weight_strategy = None
        if "in_strategy" in self._linear.matmul.get_attr_dict():
            input_fq_args["strategy"] = (self._linear.matmul.in_strategy[0],)
            weight_fq_args["strategy"] = (self._linear.matmul.in_strategy[1],)
        act_rank = 2
        self._act_observer = create_observer_perchannel(
            perchannel_args=PerChannelArgs(self._linear.in_channels, -1, act_rank), **input_fq_args)
        weight_observer_axis = 1 if linear.matmul.transpose_b else 0
        self._weight_in_observer = create_observer_perchannel(
            perchannel_args=PerChannelArgs(self._linear.in_channels, weight_observer_axis, rank), **weight_fq_args)
        self._output_quantizer = None
        prex = ""
        for _, param in linear.parameters_and_names():
            prex = param.name.rsplit(".", 1)[0]
        self._act_observer.float_min.data.name = prex + "_act_observer_float_min"
        self._act_observer.float_max.data.name = prex + "_act_observer_float_max"
        self._weight_in_observer.float_min.data.name = prex + "_weight_in_observer_float_min"
        self._weight_in_observer.float_max.data.name = prex + "_weight_in_observer_float_max"
        self._input_scale = Parameter(Tensor([1.0] * self._linear.in_channels), name=f'{prex}_input_scale')
        self._scale_store = Parameter(Tensor([1.0] * self._linear.in_channels), name=f'{prex}_scale_store')
        self.smoothscale = Parameter(Tensor([1.0] * self._linear.in_channels), name=f'{prex}_smoothscale')
        self.smoothscale_store = Parameter(Tensor([1.0] * self._linear.in_channels), name=f'{prex}_smoothscale_store')
        self.upbound_factor = Parameter(Tensor(np.ones((self._linear.out_channels, 1))*1, dtype.float32),
                                        name="f{prex}_om_upbound_factor")
        self.lowbound_factor = Parameter(Tensor(np.ones((self._linear.out_channels, 1))*1, dtype.float32),
                                         name="f{prex}_om_lowbound_factor")
        self.tempweight = Parameter(Tensor(np.ones(self._linear.weight.shape), dtype.int8), name="f{prex}_temp_weight")
        self.tempweight_store = Parameter(Tensor(np.ones(self._linear.weight.shape), dtype.int8),
                                          name="f{prex}_temp_weight_store")
        self.scale = Parameter(Tensor(np.ones((self._linear.out_channels, 1)), dtype.float32), name="f{prex}_scale")
        self.zp = Parameter(Tensor(np.ones((self._linear.out_channels, 1)), dtype.float32), name="f{prex}_zp")
        self.scale_store = Parameter(Tensor(np.ones((self._linear.out_channels, 1)), dtype.float32),
                                     name="f{prex}_scale_store")
        self.zp_store = Parameter(Tensor(np.ones((self._linear.out_channels, 1)), dtype.float32),
                                  name="f{prex}_zp_store")
        self._alpha = 0.5
        self._expand = P.ExpandDims()
        self._act_mul = P.Mul()
        self._weight_mul = P.Mul()
        self._weight_div = P.Div()
        self._act_pow = P.Pow()
        self._weight_pow = P.Pow()
        self._pow_div = P.Div()
        self._div = P.Div()
        self._sub = P.Sub()
        self._cast = P.Cast()
        self._assign = P.Assign()
        self._weight_assign = P.Assign()
        self.loss_fn = nn.MSELoss()
        self.use_temporary_parameter = False
        self.convert_flag = False
        self._factor_mul = P.Mul()
        self._add = P.Add()
        self.sigmoid = nn.Sigmoid()
        self._is_infer = False
        self.scale_store_x = None
        self.zp_store_x = None

    def _create_scale_param(self, rank, name):
        """create_scale_param"""
        bmm_rank = 3
        if rank == bmm_rank:
            param = Parameter(Tensor([[[1.0] * self._linear.in_channels]], dtype=dtype.float32), name=name)
        else:
            param = Parameter(Tensor([[1.0] * self._linear.in_channels], dtype=dtype.float32), name=name)
        return param

    def _calc_input_scale(self):
        """calc_input_scale"""
        act_max = P.Maximum()(P.Abs()(self._act_observer.float_max), P.Abs()(self._act_observer.float_min))
        input_max_pow = self._act_pow(act_max, self._alpha)
        weight_max = P.Maximum()(P.Abs()(self._weight_in_observer.float_max),
                                 P.Abs()(self._weight_in_observer.float_min))
        weight_max_pow = self._weight_pow(weight_max, 1 - self._alpha)
        input_scale = self._pow_div(input_max_pow, weight_max_pow).clip(1e-5)

        # set 0 or nan to 1.0 to avoid quantization error
        input_scale[input_max_pow == 0] = 1.0
        input_scale[weight_max_pow == 0] = 1.0
        return input_scale

    def _adjust_parameter(self):
        self._assign(self._input_scale, self._scale_store.data)
        weight_scale = self._expand(self._input_scale, 0)
        if not self._linear.transpose_b:
            weight_scale = weight_scale.transpose()
        orin_dtype = self._linear.weight.dtype
        weight = self._weight_mul(self._linear.weight, weight_scale)
        weight = self._linear.cast(weight, orin_dtype)
        self._weight_assign(self._linear.weight, weight)

    def weight_quantizer(self):
        return self._weight_quantizer

    def calibrate(self):
        if hasattr(self._linear, "dtype"):
            weight = self._linear.cast(self._linear.weight, self._linear.dtype)
        else:
            weight = self._linear.weight
        self._weight_quantizer(weight)

    def core_construct(self, *args):
        pass

    def set_use_temporary_parameter(self):
        """set_use_temporary_parameter"""
        self.use_temporary_parameter = True

    def paramstore(self):
        """
        This function store temp param.
        """
        self._assign(self.tempweight_store, self.tempweight)
        self._assign(self.scale_store, self.scale)
        self._assign(self.zp_store, self.zp)
        self._assign(self.smoothscale_store, self.smoothscale)
        self._assign(self.scale_store_x, self.scale_x)
        self._assign(self.zp_store_x, self.zp_x)

    def paramconfirm(self):
        """
        This function confirm param to infer.
        """
        self._is_infer = True
        self._linear.weight = Parameter(Tensor(self.tempweight_store, dtype=dtype.int8), name=self._linear.weight.name)
        del self.tempweight_store
        del self.tempweight
        del self.scale
        del self.zp
        del self.upbound_factor
        del self.lowbound_factor
        del self.smoothscale

    def cal_param(self, input_min, input_max, quant_min, quant_max, symmetric):
        """
        This function calculate scale and zp.
        """
        if input_min.shape != input_max.shape:
            raise ValueError("input min shape should be equal to input max.")
        if (input_max == input_min).all():
            return np.ones(input_min.shape), np.zeros(input_min.shape)
        if symmetric:
            input_max = ops.maximum(ops.abs(input_min), ops.abs(input_max))
            input_min = -input_max
        input_max = Tensor(input_max)
        input_min = Tensor(input_min)
        scale = self._div(self._sub(input_max, input_min), self._sub(quant_max, quant_min))
        zp_double = self._sub(quant_min, self._div(input_min, scale))
        zp = Tensor.int(ops.round(zp_double))
        return scale, zp

    def smoothandquanttemp(self):
        """
        fakequant.
        """
        quant_min, quant_max = get_quant_min_max(num_bits=8, signed=True)
        #x量化 scale zp
        temp_x = self._act_mul(self.x, self._div(1.0, self.smoothscale))
        shape = temp_x.shape[0]
        min_values_out_x = ops.reshape(Tensor(temp_x.min(axis=1)), (shape, 1))
        max_values_out_x = ops.reshape(Tensor(temp_x.max(axis=1)), (shape, 1))
        symmetric = True
        scale_x, zp_x = self.cal_param(min_values_out_x, max_values_out_x, quant_min, quant_max, symmetric)
        t_scale_x = ops.reshape(Tensor(scale_x), (shape, 1))
        t_zp_x = ops.reshape(Tensor(zp_x), (shape, 1))
        self.scale_x = t_scale_x
        self.zp_x = t_zp_x
        #weight 量化
        weight = self._weight_mul(self._linear.weight, self.smoothscale)
        if self._linear.has_bias:
            bias = self._linear.bias
        min_values_out = weight.min(axis=1)
        max_values_out = weight.max(axis=1)
        min_values_out = ops.reshape(Tensor(min_values_out), (self._linear.out_channels, 1))
        max_values_out = ops.reshape(Tensor(max_values_out), (self._linear.out_channels, 1))
        input_mins = self._factor_mul(min_values_out, self.lowbound_factor)
        input_maxs = self._factor_mul(max_values_out, self.upbound_factor)
        symmetric = True
        scale, zp = self.cal_param(input_mins, input_maxs, quant_min, quant_max, symmetric)
        t_scale = ops.reshape(Tensor(scale), (self._linear.out_channels, 1))
        t_zp = ops.reshape(Tensor(zp), (self._linear.out_channels, 1))
        quant_weight = quant_tensor_data(weight, scale, zp, quant_min, quant_max, self._weight_axis)
        self.tempweight = quant_weight
        self.scale = t_scale
        self.zp = t_zp
        if self._linear.has_bias:
            temp_bias = quant_bias_data(bias, scale)
        temp_weight = self._sub(quant_weight, t_zp)
        temp_weight = self._weight_mul(temp_weight, t_scale)
        temp_weight = self._cast(temp_weight, dtype.float16)
        if self._linear.has_bias:
            temp_bias = self._weight_mul(temp_bias, t_scale)
        return temp_x, temp_weight, temp_bias

    def construct(self, x):
        """
        Defines the computation of LinearWithMinMax to be performed.

        Returns:
            Tensor, returns the computed result.
        """
        out_shape = P.Shape()(x)[:-1] + (self._linear.out_channels,)
        x = P.Reshape()(x, (-1, self._linear.in_channels))
        if self._is_infer:
            x = self._act_mul(x, self._div(1.0, self.smoothscale_store))
            temp_x = x
            quant_min, quant_max = get_quant_min_max(num_bits=8, signed=True)
            quant_x = quant_tensor_data(temp_x, self.scale_store_x, self.zp_store_x,
                                        quant_min, quant_max, self._weight_axis)
            x = self._sub(quant_x, self.zp_store_x)
            x = self._weight_mul(x, self.scale_store_x)
            x = self._cast(x, dtype.float16)
            temp_weight = self._sub(self._linear.weight, self.zp_store)
            temp_weight = self._cast(temp_weight, dtype.float16)
            temp_weight = self._weight_mul(temp_weight, self.scale_store)
            temp_weight = self._cast(temp_weight, dtype.float16)
            weight = temp_weight
        else:
            if self.use_temporary_parameter:
                temp_x, temp_weight, temp_bias = self.smoothandquanttemp()
                if self._linear.has_bias:
                    bias = temp_bias
                if hasattr(self._linear, "dtype"):
                    weight = self._linear.cast(temp_weight, self._linear.dtype)
                    x = self._linear.cast(temp_x, self._linear.dtype)
            else:
                self.x = x
                self.scale_store_x = Parameter(Tensor(np.ones((x.shape[0], 1)), dtype.float32),
                                               name="f{prex}_scale_store_x")
                self.zp_store_x = Parameter(Tensor(np.ones((x.shape[0], 1)), dtype.float32),
                                            name="f{prex}_zp_store_x")
                self.scale_x = Parameter(Tensor(np.ones((x.shape[0], 1)), dtype.float32),
                                         name="f{prex}_scale_x")
                self.zp_x = Parameter(Tensor(np.ones((x.shape[0], 1)), dtype.float32),
                                      name="f{prex}_zp_x")
                weight = self._linear.weight
                weight = self._weight_in_observer(weight)
                x = self._act_observer(x)
                self.smoothscale = self._calc_input_scale()
                self.smoothscale = self._cast(self.smoothscale, dtype.float16)
                if hasattr(self._linear, "dtype"):
                    weight = self._linear.cast(weight, self._linear.dtype)
                    x = self._linear.cast(x, self._linear.dtype)
                else:
                    weight = self._linear.weight
                    x = self._linear.cast(x, self._linear.weight.dtype)
        x = self._linear.matmul(x, weight)
        if self._linear.has_bias:
            if hasattr(self._linear, "dtype"):
                bias = self._linear.cast(self._linear.bias, self._linear.dtype)
            else:
                bias = self._linear.cast(self._linear.bias, x.dtype)
            x = self._linear.bias_add(x, bias)
        if self._linear.activation_flag:
            x = self._linear.activation(x)
        out = P.Reshape()(x, out_shape)
        return out
