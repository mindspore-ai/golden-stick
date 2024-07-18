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
import time
import numpy as np
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore import Parameter, Tensor, dtype
from mindspore.common.initializer import initializer
from mindformers.modules.layers import Linear

from mindspore_gs.common.gs_enum import BackendTarget
from mindspore_gs.common import logger
from mindspore_gs.quantization.fake_quantizer import LinearFakeQuantizer
from mindspore_gs.quantization.quant_utils import get_quant_min_max, quant_tensor_data
from mindspore_gs.quantization.layer_policy import LayerPolicy, PerChannelArgs
from mindspore_gs.ptq.quant_cells import PTQCell
from mindspore_gs.ptq.convert_utils import (
    convert_to_fusion_antiquant, convert_to_quant, convert_to_dequant,
    convert_to_fusion_antiquant_for_deploy
)


class LinearQuant(PTQCell):
    """Linear layer wrapper with min max"""

    def __init__(self, linear: Linear, policy: LayerPolicy):
        super(LinearQuant, self).__init__(linear, policy)
        self._linear = linear
        rank = len(linear.weight.shape)
        self._weight_axis = rank - 2 if linear.matmul.transpose_b else rank - 1
        input_fq_args = {}
        weight_perchannel_args = PerChannelArgs(self._linear.out_channels, self._weight_axis, rank)
        weight_fq_args = {}
        self._act_strategy = None
        self._weight_strategy = None
        if "in_strategy" in self._linear.matmul.get_attr_dict():
            self._act_strategy = self._linear.matmul.in_strategy[0]
            self._weight_strategy = self._linear.matmul.in_strategy[1]
            input_fq_args["strategy"] = (self._linear.matmul.in_strategy[0],)
            weight_fq_args["strategy"] = (self._weight_strategy,)
        self._input_quantizer = self._policy.get_input_quantizer(input_index=0, **input_fq_args)
        self._output_quantizer = None
        self._weight_quantizer = self._policy.get_weight_quantizer(self._linear.weight.name, weight_perchannel_args,
                                                                   **weight_fq_args)

        prex = ""
        for _, param in linear.parameters_and_names():
            prex = param.name.rsplit(".", 1)[0]
        if self._input_quantizer:
            self._input_quantizer.float_min.data.name = prex + "_input_float_min"
            self._input_quantizer.float_max.data.name = prex + "_input_float_max"
        self._weight_quantizer.float_min.data.name = prex + "_weight_float_min"
        self._weight_quantizer.float_max.data.name = prex + "_weight_float_max"

        has_dtype = hasattr(self._linear, "dtype")
        self._cast_dtype = self._linear.dtype if has_dtype else self._linear.weight.dtype
        self._quant_deployed = False

    def weight_quantizer(self):
        return self._weight_quantizer

    def core_construct(self, *args):
        pass

    def convert(self, backend: str = BackendTarget.NONE, is_deploy=False):
        if backend == BackendTarget.NONE:
            super(LinearQuant, self).convert(backend)
            if self._weight_quantizer:
                self._weight_quantizer = self._weight_quantizer.convert_to_fakequantparam()
            return
        if backend == BackendTarget.ASCEND:
            weight_only = isinstance(self._weight_quantizer, LinearFakeQuantizer) and \
                        self._weight_quantizer.get_attr("weight_only_quant", False)
            all_quant = isinstance(self._weight_quantizer, LinearFakeQuantizer) and \
                        isinstance(self._input_quantizer, LinearFakeQuantizer)
            if not all_quant and not weight_only:
                logger.info(f"LinearQuant {self} is not quanted.")
                return

            super(LinearQuant, self).convert(backend)
            # quant weight to int8
            self._weight_quantizer = self._weight_quantizer.convert_to_fakequantparam()
            weight_quantizer: P.FakeQuantParam = self._weight_quantizer.fq
            weight = self._linear.cast(self._linear.weight, self._cast_dtype)
            quant_min, quant_max = get_quant_min_max(
                weight_quantizer.attrs[LinearFakeQuantizer.attr_key_num_bits],
                weight_quantizer.attrs[LinearFakeQuantizer.attr_key_symmetric],
                weight_quantizer.attrs[LinearFakeQuantizer.attr_key_narrow_range])
            scale = weight_quantizer.attrs[P.FakeQuantParam.attr_key_linear_quant_scale]
            zp = weight_quantizer.attrs[P.FakeQuantParam.attr_key_linear_quant_zero_point]
            weight_quant = quant_tensor_data(weight, np.squeeze(np.array(scale)), np.squeeze(np.array(zp)),
                                             quant_min, quant_max, self._weight_axis, dtype.int8)
            np_weight_quant = weight_quant.asnumpy()
            del weight_quant
            self._linear.weight = Parameter(Tensor(np_weight_quant, dtype=dtype.int8),
                                            name=self._linear.weight.name)
            if not all_quant:
                self._input_quantizer = None
                self._output_quantizer = None
                self._weight_quantizer = convert_to_fusion_antiquant(
                    self._weight_quantizer, transpose_weight=self._linear.transpose_b,
                    dst_dtype=self._cast_dtype, strategy=
                    self.antiquant_bmm_strategy(self._act_strategy, self._weight_strategy,
                                                False, is_transpose=self._linear.transpose_b)
                )
                self._quant_deployed = True
            else:
                self._output_quantizer = convert_to_dequant(self._input_quantizer, self._weight_quantizer)
                self._input_quantizer = convert_to_quant(self._input_quantizer)
                self._quant_deployed = True
                raise RuntimeError(f'current version not support all quantization, only for weight quantization')

    def calibrate(self):
        """calibrate for weight quant"""
        start = time.time()
        self._weight_quantizer(self._linear.weight)
        logger.info(
            f"Calibrated weight of Linear Cell: {self._linear.weight.name}, time cost: {time.time() - start} s.")

    # pylint: disable=W0221
    def construct(self, x):
        """
        Defines the computation of LinearWithMinMax to be performed.

        Returns:
            Tensor, returns the computed result.
        """
        out_shape = P.Shape()(x)[:-1] + (self._linear.out_channels,)
        x = P.Reshape()(x, (-1, self._linear.in_channels))
        if hasattr(self._linear, "expert_flag") and self._linear.expert_flag:
            if self._linear.use_expert_group_size is True:
                x = P.Reshape()(x, (-1, self._linear.expert_num, self._linear.expert_group_size,
                                    self._linear.in_channels))
            else:
                x = P.Reshape()(x, (self._linear.outer_batch, self._linear.expert_num, -1, self._linear.in_channels))
        ori_dtype = F.dtype(x)

        x = self._linear.cast(x, self._cast_dtype)
        weight = self._linear.cast(self._linear.weight, self._cast_dtype)
        if not self._quant_deployed:
            weight = self._weight_quantizer(weight)
        x = self._linear.matmul(x, weight)
        if self._linear.has_bias:
            x = self._linear.bias_add(x, self._linear.cast(self._linear.bias, self._linear.dtype))
        if self._linear.activation_flag:
            x = self._linear.activation(x)
        x = F.cast(x, ori_dtype)
        output = P.Reshape()(x, out_shape)
        return output


class LinearDeploy(PTQCell):
    """Linear layer wrapper with min max"""

    def __init__(self, linear: Linear, policy: LayerPolicy):
        super(LinearDeploy, self).__init__(linear, policy)
        if not isinstance(linear, Linear):
            raise ValueError(f'only Linear cell is supported, but got {type(linear)}')
        self._linear = linear

        rank = len(linear.weight.shape)
        self._weight_axis = rank - 2 if linear.matmul.transpose_b else rank - 1
        has_dtype = hasattr(self._linear, "dtype")
        self._cast_dtype = self._linear.dtype if has_dtype else self._linear.weight.dtype
        self._act_strategy = None
        self._weight_strategy = None
        if "in_strategy" in self._linear.matmul.get_attr_dict():
            self._act_strategy = self._linear.matmul.in_strategy[0]
            self._weight_strategy = self._linear.matmul.in_strategy[1]
        self._linear.weight = Parameter(initializer('ones', self._linear.weight.shape, dtype.int8),
                                        name=self._linear.weight.name)
        self._weight_quantizer = convert_to_fusion_antiquant_for_deploy(
            axis=self._weight_axis, output_channel=self._linear.out_channels,
            data_rank=len(self._linear.weight.shape),
            is_per_channel=True,
            transpose_weight=self._linear.transpose_b,
            dst_dtype=self._cast_dtype,
            strategy=self.antiquant_bmm_strategy(self._act_strategy, self._weight_strategy,
                                                 False, is_transpose=self._linear.transpose_b)
            )

    def weight_quantizer(self):
        return self._weight_quantizer

    def core_construct(self, *args):
        pass

    def calibrate(self):
        raise ValueError("Inner error, should not invoke LinearDeploy.calibrate().")

    # pylint: disable=W0221
    def construct(self, x):
        """
        Defines the computation of LinearWithMinMax to be performed.

        Returns:
            Tensor, returns the computed result.
        """
        out_shape = P.Shape()(x)[:-1] + (self._linear.out_channels,)
        x = P.Reshape()(x, (-1, self._linear.in_channels))
        if hasattr(self._linear, "expert_flag") and self._linear.expert_flag:
            if self._linear.use_expert_group_size is True:
                x = P.Reshape()(x, (-1, self._linear.expert_num, self._linear.expert_group_size,
                                    self._linear.in_channels))
            else:
                x = P.Reshape()(x, (self._linear.outer_batch, self._linear.expert_num, -1, self._linear.in_channels))
        ori_dtype = F.dtype(x)

        x = self._linear.cast(x, self._cast_dtype)
        x = self._weight_quantizer(x, self._linear.weight)
        if self._linear.has_bias:
            x = self._linear.bias_add(x, self._linear.cast(self._linear.bias, self._linear.dtype))
        if self._linear.activation_flag:
            x = self._linear.activation(x)
        x = F.cast(x, ori_dtype)
        output = P.Reshape()(x, out_shape)
        return output
