# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
MindSpore golden stick simulated-quantization ops FakeQuantPerChannel.
"""

import os

from mindspore import context
from mindspore.common import dtype as mstype
import mindspore.ops as ops
from mindspore.ops import Custom
from mindspore.ops import DataType, CustomRegOp
from mindspore.ops.functional import zeros_like
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel


class FakeQuantPerChannel(Custom):
    r"""
    Simulates the quantize and dequantize operations in training time base on per channel.

    Args:
        num_bits (int) : Number bits to quantilization. Default: 8.
        ema (bool): Uses EMA algorithm update tensor min and tensor max. Default: False.
        ema_decay (int) : EMA algorithm decay parameter. Default: 0.999.
        quant_delay (int): Quantilization delay  parameter. Before delay step in training time not
            update the weight data to simulate quantize operation. After delay step in training time
            begin simulate the quantize operation. Default: 0.
        symmetric (bool): Whether the quantization algorithm is symmetric or not. Default: False.
        narrow_range (bool): Whether the quantization algorithm uses narrow range or not. Default: False.
        training (bool): Training the network or not. Default: True.
        channel_axis (int): Quantization by channel axis. Ascend backend only supports 0 or 1. Default: 1.

    Inputs:
        - **x** (Tensor) : 4-D float32 Tensor representing the shape of the output tensor.
        - **min** (int, float) : Value of the min range of the input data.
        - **max** (int, float) : Value of the max range of the input data.

    Outputs:
        - Tensor, has the same type as input.

    Examples:
        >>> fake_quant = FakeQuantPerChannel()
        >>> input_x = Tensor(np.array([3, 4, 5, -2, -3, -1]).reshape(3, 2), mindspore.float32)
        >>> _min = Tensor(np.linspace(-2, 2, 12).reshape(3, 2, 2), mindspore.float32)
        >>> _max = Tensor(np.linspace(8, 12, 12).reshape(3, 2, 2), mindspore.float32)
        >>> result = fake_quant(input_x, _min, _max)
    """
    support_quant_bit = [4, 7, 8]
    ascend_support_x_rank = [2, 4]

    def __init__(self,
                 num_bits=8,
                 ema=False,
                 ema_decay=0.999,
                 quant_delay=0,
                 symmetric=False,
                 narrow_range=False,
                 training=True,
                 channel_axis=1):
        """Initialize FakeQuantPerChannel OP"""
        name = self.__class__.__name__
        if not context.get_context('device_target') == "GPU":
            raise NotImplementedError("For 'FakeQuantPerChannel', it is only supported on GPU right now.")
        if num_bits not in self.support_quant_bit:
            raise ValueError(
                f"For '{name}' Attr \'num_bits\' is not support.")
        if ema and not ema_decay:
            raise ValueError(
                f"For '{name}' attr \'ema\' and \'ema_decay\' should set together.")

        ema = validator.check_value_type('ema', ema, (bool,), name)
        symmetric = validator.check_value_type(
            'symmetric', symmetric, (bool,), name)
        narrow_range = validator.check_value_type(
            'narrow_range', narrow_range, (bool,), name)
        training = validator.check_value_type(
            'training', training, (bool,), name)
        ema_decay = validator.check_float_range(ema_decay, 0, 1, Rel.INC_BOTH, 'ema_decay', name)
        num_bits = validator.check_positive_int(num_bits, 'num_bits', name)
        quant_delay = validator.check_non_negative_int(quant_delay, 'quant_delay', name)
        channel_axis = validator.check_non_negative_int(channel_axis, 'channel_axis', name)

        init_args = FakeQuantPerChannel._init_aot_custom_ops((num_bits, quant_delay, symmetric, narrow_range, training))
        super(FakeQuantPerChannel, self).__init__(
            func=init_args['func'],
            out_shape=init_args['shape'],
            out_dtype=init_args['type'],
            bprop=init_args['bprop'],
            reg_info=init_args['reg_info'],
            func_type='aot'
        )

    @staticmethod
    def _init_aot_custom_ops(args):
        """Register ops."""
        num_bits, quant_delay, symmetric, narrow_range, training = args

        fake_quant_per_channel_gpu_info = CustomRegOp("fake_quant_per_channel_impl_kernel") \
            .input(0, "x") \
            .input(1, "min_val") \
            .input(2, "max_val") \
            .output(0, "y") \
            .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
            .attr("num_bits", "required", "float", value=float(num_bits)) \
            .attr("quant_delay", "required", "float", value=float(quant_delay)) \
            .attr("symmetric", "required", "bool", value=symmetric) \
            .attr("narrow_range", "required", "bool", value=narrow_range) \
            .attr("training", "required", "bool", value=training) \
            .target("GPU") \
            .get_op_info()

        fake_quant_per_channel_bprop_gpu_info = CustomRegOp("fake_quant_per_channel_grad_impl_kernel") \
            .input(0, "gradient") \
            .input(1, "x") \
            .input(2, "min_val") \
            .input(3, "max_val") \
            .output(0, "output") \
            .dtype_format(DataType.F32_Default, DataType.F32_Default,
                          DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
            .attr("num_bits", "required", "float", value=float(num_bits)) \
            .attr("quant_delay", "required", "float", value=float(quant_delay)) \
            .attr("symmetric", "required", "bool", value=symmetric) \
            .attr("narrow_range", "required", "bool", value=narrow_range) \
            .target("GPU") \
            .get_op_info()

        dir_path = os.path.dirname(os.path.abspath(__file__))
        func_path_bprop = os.path.join(dir_path, "ccsrc", "fake_quant_perchannel_grad_impl.cu")
        fqperchannel_bprop = ops.Custom(
            func_path_bprop + ":CustomFQPerChannelGrad",
            lambda dx, x, x_min, x_max: x,
            mstype.float32,
            "aot",
            reg_info=fake_quant_per_channel_bprop_gpu_info
        )

        def bprop(x, x_min, x_max, out, dout):
            """Bprop func."""
            dx = fqperchannel_bprop(dout, x, x_min, x_max)
            return dx, zeros_like(x_min), zeros_like(x_max)

        func_path = os.path.join(dir_path, "ccsrc", "fake_quant_perchannel_impl.cu")
        return_args = {
            'func': func_path + ":CustomFakeQuantPerChannel",
            'shape': lambda x, x_min, x_max: x,
            'type': mstype.float32,
            'bprop': bprop,
            'reg_info': fake_quant_per_channel_gpu_info
        }
        return return_args
