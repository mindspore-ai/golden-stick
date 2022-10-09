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
MindSpore golden stick simulated-quantization ops MinMaxUpdatePerLayer.
"""

import os

from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.ops import Custom
from mindspore.ops import DataType, CustomRegOp
from mindspore.ops.functional import zeros_like
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel


class MinMaxUpdatePerLayer(Custom):
    r"""
    Updates min and max per layer.

    Args:
        ema (bool): Uses EMA algorithm update value min and max. Default: False.
        ema_decay (int) : EMA algorithm decay parameter. Default: 0.999.

    Inputs:
        - **x** (Tensor) : float32 Tensor representing the shape of the output tensor.
        - **min** (Tensor) : Value of the min range of the input data x.
        - **max** (Tensor) : Value of the max range of the input data x.

    Outputs:
        - Tensor: Simulates quantize tensor of x.

    Examples:
        >>> input_tensor = Tensor(np.random.rand(3, 16, 5, 5), mstype.float32)
        >>> min_tensor = Tensor(np.array([-6]), mstype.float32)
        >>> max_tensor = Tensor(np.array([6]), mstype.float32)
        >>> output_tensor = MinMaxUpdatePerLayer(num_bits=8)(input_tensor, min_tensor, max_tensor)
    """
    support_quant_bit = [4, 7, 8]

    def __init__(self, ema=False, ema_decay=0.999):
        """Initialize FakeQuantMinMaxPerLayerUpdate OP"""
        name = self.__class__.__name__
        if not context.get_context('device_target') == "GPU":
            raise NotImplementedError("For 'MinMaxUpdatePerLayer', it is only supported on GPU right now.")
        if ema and not ema_decay:
            raise ValueError(
                f"For '{name}' attr \'ema\' and \'ema_decay\' should set together.")

        ema = validator.check_value_type('ema', ema, (bool,), name)
        ema_decay = validator.check_float_range(ema_decay, 0, 1, Rel.INC_BOTH, 'ema_decay', name)

        init_args = MinMaxUpdatePerLayer._init_aot_custom_ops((ema, ema_decay))
        super(MinMaxUpdatePerLayer, self).__init__(
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
        ema, ema_decay = args
        minmax_update_per_layer_gpu_info = CustomRegOp("minmax_update_per_layer_impl_kernel") \
            .input(0, "x") \
            .input(1, "min_val") \
            .input(2, "max_val") \
            .output(0, "min_out") \
            .output(1, "max_out") \
            .dtype_format(DataType.None_None, DataType.None_None, DataType.None_None, DataType.None_None,
                          DataType.None_None) \
            .attr("ema", "required", "bool", value=ema) \
            .attr("ema_decay", "required", "float", value=ema_decay) \
            .target("GPU") \
            .get_op_info()

        def bprop(x, x_min, x_max, out, dout):
            """bprop."""
            return zeros_like(x), zeros_like(x_min), zeros_like(x_max)

        dir_path = os.path.dirname(os.path.abspath(__file__))
        func_path = os.path.join(dir_path, "ccsrc", "minmax_update_perlayer_impl.cu")
        return_args = {
            'func': func_path + ":MinmaxUpdatePerLayer",
            'shape': lambda x, x_min, x_max: (x_min, x_max),
            'type': (mstype.float32, mstype.float32),
            'bprop': bprop,
            'reg_info': minmax_update_per_layer_gpu_info
        }
        return return_args
