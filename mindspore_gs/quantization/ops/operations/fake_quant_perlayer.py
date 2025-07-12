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
MindSpore golden stick simulated-quantization ops FakeQuantPerLayer.
"""
import os

from mindspore.ops import DataType
from mindspore.ops.functional import zeros_like
from mindspore_gs.common.validator import Validator as validator
from mindspore_gs.common.validator import Rel
from mindspore_gs.common import logger
from mindspore_gs.ops import GSCustom, custom_op_attr_register
from .grad_operations import FakeQuantPerLayerGrad


class FakeQuantPerLayer(GSCustom):
    r"""
    Simulates the quantize and dequantize operations in training time.

    Args:
        num_bits (int) : Number bits for quantization aware. Default: 8.
        ema (bool): Uses EMA algorithm update value min and max. Default: False.
        ema_decay (int) : EMA algorithm decay parameter. Default: 0.999.
        quant_delay (int): Quantilization delay parameter. Before delay step in training time not update
            simulate quantization aware function. After delay step in training time begin simulate the aware
            quantize function. Default: 0.
        symmetric (bool): Whether the quantization algorithm is symmetric or not. Default: False.
        narrow_range (bool): Whether the quantization algorithm uses narrow range or not. Default: False.
        training (bool): Training the network or not. Default: True.

    Inputs:
        - **x** (Tensor) : float32 Tensor representing the shape of the output tensor.
        - **min** (Tensor) : Value of the min range of the input data x.
        - **max** (Tensor) : Value of the max range of the input data x.

    Outputs:
        - Tensor: Simulates quantize tensor of x.

    Examples:
        >>> import numpy as np
        >>> from mindspore.common import dtype as mstype
        >>> from mindspore import Tensor
        >>> input_tensor = Tensor(np.random.rand(3, 16, 5, 5), mstype.float32)
        >>> min_tensor = Tensor(np.array([-6]), mstype.float32)
        >>> max_tensor = Tensor(np.array([6]), mstype.float32)
        >>> output_tensor = FakeQuantPerLayer(num_bits=8)(input_tensor, min_tensor, max_tensor)
    """
    support_quant_bit = [4, 7, 8]

    @custom_op_attr_register
    def __init__(self,
                 num_bits=8,
                 ema=False,
                 ema_decay=0.999,
                 quant_delay=0,
                 symmetric=False,
                 narrow_range=False,
                 training=True):
        """Initialize FakeQuantPerLayer OP"""
        support_device = ["GPU"]
        self._check_support_device_target(support_device)
        if num_bits not in self.support_quant_bit:
            raise ValueError(
                f"For '{self.get_custom_op_name()}' attr \'num_bits\' is not support.")
        if ema and not ema_decay:
            raise ValueError(
                f"For '{self.get_custom_op_name()}' attr \'ema\' and \'ema_decay\' should set together.")

        validator.check_value_type('ema', ema, (bool,), self.get_custom_op_name())
        validator.check_value_type('symmetric', symmetric, (bool,), self.get_custom_op_name())
        validator.check_value_type('narrow_range', narrow_range, (bool,), self.get_custom_op_name())
        validator.check_value_type('training', training, (bool,), self.get_custom_op_name())
        validator.check_float_range(ema_decay, 0, 1, Rel.INC_BOTH, 'ema_decay', self.get_custom_op_name())
        validator.check_positive_int(num_bits, 'num_bits', self.get_custom_op_name())
        validator.check_non_negative_int(quant_delay, 'quant_delay', self.get_custom_op_name())

    def _infer_shape(self, x, x_min, x_max):
        """infer_shape."""
        return x

    def _infer_dtype(self, x, x_min, x_max):
        """infer_dtype."""
        return x

    def _get_op_bprop(self):
        """Bprop func."""
        fqperlayer_bprop = FakeQuantPerLayerGrad(
            num_bits=self._get_custom_attr("num_bits"),
            quant_delay=self._get_custom_attr("quant_delay"),
            symmetric=self._get_custom_attr("symmetric"),
            narrow_range=self._get_custom_attr("narrow_range")
        )

        def bprop(x, x_min, x_max, _, dout):
            """Bprop func."""
            dx = fqperlayer_bprop(dout, x, x_min, x_max)
            return dx, zeros_like(x_min), zeros_like(x_max)
        return bprop

    def _get_op_input_names(self) -> (str,):
        """set_op_input_names"""
        return "x", "min_val", "max_val"

    def _get_op_output_names(self) -> (str,):
        """set_op_output_names"""
        return ("y",)

    def _get_op_dtype_formats(self) -> [[DataType]]:
        """set_op_dtype_format"""
        return [[DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default]]

    def _get_forward_func(self) -> str:
        """
        Automatically generate farward func according to class name.

        Returns:
            Farward func, a string represent '{dir_path}/{file_name}:{func_name}'.
        """
        dir_path = os.path.dirname(os.path.abspath(__file__))
        func_path = os.path.join(dir_path, "../kernel/gpu/fake_quant_per_layer_impl.cu")
        func_name = "Custom" + self.get_custom_op_name()
        if not os.path.exists(func_path):
            error_str = f"For {self.get_custom_op_name()}, cu file not exist, the path is {func_path}"
            logger.error(error_str)
            raise RuntimeError(error_str)
        logger.info(f"Custom op {self.get_custom_op_name()} func: {func_path}:{func_name}")
        return func_path + ":" + func_name
