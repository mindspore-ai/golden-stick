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
"""fake quant wrapper cells for mindone."""


from mindspore import nn, dtype as msdtype
from mindspore import ops

from mindspore_gs.ptq.context import InnerPTQConfig


class Quant(nn.Cell):
    """Quant"""
    def __init__(self, dst_type=msdtype.int8):
        super().__init__()
        self.dst_type = dst_type

    def construct(self, x, scale, offset):
        x = x / scale
        x = ops.round(x) + offset
        x = ops.clip(x, -128., 127.)
        return ops.cast(x, self.dst_type)


class SmoothQuant(nn.Cell):
    """SmoothQuant"""
    def __init__(self, dst_type=msdtype.int8):
        super().__init__()
        self.dst_type = dst_type
        self.quant = Quant(dst_type)

    def construct(self, x, smooth_scale, scale, offset):
        # FIXME hangangqiang2@huawei.com
        # Theoretically, weights should be multiplied by scale, while activations should be divided by scale.
        # However, during deployment, smooth_scale is inverted (reciprocal taken) after apply_scale_to_weight.
        # Since weights have already been multiplied by scale, activations must still be multiplied by scale here.
        x = x * smooth_scale
        return self.quant(x, scale, offset)


class DeQuant(nn.Cell):
    """DeQuant"""
    def __init__(self, dst_type):
        super().__init__()
        self.dst_type = dst_type

    def construct(self, x, scale, offset):
        x = (x - offset) * scale
        return x.astype(self.dst_type)


class FakeQuant(nn.Cell):
    """FakeQuant"""
    def __init__(self, quant_dtype, dst_dtype):
        super().__init__()
        self.quant = Quant(quant_dtype)
        self.de_quant = DeQuant(dst_dtype)

    def construct(self, x, scale, offset):
        x = self.quant(x, scale, offset)
        x = self.de_quant(x, scale, offset)
        return x


class SmoothFakeQuant(nn.Cell):
    """SmoothFakeQuant"""
    def __init__(self, quant_dtype, dst_dtype):
        super().__init__()
        self.quant = SmoothQuant(quant_dtype)
        self.de_quant = DeQuant(dst_dtype)

    def construct(self, x, smooth_scale, scale, offset):
        x = self.quant(x, smooth_scale, scale, offset)
        x = self.de_quant(x, scale, offset)
        x = x / smooth_scale
        return x


class DynamicQuant(nn.Cell):
    """DynamicQuant"""
    def __init__(self, dst_type=msdtype.int8):
        super().__init__()
        self.dst_type = dst_type

    def construct(self, x):
        x_max = ops.max(x, axis=1)[0]
        x_min = ops.min(x, axis=1)[0]
        x_max = ops.maximum(ops.abs(x_max), ops.abs(x_min))
        scale = ops.mul(x_max, 2) / (127 + 128)
        scale = ops.reshape(scale, (-1,))
        x = x / scale
        x = ops.round(x)
        x = ops.clip(x, -128., 127.)
        return ops.cast(x, self.dst_type), scale


class DynamicDeQuant(nn.Cell):
    """DeQuant"""
    def __init__(self, dst_type):
        super().__init__()
        self.dst_type = dst_type

    def construct(self, x, scale):
        x = x * scale
        return x.astype(self.dst_type)

class DynamicFakeQuant(nn.Cell):
    """DynamicFakeQuant"""
    def __init__(self, quant_dtype, dst_dtype=msdtype.int8):
        super().__init__()
        self.quant = DynamicQuant(quant_dtype)
        self.de_quant = DynamicDeQuant(dst_dtype)

    def construct(self, x):
        x, scale = self.quant(x)
        x = self.de_quant(x, scale)
        return x


class GMMDeQuant(nn.Cell):
    """DeQuant"""
    def __init__(self, dst_type):
        super().__init__()
        self.dst_type = dst_type

    def construct(self, x, scale, offset):
        scale = scale.expand_dims(1)
        offset = offset.expand_dims(1)
        x = (x - offset) * scale
        return x.astype(self.dst_type)


class FakeQuantLinearCell(nn.Cell):
    """FakeQuantLinearCell"""
    # pylint: disable=unused-argument
    def __init__(self, layer_name, linear: nn.Dense, context, cfg: InnerPTQConfig):
        super().__init__()
        self.layer_name = layer_name
        self.layer = linear
        self.context = context
        self.cfg = cfg
        self.compute_dtype = linear.weight.dtype
        self.dense = ops.Dense()

    def dequant_input(self, x, weight):
        """process input"""
        return x, weight

    # pylint: disable=unused-argument
    def construct(self, x):
        """linear deploy construct"""
        x, weight = self.dequant_input(x, self.weight)

        bias = None
        if self.layer.has_bias:
            bias = self.bias
        return self.dense(x, weight, bias)
