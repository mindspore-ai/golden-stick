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
"""SlbQuantConvert."""

import numpy as np
import mindspore
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from .slb_quant import Conv2dSlbQuant
from ..simulated_quantization.simulated_quantization_convert import Conv2dWithFQWeight


def create_conv2d_from_conv2dquant(conv2dquant: Conv2dSlbQuant, **quant_params):
    """
    A method to create `Conv2d` from a `Conv2dSlbQuant` with quant_params.
    """
    if conv2dquant.bias is None:
        bias = None
    else:
        bias = conv2dquant.bias.value()

    argmax = P.Argmax()
    onehot = P.OneHot()
    reduce_sum = P.ReduceSum()
    true_tensor = Tensor(1, mindspore.float32)
    false_tensor = Tensor(0, mindspore.float32)
    num_bits = quant_params['weight_num_bits']

    if num_bits == 1:
        w_list = Parameter(Tensor([-1, 1], mindspore.float32).view(1, 1, 1, 1, -1),
                           name='w_list', requires_grad=False)
    else:
        w_list_init = np.linspace(-1, 1, 2**num_bits)
        w_list = Parameter(Tensor(w_list_init, mindspore.float32).view(1, 1, 1, 1, -1),
                           name='w_list', requires_grad=False)

    # Convert 5d weight to 4d weight
    conv2dquant_weights5d = conv2dquant.weight # 5d
    # Compute one-hot representation of matrix A's argmax
    onehot_weights5d = onehot(argmax(conv2dquant_weights5d), conv2dquant_weights5d.shape[-1], true_tensor, false_tensor)
    # Compute continuous weights
    weights_5d = onehot_weights5d * w_list
    weights_4d = reduce_sum(weights_5d, -1)

    conv = Conv2dWithFQWeight(
        conv2dquant.in_channels,
        conv2dquant.out_channels,
        kernel_size=conv2dquant.kernel_size,
        stride=conv2dquant.stride,
        pad_mode=conv2dquant.pad_mode,
        padding=conv2dquant.padding,
        dilation=conv2dquant.dilation,
        group=conv2dquant.group,
        weight_init=weights_4d.value(),
        weight_name=conv2dquant.weight.name,
        has_bias=conv2dquant.has_bias,
        bias_init=bias,
        bias_name=conv2dquant)
    for key, value in quant_params.items():
        conv.conv2d.add_prim_attr(key, Tensor(value))
    return conv
