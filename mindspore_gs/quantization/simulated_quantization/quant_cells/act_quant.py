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
"""ActQuant."""
from __future__ import absolute_import

from mindspore.nn.cell import Cell
from mindspore_gs.validator import Validator
from mindspore_gs.quantization.quant_cell import QuantCell
from mindspore_gs.quantization.layer_policy import LayerPolicy


class ActQuant(QuantCell):
    r"""
    Quantization aware training activation function.

    Add the fake quantized operation to the end of activation operation, by which the output of activation
    operation will be truncated. For more details about Quantization, please refer to the implementation
    of subclass of `FakeQuantWithMinMaxObserver`, :class:`mindspore.nn.FakeQuantWithMinMaxObserver`.

    Args:
        activation (Cell): Activation cell.

    Inputs:
        - **x** (Tensor) - The input of ActQuant. The input dimension is preferably 2D or 4D.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If `activation` is not an instance of Cell.
        TypeError: If `fake_before` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore.compression import quant
        >>> from mindspore import Tensor
        >>> qconfig = quant.create_quant_config()
        >>> act_quant = nn.ActQuant(nn.ReLU(), quant_config=qconfig)
        >>> x = Tensor(np.array([[1, 2, -1], [-2, 0, -1]]), mindspore.float32)
        >>> result = act_quant(x)
        >>> print(result)
        [[0.9882355 1.9764705 0.       ]
         [0.        0.        0.       ]]
    """

    def __init__(self, activation, policy: LayerPolicy):
        """Initialize ActQuant."""
        super(ActQuant, self).__init__(activation, policy)
        self.act = Validator.check_isinstance("activation", activation, Cell)

    def weight_quantizer(self):
        return self._weight_quantizer

    # pylint: disable=arguments-differ
    def core_construct(self, x):
        """construct."""
        return self.act(x)

    def get_origin(self):
        """get_origin."""
        return self.act
