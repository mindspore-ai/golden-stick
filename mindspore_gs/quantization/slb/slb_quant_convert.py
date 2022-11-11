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
from mindspore.nn import Cell
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops.operations import _quant_ops as Q
from mindspore.common import dtype as mstype
from mindspore.common.dtype import QuantDtype
from ..quantize_wrapper_cell import QuantizeWrapperCell
from .slb_quant import Conv2dSlbQuant


class CellBlockWithFakeWeight(Cell):
    """A block of Conv, activation layer with fake quantization weight for export MINDIR model.

       Args:
        core_op (Cell): The operation cell.
        scale_w (tuple): The quantization parameter scale of the weight.
        zp_w (tuple): The quantization parameter zero point of the weight.
        weight (Tensor): The weight of the cell.
        bias (Tensor): The bias of the cell. Default: None.
        activation (str): The regularization function applied to the output of the layer, eg. 'relu'. Default: None.
        param_dict (dict): The information of the cell.
    """

    def __init__(self,
                 core_op,
                 scale_w,
                 zp_w,
                 weight,
                 bias=None,
                 activation=None,
                 param_dict=None):

        super(CellBlockWithFakeWeight, self).__init__()
        self.core_op = core_op
        if activation is not None:
            self.core_op.add_prim_attr(
                "activation_name", activation.__class__.__name__)
        if hasattr(core_op, 'pad_mode'):
            self.core_op.add_prim_attr("pad_mode", core_op.pad_mode)

        self.weight = weight
        self.bias = bias
        self.has_bias = bias is not None
        self.activation = activation
        self.has_act = activation is not None
        self.bias_add = P.BiasAdd()
        self.fake_weight = Q.FakeQuantParam.linear_quant_param(quant_dtype=param_dict["quant_dtype"],
                                                               scale=scale_w, zp=zp_w,
                                                               is_per_channel=param_dict["is_per_channel"])

    def construct(self, x):
        weight = self.fake_weight(self.weight)
        if self.has_bias:
            x = self.core_op(x, weight)
            x = self.bias_add(x, self.bias)
        else:
            x = self.core_op(x, weight)
        if self.has_act:
            x = self.activation(x)
        return x

    def extend_repr(self):
        s = f'core_op={type(self.core_op)}, weight=shape[{self.weight.shape}]'
        if self.has_bias:
            s += f', bias=shape[{self.bias.shape}]'
        if self.has_act:
            s += f', activation={self.activation}'
        return s


class ConvertToQuantInferNetwork:
    """
    Convert quantization aware network to infer network.

    Args:
        network (Cell): SlbQuantAwareTraining apply network.

    Returns:
        Cell, Infer network.
    """

    def __init__(self, network, weight_quant_bit):
        if weight_quant_bit == 1:
            self.quant_dtype = QuantDtype.INT1
        elif weight_quant_bit == 2:
            self.quant_dtype = QuantDtype.INT2
        elif weight_quant_bit == 4:
            self.quant_dtype = QuantDtype.INT4
        else:
            raise ValueError("Only support int4|int2|int1 weight quant now!")
        self.weight_quant_bit = weight_quant_bit
        self.network = network

    def run(self):
        """Start to convert."""
        self.network.update_cell_prefix()
        return self._convert_quant2deploy(self.network)

    def _get_quant_block(self, cell_core):
        """convert network's quant subcell to deploy subcell"""
        scale_w, zp_w = self.__get_quant_param()
        activation = None

        # get op
        op_core = cell_core.conv

        # get the `weight` and `bias`
        weight, bias = self.__get_weight_bias(cell_core)
        quant_params = {"quant_dtype": self.quant_dtype,
                        "is_per_channel": False}
        block = CellBlockWithFakeWeight(op_core, tuple(scale_w), tuple(
            zp_w), weight, bias, activation, quant_params)
        return block

    def __get_quant_param(self,):
        """Get scale and bias for fake quant weight"""
        scale_w = np.ones(1) * 2. ** (self.weight_quant_bit - 1)
        zp_w = np.zeros(1)
        return scale_w, zp_w

    def __convert_weight5d_to_weight4d(self, cell_core):
        """Convert slb 5d weight to normal 4d weight"""
        argmax = P.Argmax()
        onehot = P.OneHot()
        reduce_sum = P.ReduceSum()
        true_tensor = Tensor(1, mstype.float32)
        false_tensor = Tensor(0, mstype.float32)
        num_bits = self.weight_quant_bit

        if num_bits == 1:
            w_list = Parameter(Tensor([-1, 1], mstype.float32).view(1, 1, 1, 1, -1),
                               name='w_list', requires_grad=False)
        else:
            w_list_init = np.linspace(-1, 1, 2**num_bits)
            w_list = Parameter(Tensor(w_list_init, mstype.float32).view(1, 1, 1, 1, -1),
                               name='w_list', requires_grad=False)

        # Convert 5d weight to 4d weight
        conv2dquant_weights5d = cell_core.weight
        # Compute one-hot representation of matrix A's argmax
        onehot_weights5d = onehot(argmax(conv2dquant_weights5d), conv2dquant_weights5d.shape[-1],
                                  true_tensor, false_tensor)
        # Compute continuous weights
        weights_5d = onehot_weights5d * w_list
        weights_4d = reduce_sum(weights_5d, -1)
        return weights_4d

    def __get_weight_bias(self, cell_core):
        """Get weight and bias for quantizaiton"""
        weight_tensor = self.__convert_weight5d_to_weight4d(cell_core)
        bias = None
        if isinstance(cell_core, Conv2dSlbQuant):
            if cell_core.has_bias:
                bias = cell_core.bias.data.asnumpy()

        if bias is not None:
            bias_tensor = Tensor(bias, mstype.float32)
            return weight_tensor, bias_tensor
        return weight_tensor, None

    def _convert_subcell(self, cell_core):
        """Convert subcell to ant subcell."""
        if cell_core is not None and hasattr(cell_core, "fake_quant_weight"):
            new_subcell = self._get_quant_block(cell_core)
            return new_subcell
        return None

    def _convert_core_quant_subcell(self, cell_core):
        """Convert subcell for conv and dense."""
        if isinstance(cell_core, (Conv2dSlbQuant)):
            return self._convert_subcell(cell_core)
        raise ValueError("Unsupported quant cell.")

    def _convert_quant2deploy(self, network):
        """Convert network's all quant subcell to deploy subcell."""
        cells = network.name_cells()
        for name in cells:
            subcell = cells[name]
            if subcell == network:
                continue
            if isinstance(subcell, QuantizeWrapperCell):
                quant_cell = subcell.get_handler()
                new_subcell = self._convert_core_quant_subcell(quant_cell)
                subcell.insert_child_to_cell("_handler", new_subcell)
                if subcell.get_input_quantizer() is not None:
                    fake_quant_input = subcell.get_input_quantizer().convert_to_fakequantparam()
                    subcell.insert_child_to_cell("_input_quantizer", fake_quant_input)
                if subcell.get_output_quantizer() is not None:
                    fake_quant_output = subcell.get_output_quantizer().convert_to_fakequantparam()
                    subcell.insert_child_to_cell("_output_quantizer", fake_quant_output)
            else:
                self._convert_quant2deploy(subcell)
        return network
