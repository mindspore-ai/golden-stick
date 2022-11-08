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
"""Simulated Quantization Convert Utils."""

import numpy as np

from mindspore.nn import Cell
from mindspore.common import Tensor
from mindspore.ops import operations as P
from mindspore.ops.operations import _quant_ops as Q
from mindspore.common import dtype as mstype
from mindspore.common.dtype import QuantDtype
from mindspore_gs.ops.nn import Conv2dQuant, DenseQuant, Conv2dBnFoldQuantOneConv, Conv2dBnWithoutFoldQuant, \
    Conv2dBnFoldQuant
from ..quantize_wrapper_cell import QuantizeWrapperCell
from ..quant_utils import fold_batchnorm, without_fold_batchnorm
from .simulated_fake_quantizers import SimulatedFakeQuantizerPerChannel


class CellBlockWithFakeWeight(Cell):
    """A block of Conv/Dense, activation layer with fake quantization weight for export MINDIR model.

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
        network (Cell): SimulatedQuantizationAwareTraining apply network.

    Returns:
        Cell, Infer network.
    """

    def __init__(self, network):
        self.quant_dtype = QuantDtype.INT8
        self.network = network

    def run(self):
        """Start to convert."""
        self.network.update_cell_prefix()
        return self._convert_quant2deploy(self.network)

    def _get_quant_block(self, cell_core):
        """convert network's quant subcell to deploy subcell"""
        scale_w, zp_w = self.__get_quant_param(cell_core)
        activation = None

        # get op
        if isinstance(cell_core, DenseQuant):
            op_core = P.MatMul()
            activation = cell_core.activation
        else:
            op_core = cell_core.conv

        # get the `weight` and `bias`
        weight, bias = self.__get_weight_bias(cell_core)
        is_per_channel = isinstance(
            cell_core.fake_quant_weight, SimulatedFakeQuantizerPerChannel)
        quant_params = {"quant_dtype": self.quant_dtype,
                        "is_per_channel": is_per_channel}
        block = CellBlockWithFakeWeight(op_core, tuple(scale_w), tuple(
            zp_w), weight, bias, activation, quant_params)
        return block

    def __get_quant_param(self, cell_core):
        """Get scale and bias for fake quant weight"""
        _, _, scale_w, zp_w = cell_core.fake_quant_weight.extract_quant_param()
        return scale_w, zp_w

    def __get_weight_bias(self, cell_core):
        """Get weight and bias for quantizaiton"""
        weight = cell_core.weight.data.asnumpy()
        bias = None
        if isinstance(cell_core, (Conv2dQuant, DenseQuant)):
            if cell_core.has_bias:
                bias = cell_core.bias.data.asnumpy()
        elif isinstance(cell_core, (Conv2dBnFoldQuant, Conv2dBnFoldQuantOneConv)):
            weight, bias = fold_batchnorm(weight, cell_core)
        elif isinstance(cell_core, Conv2dBnWithoutFoldQuant):
            weight, bias = without_fold_batchnorm(weight, cell_core)

        if isinstance(cell_core, DenseQuant):
            weight = np.transpose(weight)

        weight_tensor = Tensor(weight)
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
        if isinstance(cell_core, (Conv2dBnFoldQuant, Conv2dBnFoldQuantOneConv, Conv2dBnWithoutFoldQuant, Conv2dQuant,
                                  DenseQuant)):
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
                fake_quant_input = subcell.get_input_quantizer().convert_to_fakequantparam()
                fake_quant_output = subcell.get_output_quantizer().convert_to_fakequantparam()
                subcell.insert_child_to_cell(
                    "_input_quantizer", fake_quant_input)
                subcell.insert_child_to_cell(
                    "_output_quantizer", fake_quant_output)
            else:
                self._convert_quant2deploy(subcell)
        return network
