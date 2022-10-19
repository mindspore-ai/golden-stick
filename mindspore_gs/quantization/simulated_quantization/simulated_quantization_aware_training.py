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
"""SimulatedQuantizationAwareTraining."""

import os
from mindspore.nn import Cell
from mindspore._checkparam import Validator, Rel
from mindspore import context
from mindspore import log as logger
from mindspore.nn.layer.quant import Conv2dQuant, DenseQuant
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from ..quantization_aware_training import QuantizationAwareTraining
from ..constant import QuantDtype
from ..quantize_wrapper_cell import QuantizeWrapperCell
from ..quant_utils import IdentityCell
from .simulated_quantization_net_policy import SimulatedNetPolicy
from .simulated_quantization_config import SimulatedQuantizationConfig
from .simulated_quantization_convert import create_conv2d_from_conv2dquant, create_dense_from_densequant


class SimulatedQuantizationAwareTraining(QuantizationAwareTraining):
    """
    Basic implementation of simulated quantization aware training, this algorithm adopts fake quantizer to simulate
    the loss of quantization calculation, and network parameters are updated through backpropagation, so that the
    network parameters can better adapt to the loss caused by quantization. See more details in `A White Paper on
    Neural Network Quantization <https://arxiv.org/pdf/2106.08295.pdf>_`.

    Args:
        config (dict): store attributes for quantization aware training, keys are attribute names,
            values are attribute values. supported attribute are listed below:

            - quant_delay (Union[int, list, tuple]): Number of steps after which weights and activations are quantized
              during train and eval. The first element represents data flow and the second element represents weights.
              Default: (0, 0).
            - quant_dtype (Union[QuantDtype, list, tuple]): Datatype used to quantize weights and activations. The first
              element represents data flow and the second element represents weights. It is necessary to consider the
              precision support of hardware devices in the practical quantization infer scenaries.
              Default: (QuantDtype.INT8, QuantDtype.INT8).
            - per_channel (Union[bool, list, tuple]):  Quantization granularity based on layer or on channel. If True
              then base on per channel, otherwise base on per layer. The first element represents data flow and the
              second element represents weights, and the first element must be False now.
              Default: (False, False).
            - symmetric (Union[bool, list, tuple]): Whether the quantization algorithm is symmetric or not. If True
              then base on symmetric, otherwise base on asymmetric. The first element represents data flow and the
              second element represents weights.
              Default: (False, False).
            - narrow_range (Union[bool, list, tuple]): Whether the quantization algorithm uses narrow range or not.
              The first element represents data flow and the second element represents weights.
              Default: (False, False).
            - enable_fusion (bool): Whether apply fusion before applying quantization.
              Default: False.
            - freeze_bn (int): Number of steps after which BatchNorm OP parameters fixed to global mean and variance.
              Default: 10000000.
            - bn_fold (bool): Whether to use bn fold ops for simulation inference operation.
              Default: False.
            - one_conv_fold (bool): Whether to use one conv bn fold ops for simulation inference operation.
              Default: True.

    Raises:
        TypeError: If `bn_fold`, `one_conv_fold` or `enable_fusion` is not bool.
        TypeError: If `freeze_bn` is not int.
        TypeError: If `quant_delay` is not int, or every element of `quant_delay` is not int.
        TypeError: If `quant_dtype` is not `QuantDtype`, or every element of `quant_dtype` is not `QuantDtype`.
        TypeError: If `per_channel` is not bool, or every element of `per_channel` is not bool.
        TypeError: If `symmetric` is not bool, or every element of `symmetric` is not bool.
        TypeError: If `narrow_range` is not bool, or every element of `narrow_range` is not bool.
        ValueError: If `freeze_bn` is less than 0.
        ValueError: If the length of `quant_delay`, `quant_dtype`, `per_channel`, `symmetric` or `narrow_range` is not
            less than 2.
        ValueError: If `quant_delay` is less than 0, or any element of `quant_delay` is less than 0.
        TypeError: If `quant_dtype` is not `QuantDtype.INT8`, or any element of `quant_dtype` is not
            `QuantDtype.INT8`.
        ValueError: If `per_channel` is True, or the first element of `per_channel` is True.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> from mindspore_gs.quantization import SimulatedQuantizationAwareTraining
        >>> from mindspore import nn
        ... class NetToQuant(nn.Cell):
        ...     def __init__(self, num_channel=1):
        ...         super(NetToQuant, self).__init__()
        ...         self.conv = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        ...         self.bn = nn.BatchNorm2d(6)
        ...
        ...     def construct(self, x):
        ...         x = self.conv(x)
        ...         x = self.bn(x)
        ...         return x
        ...
        >>> ## 1) Define network to be quantized
        >>> net = NetToQuant()
        >>> ## 2) Define SimQAT Algorithm
        >>> simulated_quantization = SimulatedQuantizationAwareTraining()
        >>> ## 3) Use set functions to change config
        >>> simulated_quantization.set_enable_fusion(True)
        >>> simulated_quantization.set_bn_fold(False)
        >>> simulated_quantization.set_act_quant_delay(900)
        >>> simulated_quantization.set_weight_quant_delay(900)
        >>> simulated_quantization.set_act_per_channel(False)
        >>> simulated_quantization.set_weight_per_channel(True)
        >>> simulated_quantization.set_act_narrow_range(False)
        >>> simulated_quantization.set_weight_narrow_range(False)
        >>> ## 4) Apply SimQAT algorithm to origin network
        >>> net_qat = simulated_quantization.apply(net)
        >>> ## 5) Print network and check the result. Conv2d and Dense should be transformed to QuantizeWrapperCells.
        >>> ## Since we set enable_fusion to be True, bn_fold to be False, the Conv2d and BatchNorm2d Cells are
        >>> ## fused and converted to Conv2dBnWithoutFoldQuant.
        >>> ## Since we set act_quant_delay to be 900, the quant_delay value of _input_quantizer and _output_quantizer
        >>> ## are set to be 900.
        >>> ## Since we set weight_quant_delay to be 900, the quant_delay value of fake_quant_weight are set to be 900.
        >>> ## Since we set act_per_channel to be False, the per_channel value of _input_quantizer and
        >>> ## _output_quantizer are set to be False.
        >>> ## Since we set weight_per_channel to be True, the per_channel value of fake_quant_weight are set to be
        >>> ## True.
        >>> ## Since we set act_narrow_range to be False, the narrow_range value of _input_quantizer and
        >>> ## _output_quantizer are set to be False.
        >>> ## Since we set weight_narrow_range to be False, the narrow_range value of fake_quant_weight are set to be
        >>> ## True.
        >>> print(net_qat)
        NetToQuantOpt<
          (_handler): NetToQuant<
            (conv): Conv2d<input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
            (bn): BatchNorm2d<num_features=6, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=_handler.bn.gamma, shape=(6,), dtype=Float32, requires_grad=True), beta=Parameter (name=_handler.bn.beta, shape=(6,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=_handler.bn.moving_mean, shape=(6,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=_handler.bn.moving_variance, shape=(6,), dtype=Float32, requires_grad=False)>
            >
          (Conv2dBnWithoutFoldQuant): QuantizeWrapperCell<
            handler: in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False, input quantizer: bit_num=8, symmetric=False, narrow_range=False, ema=False(0.999), per_channel=False, quant_delay=900, output quantizer: bit_num=8, symmetric=False, narrow_range=False, ema=False(0.999), per_channel=False, quant_delay=900
            (_handler): Conv2dBnWithoutFoldQuant<
              in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False
              (fake_quant_weight): SimulatedFakeQuantizerPerChannel<bit_num=8, symmetric=True, narrow_range=False, ema=False(0.999), per_channel=True(0, 6), quant_delay=900>
              (batchnorm): BatchNorm2d<num_features=6, eps=1e-05, momentum=0.0030000000000000027, gamma=Parameter (name=Conv2dBnWithoutFoldQuant._handler.batchnorm.gamma, shape=(6,), dtype=Float32, requires_grad=True), beta=Parameter (name=Conv2dBnWithoutFoldQuant._handler.batchnorm.beta, shape=(6,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=Conv2dBnWithoutFoldQuant._handler.batchnorm.moving_mean, shape=(6,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=Conv2dBnWithoutFoldQuant._handler.batchnorm.moving_variance, shape=(6,), dtype=Float32, requires_grad=False)>
              >
            (_input_quantizer): SimulatedFakeQuantizerPerLayer<bit_num=8, symmetric=False, narrow_range=False, ema=False(0.999), per_channel=False, quant_delay=900>
            (_output_quantizer): SimulatedFakeQuantizerPerLayer<bit_num=8, symmetric=False, narrow_range=False, ema=False(0.999), per_channel=False, quant_delay=900>
            >
          >
    """

    def __init__(self, config=None):
        super(SimulatedQuantizationAwareTraining, self).__init__(config)
        self._is_cpu = context.get_context('device_target') == "CPU"
        if config is None:
            config = {}
        Validator.check_value_type("config", config, [dict], self.__class__.__name__)
        self._config = None
        self._create_qconfig_by_dict(config)
        self._qat_policy = self._init_net_policy(self._config)
        self._custom_transforms = {}
        self._custom_layer_policy_map = {}
        if "custom_transforms" in config.keys():
            self._custom_transforms = config["custom_transforms"]
        if "custom_policies" in config.keys():
            self._custom_layer_policy_map = config["custom_policies"]

    def set_bn_fold(self, bn_fold):
        """
        Set value of bn_fold of quantization aware training `config`

        Args:
            bn_fold (bool): Whether quantization algorithm use bn_fold or not.

        Raises:
            TypeError: If `bn_fold` is not bool.
        """
        Validator.check_bool(bn_fold, "bn_fold", self.__class__.__name__)
        self._config.bn_fold = bn_fold
        if self._is_cpu and self._config.enable_fusion and self._config.bn_fold and not self._config.one_conv_fold:
            logger.warning("Current device target is CPU, and Current config: enable_fusion=True, bn_fold=True, "
                           "one_conv_fold=False, this may lead to replacing Conv2d + BatchNorm2d pattern with "
                           "Conv2dBnFoldQuant which is not implemented in CPU backend!")

    def set_one_conv_fold(self, one_conv_fold):
        """
        Set value of one_conv_fold of quantization aware training `config`

        Args:
            one_conv_fold (bool): Whether quantization algorithm use one_conv_fold or not.

        Raises:
            TypeError: If `one_conv_fold` is not bool.
        """
        Validator.check_bool(one_conv_fold, "one_conv_fold", self.__class__.__name__)
        self._config.one_conv_fold = one_conv_fold
        if self._is_cpu and self._config.enable_fusion and self._config.bn_fold and not self._config.one_conv_fold:
            logger.warning("Current device target is CPU, and Current config: enable_fusion=True, bn_fold=True, "
                           "one_conv_fold=False, this may lead to replacing Conv2d + BatchNorm2d pattern with "
                           "Conv2dBnFoldQuant which is not implemented in CPU backend!")

    def set_act_quant_delay(self, act_quant_delay):
        """
        Set value of act_quant_delay of quantization aware training `config`

        Args:
            act_quant_delay (int): Number of steps after which activation is quantized during train and eval.

        Raises:
            TypeError: If `act_quant_delay` is not int.
            ValueError: act_quant_delay is less than 0.
        """
        Validator.check_is_int(act_quant_delay, "act_quant_delay", self.__class__.__name__)
        Validator.check_int(act_quant_delay, 0, Rel.GE, "act_quant_delay", self.__class__.__name__)
        self._config.act_quant_delay = act_quant_delay

    def set_weight_quant_delay(self, weight_quant_delay):
        """
        Set value of weight_quant_delay of quantization aware training `config`

        Args:
            weight_quant_delay (int): Number of steps after which weight is quantized during train and eval.

        Raises:
            TypeError: If `weight_quant_delay` is not int.
            ValueError: weight_quant_delay is less than 0.
        """
        Validator.check_is_int(weight_quant_delay, "weight_quant_delay", self.__class__.__name__)
        Validator.check_int(weight_quant_delay, 0, Rel.GE, "weight_quant_delay", self.__class__.__name__)
        self._config.weight_quant_delay = weight_quant_delay

    def set_act_per_channel(self, act_per_channel):
        """
        Set value of act_per_channel of quantization aware training `config`

        Args:
            act_per_channel (bool): Quantization granularity based on layer or on channel. If True then base on
                per channel, otherwise base on per layer. Only support False now.

        Raises:
            TypeError: If `act_per_channel` is not bool.
            ValueError: Only supported if `act_per_channel` is False yet.
        """
        Validator.check_bool(act_per_channel, "act_per_channel", self.__class__.__name__)
        if act_per_channel:
            raise ValueError(f'Only supported if `act_per_channel` is False yet.')
        self._config.act_per_channel = act_per_channel

    def set_weight_per_channel(self, weight_per_channel):
        """
        Set value of weight_per_channel of quantization aware training `config`

        Args:
            weight_per_channel (bool): Quantization granularity based on layer or on channel. If True then base on
                per channel, otherwise base on per layer.

        Raises:
            TypeError: If `weight_per_channel` is not bool.
        """
        Validator.check_bool(weight_per_channel, "weight_per_channel", self.__class__.__name__)
        self._config.weight_per_channel = weight_per_channel

    def set_act_quant_dtype(self, act_quant_dtype):
        """
        Set value of act_quant_dtype of quantization aware training `config`

        Args:
            act_quant_dtype (QuantDtype): Datatype used to quantize activations.

        Raises:
            TypeError: If `act_quant_dtype` is not QuantDtype.
            TypeError: Only supported if `act_quant_dtype` is `QuantDtype.INT8` yet.
        """
        if not isinstance(act_quant_dtype, QuantDtype):
            raise TypeError(f'The parameter `act quant dtype` must be isinstance of QuantDtype, '
                            f'but got {act_quant_dtype}.')
        if act_quant_dtype != QuantDtype.INT8:
            raise TypeError("Only supported if `act_quant_dtype` is `QuantDtype.INT8` yet.")
        self._config.act_quant_dtype = act_quant_dtype

    def set_weight_quant_dtype(self, weight_quant_dtype):
        """
        Set value of weight_quant_dtype of quantization aware training `config`

        Args:
            weight_quant_dtype (QuantDtype): Datatype used to quantize weight.

        Raises:
            TypeError: If `weight_quant_dtype` is not QuantDtype.
            TypeError: Only supported if `weight_quant_dtype` is `QuantDtype.INT8` yet.
        """
        if not isinstance(weight_quant_dtype, QuantDtype):
            raise TypeError(f'The parameter `weight quant dtype` must be isinstance of QuantDtype, '
                            f'but got {weight_quant_dtype}.')
        if weight_quant_dtype != QuantDtype.INT8:
            raise TypeError("Only supported if `weight_quant_dtype` is `QuantDtype.INT8` yet.")
        self._config.weight_quant_dtype = weight_quant_dtype

    def set_act_symmetric(self, act_symmetric):
        """
        Set value of act_symmetric of quantization aware training `config`

        Args:
            act_symmetric (bool): Whether the quantization algorithm use act symmetric or not. If True then base on
                symmetric, otherwise base on asymmetric.

        Raises:
            TypeError: If `act_symmetric` is not bool.
        """
        Validator.check_bool(act_symmetric, "act_symmetric", self.__class__.__name__)
        self._config.act_symmetric = act_symmetric

    def set_weight_symmetric(self, weight_symmetric):
        """
        Set value of weight_symmetric of quantization aware training `config`

        Args:
            weight_symmetric (bool): Whether the quantization algorithm use weight symmetric or not. If True then
                base on symmetric, otherwise base on asymmetric.

        Raises:
            TypeError: If `weight_symmetric` is not bool.
        """
        Validator.check_bool(weight_symmetric, "weight_symmetric", self.__class__.__name__)
        self._config.weight_symmetric = weight_symmetric

    def set_act_narrow_range(self, act_narrow_range):
        """
        Set value of act_narrow_range of quantization aware training `config`

        Args:
            act_narrow_range (bool): Whether the quantization algorithm use act narrow_range or not. If True then
                base on narrow_range, otherwise base on not narrow_range.

        Raises:
            TypeError: If `act_narrow_range` is not bool.
        """
        Validator.check_bool(act_narrow_range, "act_narrow_range", self.__class__.__name__)
        self._config.act_narrow_range = act_narrow_range

    def set_weight_narrow_range(self, weight_narrow_range):
        """
        Set value of weight_narrow_range of quantization aware training `config`

        Args:
            weight_narrow_range (bool): Whether the quantization algorithm use weight narrow_range or not. If
                True then base on narrow_range, otherwise base on not narrow_range.

        Raises:
            TypeError: If `weight_narrow_range` is not bool.
        """
        Validator.check_bool(weight_narrow_range, "weight_narrow_range", self.__class__.__name__)
        self._config.weight_narrow_range = weight_narrow_range

    def set_freeze_bn(self, freeze_bn):
        """
        Set value of freeze_bn of quantization aware training `config`

        Args:
            freeze_bn (int): Number of steps after which BatchNorm OP parameters fixed to global mean and variance.

        Raises:
            TypeError: If `freeze_bn` is not int.
            ValueError: `freeze_bn` is less than 0.
        """
        Validator.check_is_int(freeze_bn, "freeze_bn", self.__class__.__name__)
        Validator.check_int(freeze_bn, 0, Rel.GE, "freeze_bn", self.__class__.__name__)
        self._config.freeze_bn = freeze_bn

    def set_enable_fusion(self, enable_fusion):
        """
        Set value of enable_fusion of quantization aware training `config`

        Args:
            enable_fusion (bool): Whether apply fusion before applying quantization, default is False.

        Raises:
            TypeError: If `enable_fusion` is not bool.
        """
        Validator.check_bool(enable_fusion, "enable_fusion", self.__class__.__name__)
        self._config.enable_fusion = enable_fusion
        if self._is_cpu and self._config.enable_fusion and self._config.bn_fold and not self._config.one_conv_fold:
            logger.warning("Current device target is CPU, and Current config: enable_fusion=True, bn_fold=True, "
                           "one_conv_fold=False, this may lead to replacing Conv2d + BatchNorm2d pattern with "
                           "Conv2dBnFoldQuant which is not implemented in CPU backend!")

    @staticmethod
    def _convert2list(name, value):
        if not isinstance(value, list) and not isinstance(value, tuple):
            value = [value]
        elif len(value) > 2:
            raise ValueError("input `{}` len should less then 2".format(name))
        return value

    def _init_net_policy(self, config):
        return SimulatedNetPolicy(config)

    def _create_qconfig_by_dict(self, config: dict):
        """Create quantization aware training `config` from a dict"""
        self._config = SimulatedQuantizationConfig()
        quant_delay_list = SimulatedQuantizationAwareTraining._convert2list("quant delay",
                                                                            config.get("quant_delay", [0, 0]))
        quant_dtype_list = SimulatedQuantizationAwareTraining. \
            _convert2list("quant dtype", config.get("quant_dtype", [QuantDtype.INT8, QuantDtype.INT8]))
        per_channel_list = SimulatedQuantizationAwareTraining._convert2list("per channel",
                                                                            config.get("per_channel", [False, True]))
        symmetric_list = SimulatedQuantizationAwareTraining. \
            _convert2list("symmetric", config.get("symmetric", [False, True]))
        narrow_range_list = SimulatedQuantizationAwareTraining. \
            _convert2list("narrow range", config.get("narrow_range", [False, False]))

        self.set_act_quant_delay(quant_delay_list[0])
        self.set_weight_quant_delay(quant_delay_list[-1])

        self.set_act_quant_dtype(quant_dtype_list[0])
        self.set_weight_quant_dtype(quant_dtype_list[-1])

        self.set_act_per_channel(per_channel_list[0])
        self.set_weight_per_channel(per_channel_list[-1])

        self.set_act_symmetric(symmetric_list[0])
        self.set_weight_symmetric(symmetric_list[-1])

        self.set_act_narrow_range(narrow_range_list[0])
        self.set_weight_narrow_range(narrow_range_list[-1])

        self.set_enable_fusion(config.get("enable_fusion", False))
        self.set_bn_fold(config.get("bn_fold", False))
        self.set_one_conv_fold(config.get("one_conv_fold", True))
        self.set_freeze_bn(config.get("freeze_bn", 10000000))

    def apply(self, network: Cell) -> Cell:
        """
        Apply SimQAT Algorithm on `network`, use the following steps to make `network` available for quantization aware
        training:

        1. Fuse certain cells in `network` using pattern engine which is defined by net policy. Default fuse pattern:
           Conv2d + BatchNorm2d + ReLU, Conv2d + ReLU, Dense + BatchNorm2d + ReLU, Dense + BatchNorm2d, Dense + ReLU.
        2. Propagate LayerPolicies defined in NetPolicy through network.
        3. Reduce redundant fake quantizers which means two or more fake quantizers existing on one tensor.
        4. Apply LayerPolicies to convert normal cell to `QuantizeWrapperCell`. We will insert real fake quantizer
           into network in this step.

        Args:
            network (Cell): Network to be quantized.

        Returns:
            Quantized network.
        """
        self._qat_policy.build()
        return super(SimulatedQuantizationAwareTraining, self).apply(network)

    def convert(self, net_opt: Cell, ckpt_path="") -> Cell:
        """
        Define how to convert a SimQAT network to a standard network before exporting to MindIR.

        Args:
            net_opt (Cell): SimQAT network to be converted.
            ckpt_path (str): Path to checkpoint file for `net_opt`. Default is a empty string which means not loading
                checkpoint file to `net_opt`.

        Raises:
            TypeError: If `net_opt` is not Cell.
            TypeError: If `ckpt_path` is not string.

        Returns:
            An instance of Cell represents converted network.
        """
        if not isinstance(net_opt, Cell):
            raise TypeError(
                f'The parameter `net_opt` must be isinstance of Cell, but got {type(net_opt)}.')
        if not isinstance(ckpt_path, str):
            raise TypeError(
                f'The parameter `ckpt_path` must be isinstance of str, but got {type(ckpt_path)}.')
        ckpt_path = os.path.realpath(ckpt_path)
        if os.path.isfile(ckpt_path):
            param_dict = load_checkpoint(ckpt_path)
            load_param_into_net(net_opt, param_dict)
        self._visit_cell(net_opt)
        return net_opt

    def _visit_cell(self, network: Cell):
        """
        Visit sub_cell of network recursively, replace Simulated Quantization OP with standard OP.
        """
        cells = network.name_cells()
        for _, sub_cell in cells.items():
            if sub_cell == network:
                continue
            if isinstance(sub_cell, QuantizeWrapperCell):
                quant_handle = sub_cell.get_handler()
                _, _, weight_scale, weight_zp = quant_handle.fake_quant_weight.extract_quant_param()
                _, _, input_scale, input_zp = sub_cell.get_input_quantizer().extract_quant_param()
                _, _, output_scale, output_zp = sub_cell.get_output_quantizer().extract_quant_param()
                quant_params = {'weight_scale': weight_scale, 'weight_zp': weight_zp, 'input_scale': input_scale,
                                'input_zp': input_zp, 'output_scale': output_scale, 'output_zp': output_zp}
                if isinstance(quant_handle, Conv2dQuant):
                    conv = create_conv2d_from_conv2dquant(
                        quant_handle, **quant_params)
                    sub_cell.insert_child_to_cell("_handler", conv)
                elif isinstance(quant_handle, DenseQuant):
                    dense = create_dense_from_densequant(
                        quant_handle, **quant_params)
                    sub_cell.insert_child_to_cell("_handler", dense)
                else:
                    raise NotImplementedError(f"Not supported {type(quant_handle)} for convert api yet. \
                        Please set `enable_fusion` to False for SimulatedQuantizationAwareTraining.")
                identity = IdentityCell()
                sub_cell.insert_child_to_cell("_input_quantizer", identity)
                sub_cell.insert_child_to_cell("_output_quantizer", identity)
            else:
                self._visit_cell(sub_cell)
