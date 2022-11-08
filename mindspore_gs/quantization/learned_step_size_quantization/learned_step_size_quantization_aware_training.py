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
"""lsq algorithm"""
from mindspore.nn import Cell
from mindspore._checkparam import Validator
from mindspore.common.dtype import QuantDtype
from ..simulated_quantization.simulated_quantization_aware_training import SimulatedQuantizationAwareTraining as SimQAT
from .learned_step_size_quantization_net_policy import LearnedStepSizeQuantizationNetPolicy as LsqNetPolicy
from .learned_step_size_quantization_config import LearnedStepSizeQuantizationConfig as LsqConfig
from .learned_step_size_fake_quantizers import LearnedStepSizeFakeQuantizerPerLayer as LsqFqPerLayer, \
    LearnedStepSizeFakeQuantizePerChannel as LsqFqPerChannel
from ..quantize_wrapper_cell import QuantizeWrapperCell


class LearnedStepSizeQuantizationAwareTraining(SimQAT):
    """
    Derived class of SimQAT. LSQ quantization algorithm.

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
        ValueError: If the length of `quant_delay`, `quant_dtype`, `per_channel`, `symmetric` or `narrow_range` is not
            less than 2.
        ValueError：　If `freeze_bn` is not 0.
        ValueError: If `quant_delay` is not 0, or any element of `quant_delay` is not 0.
        TypeError: If `quant_dtype` is not `QuantDtype.INT8`, or any element of `quant_dtype` is not `QuantDtype.INT8`.
        ValueError: If `per_channel` is True, or the first element of `per_channel` is True.
        ValueError: If `symmetric` is False, or any element of `symmetric` is False.
        ValueError: If `narrow_range` is False, or any element of `narrow_range` is False.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> from mindspore_gs.quantization.learned_step_size_quantization \
        >>>     import LearnedStepSizeQuantizationAwareTraining
        >>> from mindspore import nn
        >>> from mindspore.common.initializer import Normal
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
        ...
        >>> ## 1) Define network to be quantized
        >>> net = NetToQuant()
        >>> ## 2) Define LSQ Algorithm
        >>> learned_quantization = LearnedStepSizeQuantizationAwareTraining()
        >>> ## 3) Use set functions to change config
        >>> learned_quantization.set_enable_fusion(True)
        >>> learned_quantization.set_bn_fold(False)
        >>> learned_quantization.set_act_symmetric(True)
        >>> learned_quantization.set_weight_symmetric(True)
        >>> learned_quantization.set_act_narrow_range(True)
        >>> learned_quantization.set_weight_narrow_range(True)
        >>> learned_quantization.set_act_quant_delay(0)
        >>> learned_quantization.set_weight_quant_delay(0)
        >>> ## 4) Apply LSQ algorithm to origin network
        >>> net_qat = learned_quantization.apply(net)
        >>> ## 5) Print network and check the result. Conv2d and Dense should be transformed to QuantizeWrapperCells.
        >>> ## Since we set enable_fusion to be True, bn_fold to be False, the Conv2d and BatchNorm2d Cells are
        >>> ## fused and converted to Conv2dBnWithoutFoldQuant.
        >>> ## Since we set act_symmetric to be True, the symmetric value of _input_quantizer and _output_quantizer
        >>> ## are set to be True.
        >>> ## Since we set weight_symmetric to be True, the symmetric value of fake_quant_weight are set to be
        >>> ## True.
        >>> ## Since we set act_narrow_range to be True, the narrow_range value of _input_quantizer and
        >>> ## _output_quantizer are set to be True.
        >>> ## Since we set weight_narrow_range to be True, the narrow_range value of fake_quant_weight are set to be
        >>> ## True.
        >>> ## Since we set act_quant_delay to be 0, the quant_delay value of _input_quantizer and _output_quantizer
        >>> ## are set to be 0.
        >>> ## Since we set weight_quant_delay to be 0, the quant_delay value of fake_quant_weight are set to be 0.
        >>> print(net_qat)
        NetToQuantOpt<
          (_handler): NetToQuant<
            (conv): Conv2d<input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
            (bn): BatchNorm2d<num_features=6, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=_handler.bn.gamma, shape=(6,), dtype=Float32, requires_grad=True), beta=Parameter (name=_handler.bn.beta, shape=(6,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=_handler.bn.moving_mean, shape=(6,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=_handler.bn.moving_variance, shape=(6,), dtype=Float32, requires_grad=False)>
            >
          (Conv2dBnWithoutFoldQuant): QuantizeWrapperCell<
            handler: in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False, input quantizer: bit_num=8, neg_trunc=False, symmetric=True, narrow_range=True, per_channel=False, quant_delay=0, output quantizer: bit_num=8, neg_trunc=False, symmetric=True, narrow_range=True, per_channel=False, quant_delay=0
            (_handler): Conv2dBnWithoutFoldQuant<
              in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False
              (fake_quant_weight): LearnedStepSizeFakeQuantizePerChannel<num_bits=8, symmetric=True, narrow_range=True, neg_trunc=False, per_channel=True(0, 6), quant_delay=0>
              (batchnorm): BatchNorm2d<num_features=6, eps=1e-05, momentum=0.0030000000000000027, gamma=Parameter (name=Conv2dBnWithoutFoldQuant._handler.batchnorm.gamma, shape=(6,), dtype=Float32, requires_grad=True), beta=Parameter (name=Conv2dBnWithoutFoldQuant._handler.batchnorm.beta, shape=(6,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=Conv2dBnWithoutFoldQuant._handler.batchnorm.moving_mean, shape=(6,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=Conv2dBnWithoutFoldQuant._handler.batchnorm.moving_variance, shape=(6,), dtype=Float32, requires_grad=False)>
              >
            (_input_quantizer): LearnedStepSizeFakeQuantizerPerLayer<bit_num=8, neg_trunc=False, symmetric=True, narrow_range=True, per_channel=False, quant_delay=0>
            (_output_quantizer): LearnedStepSizeFakeQuantizerPerLayer<bit_num=8, neg_trunc=False, symmetric=True, narrow_range=True, per_channel=False, quant_delay=0>
            >
          >
    """

    def set_bn_fold(self, bn_fold):
        """
        Set value of bn_fold of `_config`

        Args:
            bn_fold (bool): Whether quantization algorithm use bn_fold or not.

        Raises:
            TypeError: If `bn_fold` is not bool.
        """
        super(LearnedStepSizeQuantizationAwareTraining, self).set_bn_fold(bn_fold)

    def set_one_conv_fold(self, one_conv_fold):
        """
        Set value of one_conv_fold of `_config`

        Args:
            one_conv_fold (bool): Whether quantization algorithm use one_conv_fold or not.

        Raises:
            TypeError: If `one_conv_fold` is not bool.
        """
        super(LearnedStepSizeQuantizationAwareTraining, self).set_one_conv_fold(one_conv_fold)

    def set_act_quant_delay(self, act_quant_delay):
        """
        Set value of act_quant_delay of `_config`

        Args:
            act_quant_delay (int): Number of steps after which activation is quantized during train and eval.

        Raises:
            TypeError: If `act_quant_delay` is not int.
            ValueError:  Learned step size quantization only support `act_quant_delay` is 0 currently.
        """
        Validator.check_is_int(act_quant_delay, "act_quant_delay", self.__class__.__name__)
        if act_quant_delay != 0:
            raise ValueError("Learned step size quantization only support `act_quant_delay` is 0 currently")
        super(LearnedStepSizeQuantizationAwareTraining, self).set_act_quant_delay(act_quant_delay)

    def set_weight_quant_delay(self, weight_quant_delay):
        """
        Set value of weight_quant_delay of `_config`

        Args:
            weight_quant_delay (int): Number of steps after which weight is quantized during train and eval.

        Raises:
            TypeError: If `weight_quant_delay` is not int.
            ValueError:  Learned step size quantization only support `weight_quant_delay` is 0 currently
        """
        Validator.check_is_int(weight_quant_delay, "weight_quant_delay", self.__class__.__name__)
        if weight_quant_delay != 0:
            raise ValueError("Learned step size quantization only support `weight_quant_delay` is 0 currently")
        super(LearnedStepSizeQuantizationAwareTraining, self).set_weight_quant_delay(weight_quant_delay)

    def set_act_per_channel(self, act_per_channel):
        """
        Set value of act_per_channel of `_config`

        Args:
            act_per_channel (bool): Quantization granularity based on layer or on channel. If True then base on
                per channel, otherwise base on per layer. Only support False now.

        Raises:
            TypeError: If `act_per_channel` is not bool.
            ValueError: Only supported if `act_per_channel` is False yet.
        """
        super(LearnedStepSizeQuantizationAwareTraining, self).set_act_per_channel(act_per_channel)

    def set_weight_per_channel(self, weight_per_channel):
        """
        Set value of weight_per_channel of `_config`

        Args:
            weight_per_channel (bool): Quantization granularity based on layer or on channel. If True then base on
                per channel, otherwise base on per layer.

        Raises:
            TypeError: If `weight_per_channel` is not bool.
        """
        super(LearnedStepSizeQuantizationAwareTraining, self).set_weight_per_channel(weight_per_channel)

    def set_act_quant_dtype(self, act_quant_dtype):
        """
        Set value of act_quant_dtype of `_config`

        Args:
            act_quant_dtype (QuantDtype): Datatype used to quantize activations.

        Raises:
            TypeError: If `act_quant_dtype` is not QuantDtype.
            TypeError: Only supported if `act_quant_dtype` is `QuantDtype.INT8` yet.
        """
        super(LearnedStepSizeQuantizationAwareTraining, self).set_act_quant_dtype(act_quant_dtype)

    def set_weight_quant_dtype(self, weight_quant_dtype):
        """
        Set value of weight_quant_dtype of `_config`

        Args:
            weight_quant_dtype (QuantDtype): Datatype used to quantize activations.

        Raises:
            TypeError: If `weight_quant_dtype` is not QuantDtype.
            TypeError: Only supported if `weight_quant_dtype` is `QuantDtype.INT8` yet.
        """
        super(LearnedStepSizeQuantizationAwareTraining, self).set_weight_quant_dtype(weight_quant_dtype)

    def set_act_symmetric(self, act_symmetric):
        """
        Set value of act_symmetric of `_config`

        Args:
            act_symmetric (bool): Whether the quantization algorithm use act symmetric or not. Learned step size
                quantization only support `act_symmetric` is True currently.

        Raises:
            TypeError: If `act_symmetric` is not bool.
            ValueError: If `act_symmetric` is not True.
        """
        Validator.check_bool(act_symmetric, "act_symmetric", self.__class__.__name__)
        if not act_symmetric:
            raise ValueError("Learned step size quantization only support `act_symmetric` is True currently")
        super(LearnedStepSizeQuantizationAwareTraining, self).set_act_symmetric(act_symmetric)

    def set_weight_symmetric(self, weight_symmetric):
        """
        Set value of weight_symmetric of `_config`

        Args:
            weight_symmetric (bool): Whether the quantization algorithm use weight symmetric or not. Learned step size
                quantization only support `weight_symmetric` is True currently.

        Raises:
            TypeError: If `weight_symmetric` is not bool.
            ValueError: If `act_symmetric` is not True.
        """
        Validator.check_bool(weight_symmetric, "weight_symmetric", self.__class__.__name__)
        if not weight_symmetric:
            raise ValueError("Learned step size quantization only support `weight_symmetric` is True currently")
        super(LearnedStepSizeQuantizationAwareTraining, self).set_act_symmetric(weight_symmetric)

    def set_act_narrow_range(self, act_narrow_range):
        """
        Set value of act_narrow_range of `_config`

        Args:
            act_narrow_range (bool): Whether the quantization algorithm use act narrow_range or not. Learned step size
                quantization only support `act_narrow_range` is True currently

        Raises:
            TypeError: If `act_narrow_range` is not bool.
            ValueError: If `act_narrow_range` is not True.
        """
        Validator.check_bool(act_narrow_range, "act_narrow_range", self.__class__.__name__)
        if not act_narrow_range:
            raise ValueError("Learned step size quantization only support `act_narrow_range` is True currently")
        self._config.act_narrow_range = act_narrow_range

    def set_weight_narrow_range(self, weight_narrow_range):
        """
        Set value of weight_narrow_range of `_config`

        Args:
            weight_narrow_range (bool): Whether the quantization algorithm use weight narrow_range or not. Learned step
                size quantization only support `weight_narrow_range` is True currently.

        Raises:
            TypeError: If `weight_narrow_range` is not bool.
            ValueError: If `weight_narrow_range` is not True.
        """
        Validator.check_bool(weight_narrow_range, "weight_narrow_range", self.__class__.__name__)
        if not weight_narrow_range:
            raise ValueError("Learned step size quantization only support `weight_narrow_range` is True currently")
        super(LearnedStepSizeQuantizationAwareTraining, self).set_weight_narrow_range(weight_narrow_range)

    def set_freeze_bn(self, freeze_bn):
        """
        Set value of freeze_bn of `_config`

        Args:
            freeze_bn (int): Number of steps after which BatchNorm OP parameters fixed to global mean and variance.

        Raises:
            TypeError: If `freeze_bn` is not int.
            ValueError: Learned step size quantization only support `freeze_bn` is 0 currently
        """
        Validator.check_is_int(freeze_bn, "freeze_bn", self.__class__.__name__)
        if freeze_bn != 0:
            raise ValueError("Learned step size quantization only support `freeze_bn` is 0 currently")
        super(LearnedStepSizeQuantizationAwareTraining, self).set_freeze_bn(freeze_bn)

    def set_enable_fusion(self, enable_fusion):
        """
        Set value of enable_fusion of `_config`

        Args:
            enable_fusion (bool): Whether apply fusion before applying quantization, default is False.

        Raises:
            TypeError: If `enable_fusion` is not bool.
        """
        super(LearnedStepSizeQuantizationAwareTraining, self).set_enable_fusion(enable_fusion)

    def apply(self, network: Cell) -> Cell:
        """
        Apply LSQ Algorithm on `network`, use the following steps to make `network` available for quantization aware
        training:

        1. Fuse certain cells in `network` using pattern engine which is defined by net policy.
        2. Propagate layer policies defined through cells.
        3. Reduce redundant fake quantizers when they are redundant.
        4. Apply layer policies to convert normal cell to `QuantizeWrapperCell`.

        Args:
            network (Cell): Network to be quantized.

        Returns:
            Quantized network.
        """
        quanted_net = super(LearnedStepSizeQuantizationAwareTraining, self).apply(network)
        self._reset_weights_quantization_params(quanted_net)
        return quanted_net

    def _reset_weights_quantization_params(self, network: Cell):
        for _, cell in network.name_cells().items():
            if isinstance(cell, QuantizeWrapperCell):
                weight_fq = cell.get_handler().fake_quant_weight
                if isinstance(weight_fq, (LsqFqPerLayer, LsqFqPerChannel)):
                    weight_fq.compute_quant_param(cell.get_handler().weight)

    def _init_net_policy(self, config):
        return LsqNetPolicy(config)

    def _create_config(self):
        """Create LsqConfig."""
        self._config = LsqConfig()

    def _update_config_from_dict(self, config: dict):
        quant_dtype_list = SimQAT._convert2list("quant dtype",
                                                config.get("quant_dtype", [QuantDtype.INT8, QuantDtype.INT8]))
        per_channel_list = SimQAT._convert2list("per channel", config.get("per_channel", [False, True]))
        self.set_act_quant_dtype(quant_dtype_list[0])
        self.set_weight_quant_dtype(quant_dtype_list[-1])

        self.set_act_per_channel(per_channel_list[0])
        self.set_weight_per_channel(per_channel_list[-1])

        self.set_act_symmetric(True)
        self.set_weight_symmetric(True)
        self.set_act_quant_delay(0)
        self.set_weight_quant_delay(0)
        self.set_act_narrow_range(True)
        self.set_weight_narrow_range(True)

        self.set_enable_fusion(config.get("enable_fusion", False))
        self.set_bn_fold(config.get("bn_fold", False))
        self.set_one_conv_fold(config.get("one_conv_fold", True))
        self.set_freeze_bn(0)
