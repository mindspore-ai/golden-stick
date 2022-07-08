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
"""Basic implementation of simulated quantization aware training, this algorithm adopts fake quantizer to simulate
quantization, statistic min max of ops to be quantized through training procession, then calculates quantization
factors after training. See more details in `A White Paper on Neural Network Quantization
<https://arxiv.org/pdf/2106.08295.pdf>`. """

from mindspore.nn import Cell
from mindspore._checkparam import Validator, Rel
from ..quantization_aware_training import QuantizationAwareTraining
from ..constant import QuantDtype
from .simulated_quantization_net_policy import SimulatedNetPolicy
from .simulated_quantization_config import SimulatedQuantizationConfig


class SimulatedQuantizationAwareTraining(QuantizationAwareTraining):
    """
    Derived class of GoldenStick. Simulated QAT-algorithm.

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
        TypeError: If the element of `quant_delay` is not int.
        TypeError: If the element of `per_channel`, `symmetric`, `narrow_range`, `bn_fold`, `one_conv_fold` is not bool.
        TypeError: If the element of `quant_dtype` is not `QuantDtype`.
        TypeError: If `freeze_bn` is not int.
        ValueError: `freeze_bn` is less than 0.
        ValueError: If the length of `quant_delay`, `quant_dtype`, `per_channel`, `symmetric` or `narrow_range` is not
            less than 2.
        ValueError: If the element of `quant_delay` is less than 0.
        ValueError: If the first element of `per_channel` is True.
        NotImplementedError: If the element of `quant_dtype` is not `QuantDtype.INT8`.


    Supported Platforms:
        ``GPU``


    Examples:
        >>> from mindspore_gs.quantization.simulated_quantization import SimulatedQuantizationAwareTraining
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
        Set value of bn_fold of `_config`

        Args:
            bn_fold (bool): Whether quantization algorithm use bn_fold or not.

        Raises:
            TypeError: If `bn_fold` is not bool.
        """
        Validator.check_bool(bn_fold, "bn_fold", self.__class__.__name__)
        self._config.bn_fold = bn_fold

    def set_one_conv_fold(self, one_conv_fold):
        """
        Set value of one_conv_fold of `_config`

        Args:
            one_conv_fold (bool): Whether quantization algorithm use one_conv_fold or not.

        Raises:
            TypeError: If `one_conv_fold` is not bool.
        """
        Validator.check_bool(one_conv_fold, "one_conv_fold", self.__class__.__name__)
        self._config.one_conv_fold = one_conv_fold

    def set_act_quant_delay(self, act_quant_delay):
        """
        Set value of act_quant_delay of `_config`

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
        Set value of weight_quant_delay of `_config`

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
        Set value of act_per_channel of `_config`

        Args:
            act_per_channel (bool): Quantization granularity based on layer or on channel. If True then base on
                per channel, otherwise base on per layer. Only support False now.

        Raises:
            TypeError: If `act_per_channel` is not bool.
            NotImplementedError: Only supported if `act_per_channel` is False yet.
        """
        Validator.check_bool(act_per_channel, "act_per_channel", self.__class__.__name__)
        if act_per_channel:
            raise NotImplementedError(f'Only supported if `act_per_channel` is False yet.')
        self._config.act_per_channel = act_per_channel

    def set_weight_per_channel(self, weight_per_channel):
        """
        Set value of weight_per_channel of `_config`

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
        Set value of act_quant_dtype of `_config`

        Args:
            act_quant_dtype (QuantDtype): Datatype used to quantize activations.

        Raises:
            TypeError: If `act_quant_dtype` is not QuantDtype.
            NotImplementedError: Only supported if `act_quant_dtype` is `QuantDtype.INT8` yet.
        """
        Validator.check_isinstance("act quant dtype", act_quant_dtype, QuantDtype)
        if act_quant_dtype != QuantDtype.INT8:
            raise NotImplementedError("Only supported if `act_quant_dtype` is `QuantDtype.INT8` yet.")
        self._config.act_quant_dtype = act_quant_dtype

    def set_weight_quant_dtype(self, weight_quant_dtype):
        """
        Set value of weight_quant_dtype of `_config`

        Args:
            weight_quant_dtype (QuantDtype): Datatype used to quantize activations.

        Raises:
            TypeError: If `weight_quant_dtype` is not QuantDtype.
            NotImplementedError: Only supported if `weight_quant_dtype` is `QuantDtype.INT8` yet.
        """
        Validator.check_isinstance("weight quant dtype", weight_quant_dtype, QuantDtype)
        if weight_quant_dtype != QuantDtype.INT8:
            raise NotImplementedError("Only supported if `weight_quant_dtype` is `QuantDtype.INT8` yet.")
        self._config.weight_quant_dtype = weight_quant_dtype

    def set_act_symmetric(self, act_symmetric):
        """
        Set value of act_symmetric of `_config`

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
        Set value of weight_symmetric of `_config`

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
        Set value of act_narrow_range of `_config`

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
        Set value of weight_narrow_range of `_config`

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
        Set value of freeze_bn of `_config`

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
        Set value of enable_fusion of `_config`

        Args:
            enable_fusion (bool): Whether apply fusion before applying quantization, default is False.

        Raises:
            TypeError: If `enable_fusion` is not bool.
        """
        Validator.check_bool(enable_fusion, "enable_fusion", self.__class__.__name__)
        self._config.enable_fusion = enable_fusion

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
        """Create `_config` from a dict"""
        self._config = SimulatedQuantizationConfig()
        quant_delay_list = SimulatedQuantizationAwareTraining._convert2list("quant delay",
                                                                            config.get("quant_delay", [0, 0]))
        quant_dtype_list = SimulatedQuantizationAwareTraining.\
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
        Override from `QuantizationAwareTraining`, apply simulated QAT-Algorithm on `network`.
        """
        self._qat_policy.build()
        return super(SimulatedQuantizationAwareTraining, self).apply(network)
