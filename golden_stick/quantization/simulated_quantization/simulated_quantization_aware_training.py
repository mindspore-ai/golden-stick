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
            - per_channel (Union[bool, list, tuple]):  Quantization granularity based on layer or on channel. If `True`
              then base on per channel, otherwise base on per layer. The first element represents data flow and the
              second element represents weights, and the first element must be `False` now.
              Default: (False, False).
            - symmetric (Union[bool, list, tuple]): Whether the quantization algorithm is symmetric or not. If `True`
              then base on symmetric, otherwise base on asymmetric. The first element represents data flow and the
              second element represents weights.
              Default: (False, False).
            - narrow_range (Union[bool, list, tuple]): Whether the quantization algorithm uses narrow range or not.
              The first element represents data flow and the second element represents weights.
              Default: (False, False).
            - enable_fusion (bool): Whether apply fusion before applying quantization.
            - bn_fold (bool): Whether to use bn fold ops for simulation inference operation.
            - one_conv_fold (bool): Whether to use one conv bn fold ops for simulation inference operation.

    Supported Platforms:
         ``Ascend`` ``GPU``

    Raises:
        TypeError: If the element of `quant_delay` is not int.
        TypeError: If the element of `per_channel`, `symmetric`, `narrow_range`, `bn_fold`, `one_conv_fold` is not bool.
        TypeError: If the element of `quant_dtype` is not `QuantDtype`.
        ValueError: If the length of `quant_delay`, `quant_dtype`, `per_channel`, `symmetric` or `narrow_range` is not
         less than 2.
        ValueError: If the element of `quant_delay` is less than 0.
        ValueError: If the first element of `per_channel` is `True`.
        NotImplementedError: If `bn_fold` is `True`.
        NotImplementedError: If `one_conv_fold` is `False`.
        NotImplementedError: If the element of `quant_dtype` is not `QuantDtype.INT8`.

    Examples:
        >>> from golden_stick.quantization.simulated_quantization import SimulatedQuantizationAwareTraining
        >>> from mindspore import nn
        >>> from mindspore.common.initializer import Normal
        >>> class LeNet5(nn.Cell):
        ...     def __init__(self, num_class=10, num_channel=1):
        ...         super(LeNet5, self).__init__()
        ...         self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        ...         self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        ...         self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        ...         self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        ...         self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        ...         self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        ...         self.flatten = nn.Flatten()
        ...         self.relu = nn.ReLU()
        ...
        ...     def construct(self, x):
        ...         x = self.conv1(x)
        ...         x = self.relu(x)
        ...         x = self.max_pool2d(x)
        ...         x = self.conv2(x)
        ...         x = self.relu(x)
        ...         x = self.max_pool2d(x)
        ...         x = self.flatten(x)
        ...         x = self.relu(self.fc1(x))
        ...         x = self.relu(self.fc2(x))
        ...         x = self.fc3(x)
        ...         return x
        ...
        >>> net = LeNet5()
        >>> simulated_quantization = SimulatedQuantizationAwareTraining()
        >>> net_qat = simulated_quantization.apply(net)
    """

    def __init__(self, config=None):
        super(SimulatedQuantizationAwareTraining, self).__init__(config)
        if config is None:
            config = {}
        Validator.check_value_type("config", config, [dict], self.__class__.__name__)
        self._config: SimulatedQuantizationConfig = SimulatedQuantizationConfig()
        self._update_qconfig_by_dict(config)

        self._qat_policy = SimulatedNetPolicy(self._config)
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
            NotImplementedError: Only supported if `bn_fold` is False yet.
        """
        Validator.check_bool(bn_fold, "bn_fold", self.__class__.__name__)
        if bn_fold:
            raise NotImplementedError(f"Only supported if `bn_fold` is False yet.")
        self._config.bn_fold = bn_fold

    def set_one_conv_fold(self, one_conv_fold):
        """
        Set value of bn_fold of `_config`

        Args:
            one_conv_fold (bool): Whether quantization algorithm use one_conv_fold or not.

        Raises:
            TypeError: If `one_conv_fold` is not bool.
            NotImplementedError: Only supported if `one_conv_fold` is True yet.
        """
        Validator.check_bool(one_conv_fold, "one_conv_fold", self.__class__.__name__)
        if not one_conv_fold:
            raise NotImplementedError(f"Only supported if `one_conv_fold` is True yet.")
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
        Set value of act_quant_delay of `_config`

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
            act_per_channel (bool): Quantization granularity based on layer or on channel. If `True` then base on
                per channel, otherwise base on per layer. Only support `False` now.

        Raises:
            TypeError: If `act_per_channel` is not bool.
            NotImplementedError: Only supported if `act_per_channel` is `False` yet.
        """
        Validator.check_bool(act_per_channel, "act_per_channel", self.__class__.__name__)
        if act_per_channel:
            raise NotImplementedError(f'Only supported if `act_per_channel` is `False` yet.')
        self._config.act_per_channel = act_per_channel

    def set_weight_per_channel(self, weight_per_channel):
        """
        Set value of weight_per_channel of `_config`

        Args:
            weight_per_channel (bool): Quantization granularity based on layer or on channel. If `True` then base on
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
            act_symmetric (bool): Whether the quantization algorithm use act symmetric or not. If `True` then base on
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
            weight_symmetric (bool): Whether the quantization algorithm use weight symmetric or not. If `True` then
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
            act_narrow_range (bool): Whether the quantization algorithm use act narrow_range or not. If `True` then
                base on narrow_range, otherwise base on not narrow_range.

        Raises:
            TypeError: If `act_narrow_range` is not bool.
        """
        Validator.check_bool(act_narrow_range, "act_narrow_range", self.__class__.__name__)
        self._config.act_narrow_range = act_narrow_range

    def set_weight_narrow_range(self, weight_narrow_range):
        """
        Set value of weight_symmetric of `_config`

        Args:
            weight_narrow_range (bool): Whether the quantization algorithm use weight narrow_range or not. If
                `True` then base on narrow_range, otherwise base on not narrow_range.

        Raises:
            TypeError: If `weight_narrow_range` is not bool.
        """
        Validator.check_bool(weight_narrow_range, "weight_narrow_range", self.__class__.__name__)
        self._config.weight_narrow_range = weight_narrow_range

    def set_enable_fusion(self, enable_fusion):
        """
        Set value of enable_fusion of `_config`

        Args:
            enable_fusion (bool): Whether apply fusion before applying quantization, default is False.

        Raises:
            TypeError: If `enable_fusion` is not bool.
            NotImplementedError: Only supported if `act_per_channel` is `False` yet.
        """
        Validator.check_bool(enable_fusion, "enable_fusion", self.__class__.__name__)
        self._config.enable_fusion = enable_fusion

    def _update_qconfig_by_dict(self, config: dict):
        """Update `_config` from a dict"""

        def convert2list(name, value):
            if not isinstance(value, list) and not isinstance(value, tuple):
                value = [value]
            elif len(value) > 2:
                raise ValueError("input `{}` len should less then 2".format(name))
            return value

        quant_delay_list = convert2list("quant delay", config.get("quant_delay", [0, 0]))
        quant_dtype_list = convert2list("quant dtype", config.get("quant_dtype", [QuantDtype.INT8, QuantDtype.INT8]))
        per_channel_list = convert2list("per channel", config.get("per_channel", [False, True]))
        symmetric_list = convert2list("symmetric", config.get("symmetric", [False, True]))
        narrow_range_list = convert2list("narrow range", config.get("narrow_range", [False, False]))

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

    def apply(self, network: Cell) -> Cell:
        """
        Apply default QAT-Algorithm on `network`

        Args:
            network (Cell): Network to be quantized.

        Returns:
            Quantized network.
        """
        self._qat_policy.build()
        return super(SimulatedQuantizationAwareTraining, self).apply(network)
