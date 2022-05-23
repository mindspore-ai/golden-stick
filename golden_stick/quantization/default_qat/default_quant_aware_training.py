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
"""DefaultQuantAwareTraining."""

from mindspore.nn import Cell
from mindspore._checkparam import Validator, Rel
from ..quant_aware_training import QuantAwareTraining
from .default_net_policy import DefaultNetworkPolicy
from .constant import QuantDtype
from .quant_config import QuantConfig


class DefaultQuantAwareTraining(QuantAwareTraining):
    """
    Derived class of GoldenStick. Default QAT-algorithm.
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

    Supported Platforms:
         ``Ascend`` ``GPU``

    Raises:
        TypeError: If the element of `quant_delay` is not int.
        TypeError: If the element of `per_channel`, `symmetric`, `narrow_range` is not bool.
        TypeError: If the element of `quant_dtype` is not `QuantDtype`.
        ValueError: If the length of `quant_delay`, `quant_dtype`, `per_channel`, `symmetric` or `narrow_range` is not
         less than 2.
        ValueError: If the element of `quant_delay` is less than 0.
        ValueError: If the first element of `per_channel` is `True`

    Examples:
        >>> from golden_stick.quantization.default_qat import DefaultQuantAwareTraining
        >>> from mindspore import nn
        >>> from models.official.cv.lenet.src.lenet import LeNet5
        >>> net = LeNet5()
        >>> default_qat = DefaultQuantAwareTraining()
        >>> net_qat = default_qat.apply(net)
    """

    def __init__(self, config=None):
        super(DefaultQuantAwareTraining, self).__init__(config)
        if config is None:
            config = {}
        Validator.check_value_type("config", config, [dict], self.__class__.__name__)
        self._config: QuantConfig = QuantConfig()
        self._update_qconfig_by_dict(config)

        self._qat_policy = DefaultNetworkPolicy(self._config)
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
            bn_fold (bool): Whether use bn_fold or not.

        Raises:
            RuntimeError: Not support for bn fold yet.
        """
        raise NotImplementedError(f"Not support for bn fold yet.")

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
            ValueError: act_per_channel is True.
        """
        Validator.check_bool(act_per_channel, "act_per_channel", self.__class__.__name__)
        if act_per_channel:
            raise ValueError(f'act_per_channel only support False now')
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

    def set_act_symmetric(self, act_symmetric):
        """
        Set value of act_symmetric of `_config`

        Args:
            act_symmetric (bool): Whether the quantization algorithm is symmetric or not. If `True` then base on
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
            weight_symmetric (bool): Whether the quantization algorithm is symmetric or not. If `True` then base on
                symmetric, otherwise base on asymmetric.

        Raises:
            TypeError: If `weight_symmetric` is not bool.
        """
        Validator.check_bool(weight_symmetric, "weight_symmetric", self.__class__.__name__)
        self._config.weight_symmetric = weight_symmetric

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
        symmetric_list = convert2list("symmetric", config.get("symmetric", [False, False]))
        narrow_range_list = convert2list("narrow range", config.get("narrow_range", [False, False]))

        self._config.act_quant_delay = Validator.check_non_negative_int(quant_delay_list[0], "quant delay")
        self._config.weight_quant_delay = Validator.check_non_negative_int(quant_delay_list[-1], "quant delay")
        self._config.act_quant_dtype = Validator.check_isinstance("weights dtype", quant_dtype_list[0], QuantDtype)
        self._config.weight_quant_dtype = Validator.check_isinstance("weights dtype", quant_dtype_list[-1], QuantDtype)
        self._config.act_per_channel = Validator.check_bool(per_channel_list[0], "per channel")
        self._config.weight_per_channel = Validator.check_bool(per_channel_list[-1], "per channel")
        self._config.act_symmetric = Validator.check_bool(symmetric_list[0], "symmetric")
        self._config.weight_symmetric = Validator.check_bool(symmetric_list[-1], "symmetric")
        self._config.act_narrow_range = Validator.check_bool(narrow_range_list[0], "narrow range")
        self._config.weight_narrow_range = Validator.check_bool(narrow_range_list[-1], "narrow range")
        self._config.one_conv_fold = Validator.check_bool(config.get("one_conv_fold", True), "one conv fold")
        self._config.enable_fusion = Validator.check_bool(config.get("enable_fusion", False), "enable fusion")

    def apply(self, network: Cell) -> Cell:
        """
        Apply default QAT-Algorithm on `network`

        Args:
            network (Cell): Network to be quantized.

        Returns:
            Quantized network.
        """
        self._qat_policy.build()
        return super(DefaultQuantAwareTraining, self).apply(network)
