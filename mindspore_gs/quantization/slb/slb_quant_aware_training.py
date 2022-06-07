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
"""Basic implementation of slb quantization method, this algorithm regards the discrete weights
in an arbitrary quantized neural network as searchable variables, and utilize a differential method
to search them accurately. In particular, each weight is represented as a probability distribution
over the discrete value set. The probabilities are optimized during training and the values
with the highest probability are selected to establish the desired quantized network.
See more details in `Searching for Low-Bit Weights in Quantized Neural Networks
<https://arxiv.org/pdf/2009.08695.pdf>`. """

from mindspore.nn import Cell
from mindspore._checkparam import Validator
from ..quantization_aware_training import QuantizationAwareTraining
from ..constant import QuantDtype
from .slb_net_policy import SlbNetPolicy
from .slb_quant_config import SlbQuantConfig


class SlbQuantAwareTraining(QuantizationAwareTraining):
    """
    Derived class of GoldenStick. SLB QAT-algorithm.
    Args:
        config (dict): store attributes for quantization aware training, keys are attribute names,
            values are attribute values. Supported attribute are listed below:

            - quant_dtype (Union[QuantDtype, list, tuple]): Datatype used to quantize weights and activations. The first
              element represents data flow and the second element represents weights. It is necessary to consider the
              precision support of hardware devices in the practical quantization infer scenaries.
              Weights quantization support int4|int2|int1, and activations quantization support int8|int4 now.
              Default: (QuantDtype.INT8, QuantDtype.INT1).

    Supported Platforms:
        ``GPU``

    Raises:
        TypeError: If the element of `quant_dtype` is not `QuantDtype`.
        ValueError: If the length of `quant_dtype` is not less than 2.

    Examples:
        >>> from mindspore_gs.quantization.slb import SlbQuantAwareTraining
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
        >>> slb_qat = SlbQuantAwareTraining()
        >>> net_qat = slb_qat.apply(net)
    """

    def __init__(self, config=None):
        super(SlbQuantAwareTraining, self).__init__(config)
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

    def set_act_quant_dtype(self, act_quant_dtype):
        """
        Set value of act_quant_dtype of `_config`

        Args:
            act_quant_dtype (QuantDtype): Datatype used to quantize activations. Default: QuantDtype.INT8.

        Raises:
            TypeError: If `act_quant_dtype` is not QuantDtype.
            NotImplementedError: Only supported if `act_quant_dtype` is `QuantDtype.INT4` or `QuantDtype.INT8` yet.
        """
        Validator.check_isinstance("activation quant dtype", act_quant_dtype, QuantDtype)
        if act_quant_dtype not in [QuantDtype.INT4, QuantDtype.INT8]:
            raise NotImplementedError("Only supported if `act_quant_dtype` is `QuantDtype.INT4` " \
                                      "or `QuantDtype.INT8` yet.")
        self._config.act_quant_dtype = act_quant_dtype

    def set_weight_quant_dtype(self, weight_quant_dtype):
        """
        Set value of weight_quant_dtype of `_config`

        Args:
            weight_quant_dtype (QuantDtype): Datatype used to quantize activations. Default: QuantDtype.INT1.

        Raises:
            TypeError: If `weight_quant_dtype` is not QuantDtype.
            NotImplementedError: Only supported if `weight_quant_dtype` is `QuantDtype.INT1`, `QuantDtype.INT2`
            or `QuantDtype.INT4` yet.
        """
        Validator.check_isinstance("weight quant dtype", weight_quant_dtype, QuantDtype)
        if weight_quant_dtype not in [QuantDtype.INT1, QuantDtype.INT2, QuantDtype.INT4]:
            raise NotImplementedError("Only supported if `weight_quant_dtype` is `QuantDtype.INT1`, " \
                                      "`QuantDtype.INT2` or `QuantDtype.INT4` yet.")
        self._config.weight_quant_dtype = weight_quant_dtype

    @staticmethod
    def _convert2list(name, value):
        if not isinstance(value, list) and not isinstance(value, tuple):
            value = [value]
        elif len(value) > 2:
            raise ValueError("input `{}` len should less then 2".format(name))
        return value

    def _init_net_policy(self, config):
        return SlbNetPolicy(config)

    def _create_qconfig_by_dict(self, config: dict):
        """Create `_config` from a dict"""
        self._config = SlbQuantConfig()
        quant_dtype_list = SlbQuantAwareTraining.\
            _convert2list("quant dtype", config.get("quant_dtype", [QuantDtype.INT8, QuantDtype.INT1]))

        self.set_act_quant_dtype(quant_dtype_list[0])
        self.set_weight_quant_dtype(quant_dtype_list[-1])


    def apply(self, network: Cell) -> Cell:
        """
        Apply SLB QAT-Algorithm on `network`

        Args:
            network (Cell): Network to be quantized.

        Returns:
            Quantized network.
        """
        self._qat_policy.build()
        return super(SlbQuantAwareTraining, self).apply(network)
