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
    Derived class of GoldenStick. SLB(Searching for Low-Bit Weights) QAT-algorithm.

    Args:
        config (dict): store attributes for quantization aware training, keys are attribute names,
            values are attribute values. Supported attribute are listed below:

            - quant_dtype (QuantDtype): Datatype used to quantize weights, weights quantization
              support int4|int2|int1 now.
              Default: QuantDtype.INT1.

    Raises:
        TypeError: If `quant_dtype` is not `QuantDtype`.

    Supported Platforms:
        ``GPU``

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

    def set_weight_quant_dtype(self, weight_quant_dtype):
        """
        Set value of weight_quant_dtype of `_config`

        Args:
            weight_quant_dtype (QuantDtype): Datatype used to quantize weights. Default: QuantDtype.INT1.

        Raises:
            TypeError: If `weight_quant_dtype` is not QuantDtype.
            NotImplementedError: Only supported if `weight_quant_dtype` is `QuantDtype.INT1`, `QuantDtype.INT2`
                or `QuantDtype.INT4` yet.
        """
        Validator.check_isinstance("weight quant dtype", weight_quant_dtype, QuantDtype)
        if weight_quant_dtype not in [QuantDtype.INT1, QuantDtype.INT2, QuantDtype.INT4]:
            raise NotImplementedError("Only supported if `weight_quant_dtype` is `QuantDtype.INT1`, " \
                                      "`QuantDtype.INT2` or `QuantDtype.INT4` yet. " \
                                      "but got {}".format(weight_quant_dtype))
        self._config.weight_quant_dtype = weight_quant_dtype

    def _init_net_policy(self, config):
        return SlbNetPolicy(config)

    def _create_qconfig_by_dict(self, config: dict):
        """Create `_config` from a dict"""
        self._config = SlbQuantConfig()
        self.set_weight_quant_dtype(config.get("quant_dtype", QuantDtype.INT1))

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
