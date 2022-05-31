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
from ..constant import QuantDtype
from ..simulated_quantization.simulated_quantization_aware_training import SimulatedQuantizationAwareTraining as SimQAT
from .learned_scale_quantization_net_policy import LearnedScaleQuantizationNetPolicy as LsqNetPolicy
from .learned_scale_quantization_config import LearnedScaleQuantizationConfig as LsqConfig
from .learned_scale_fake_quantizers import LearnedScaleFakeQuantizerPerLayer as LsqFqPerLayer, \
    LearnedScaleFakeQuantizePerChannel as LsqFqPerChannel
from ..quantize_wrapper_cell import QuantizeWrapperCell


class LearnedScaleQuantizationAwareTraining(SimQAT):
    """
    Derived class of SimQAT. LSQ quantization algorithm..
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
            - freeze_bn (int): Number of steps after which BatchNorm OP parameters fixed to global mean and variance.
            - bn_fold (bool): Whether to use bn fold ops for simulation inference operation.
            - one_conv_fold (bool): Whether to use one conv bn fold ops for simulation inference operation.

    Supported Platforms:
         ``Ascend`` ``GPU``

    Raises:
        TypeError: If the element of `quant_delay` is not int.
        TypeError: If the element of `per_channel`, `symmetric`, `narrow_range`, `bn_fold`, `one_conv_fold` is not bool.
        TypeError: If the element of `quant_dtype` is not `QuantDtype`.
        TypeError: If `freeze_bn` is not int.
        ValueError: `freeze_bn` is less than 0.
        ValueError: If the length of `quant_delay`, `quant_dtype`, `per_channel`, `symmetric` or `narrow_range` is not
         less than 2.
        ValueError: If the element of `quant_delay` is less than 0.
        ValueError: If the first element of `per_channel` is `True`.
        NotImplementedError: If the element of `quant_dtype` is not `QuantDtype.INT8`.

    Examples:
        >>> from mindspore_gs.quantization.learned_scale_quantization import LearnedScaleQuantizationAwareTraining
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
        >>> learned_quantization = LearnedScaleQuantizationAwareTraining()
        >>> net_qat = learned_quantization.apply(net)
    """

    def set_act_symmetric(self, act_symmetric):
        """
        Raises:
            TypeError: If `act_symmetric` is not bool.
            NotImplementedError:  Learned scale quantization only support `act_symmetric` is `True` currently.
        """
        Validator.check_bool(act_symmetric, "act_symmetric", self.__class__.__name__)
        if not act_symmetric:
            raise NotImplementedError("Learned scale quantization only support `act_symmetric` is `True` currently")
        super(LearnedScaleQuantizationAwareTraining, self).set_act_symmetric(act_symmetric)

    def set_weight_symmetric(self, weight_symmetric):
        """
        Raises:
            TypeError: If `weight_symmetric` is not bool.
            NotImplementedError:  Learned scale quantization only support `weight_symmetric` is `True` currently.
        """
        Validator.check_bool(weight_symmetric, "weight_symmetric", self.__class__.__name__)
        if not weight_symmetric:
            raise NotImplementedError("Learned scale quantization only support `weight_symmetric` is `True` currently")
        super(LearnedScaleQuantizationAwareTraining, self).set_act_symmetric(weight_symmetric)

    def set_act_narrow_range(self, act_narrow_range):
        """
        Raises:
            TypeError: If `act_narrow_range` is not bool.
            NotImplementedError:  Learned scale quantization only support `act_narrow_range` is `True` currently
        """
        Validator.check_bool(act_narrow_range, "act_narrow_range", self.__class__.__name__)
        if not act_narrow_range:
            raise NotImplementedError("Learned scale quantization only support `act_narrow_range` is `True` currently")
        self._config.act_narrow_range = act_narrow_range

    def set_weight_narrow_range(self, weight_narrow_range):
        """
        Raises:
            TypeError: If `weight_narrow_range` is not bool.
            NotImplementedError:  Learned scale quantization only support `weight_narrow_range` is `True` currently
        """
        Validator.check_bool(weight_narrow_range, "weight_narrow_range", self.__class__.__name__)
        if not weight_narrow_range:
            raise NotImplementedError("Learned scale quantization only support `weight_narrow_range` is `True` "
                                      "currently")
        super(LearnedScaleQuantizationAwareTraining, self).set_weight_narrow_range(weight_narrow_range)

    def set_act_quant_delay(self, act_quant_delay):
        """
        Raises:
            TypeError: If `act_quant_delay` is not int.
            NotImplementedError:  Learned scale quantization only support `act_quant_delay` is 0 currently
        """
        Validator.check_is_int(act_quant_delay, "act_quant_delay", self.__class__.__name__)
        if act_quant_delay != 0:
            raise NotImplementedError("Learned scale quantization only support `act_quant_delay` is 0 currently")
        super(LearnedScaleQuantizationAwareTraining, self).set_act_quant_delay(act_quant_delay)

    def set_weight_quant_delay(self, weight_quant_delay):
        """
        Raises:
            TypeError: If `weight_quant_delay` is not int.
            NotImplementedError:  Learned scale quantization only support `weight_quant_delay` is 0 currently
        """
        Validator.check_is_int(weight_quant_delay, "weight_quant_delay", self.__class__.__name__)
        if weight_quant_delay != 0:
            raise NotImplementedError("Learned scale quantization only support `weight_quant_delay` is 0 currently")
        super(LearnedScaleQuantizationAwareTraining, self).set_weight_quant_delay(weight_quant_delay)

    def set_freeze_bn(self, freeze_bn):
        """
        Raises:
            TypeError: If `freeze_bn` is not int.
            NotImplementedError:  Learned scale quantization only support `freeze_bn` is 0 currently
        """
        Validator.check_is_int(freeze_bn, "freeze_bn", self.__class__.__name__)
        if freeze_bn != 0:
            raise NotImplementedError("Learned scale quantization only support `freeze_bn` is 0 currently")
        super(LearnedScaleQuantizationAwareTraining, self).set_freeze_bn(freeze_bn)

    def apply(self, network: Cell) -> Cell:
        quanted_net = super(LearnedScaleQuantizationAwareTraining, self).apply(network)
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

    def _create_qconfig_by_dict(self, config: dict):
        self._config = LsqConfig()
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
