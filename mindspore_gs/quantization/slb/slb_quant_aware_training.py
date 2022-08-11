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

from mindspore import Model
from mindspore.nn import Cell
from mindspore.train.callback import Callback
from mindspore._checkparam import Validator, Rel
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
            - epoch_size (int): Total training epochs. Default: 100.
            - has_trained_epoch (int): The trained epochs. Default: 0.
            - t_start_val (float): Initial value of temperature hyperparameters. Default: 1.
            - t_start_time (float): Fraction of epochs after which temperature hyperparameters starting changing.
              Default: 0.2.
            - t_end_time (float): Fraction of epochs after which temperature hyperparameters stopping changing.
              Default: 0.6.
            - t_factor (float): Multiplicative factor of temperature hyperparameters changing.
              Default: 1.2.

    Raises:
        TypeError: If `quant_dtype` is not `QuantDtype`.
        TypeError: If `epoch_size` or `has_trained_epoch` is not an int.
        TypeError: If `t_start_val`, `t_start_time`, `t_end_time` or `t_factor` is not float.
        ValueError: If `epoch_size` or `has_trained_epoch` is less than 0.
        ValueError: If `t_start_val`, `t_start_time`, `t_end_time` or `t_factor` is less than 0.
        ValueError: If `t_start_time` or `t_end_time` is greater than 1.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> from mindspore_gs.quantization.slb import SlbQuantAwareTraining
        >>> from mindspore_gs.quantization.constant import QuantDtype
        >>> from mindspore import nn
        >>> class NetToQuant(nn.Cell):
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
        >>> ## 2) Define SLB QAT-Algorithm
        >>> slb_quantization = SlbQuantAwareTraining()
        >>> ## 3) Use set functions to change config
        >>> ## 3.1) set_weight_quant_dtype is used to set the weight quantization bit, and support QuantDtype.INT4, QuantDtype.INT2,
        >>> ## QuantDtype.INT1 now.
        >>> slb_quantization.set_weight_quant_dtype(QuantDtype.INT1)
        >>> ## 3.2) set_epoch_size is used to set the epoch size of training.
        >>> slb_quantization.set_epoch_size(100)
        >>> ## 3.3) set_has_trained_epoch is used to set the trained epoch size of training.
        >>> slb_quantization.set_has_trained_epoch(0)
        >>> ## 3.4) set_t_start_val is used to set the initial value of temperature hyperparameters.
        >>> slb_quantization.set_t_start_val(1.0)
        >>> ## 3.5) set_t_start_time is used to set the fraction of epochs after which temperature hyperparameters starting changing.
        >>> slb_quantization.set_t_start_time(0.2)
        >>> ## 3.6) set_t_end_time is used to set the fraction of epochs after which temperature hyperparameters stopping changing.
        >>> slb_quantization.set_t_end_time(0.6)
        >>> ## 3.7) set_t_factor is used to set the multiplicative factor of temperature hyperparameters changing.
        >>> slb_quantization.set_t_factor(1.2)
        >>> ## 4) Print SLB QAT-Algorithm object and check the config setting result
        >>> ## Since we set weight_quant_dtype to be QuantDtype.INT1, the value of the attribute weight_quant_dtype is INT1
        >>> ## Since we set epoch_size to be 100, the value of the attribute epoch_size is 100
        >>> ## Since we set has_trained_epoch to be 0, the value of the attribute has_trained_epoch is 0
        >>> ## Since we set t_start_val to be 1.0, the value of the attribute t_start_val is 1.0
        >>> ## Since we set t_start_time to be 0.2, the value of the attribute t_start_time is 0.2
        >>> ## Since we set t_end_time to be 0.6, the value of the attribute t_end_time is 0.6
        >>> ## Since we set t_factor to be 1.2, the value of the attribute t_factor is 1.2
        >>> print(slb_quantization)
        SlbQuantAwareTraining<weight_quant_dtype=INT1, epoch_size=100, has_trained_epoch=0, t_start_val=1.0, t_start_time=0.2, t_end_time=0.6, t_factor=1.2>
        >>> ## 5) Apply SLB QAT-algorithm to origin network
        >>> net_qat = slb_quantization.apply(net)
        >>> ## 6) Print network and check the result. Conv2d should be transformed to QuantizeWrapperCells.
        >>> ## Since we set weight_quant_dtype to be QuantDtype.INT1, the bit_num value of fake_quant_weight
        >>> ## should be 1, and the weight_bit_num value of Conv2dSlbQuant should be 1.
        >>> print(net_qat)
        NetToQuantOpt<
          (_handler): NetToQuant<
            (conv): Conv2d<input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
            (bn): BatchNorm2d<num_features=6, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=bn.gamma, shape=(6,), dtype=Float32, requires_grad=True), beta=Parameter (name=bn.beta, shape=(6,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=bn.moving_mean, shape=(6,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=bn.moving_variance, shape=(6,), dtype=Float32, requires_grad=False)>
            >
          (bn): BatchNorm2d<num_features=6, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=bn.gamma, shape=(6,), dtype=Float32, requires_grad=True), beta=Parameter (name=bn.beta, shape=(6,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=bn.moving_mean, shape=(6,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=bn.moving_variance, shape=(6,), dtype=Float32, requires_grad=False)>
          (Conv2dSlbQuant): QuantizeWrapperCell<
            (_handler): Conv2dSlbQuant<
              in_channels=1, out_channels=6, kernel_size=(5, 5), weight_bit_num=1, stride=(1, 1), pad_mode=valid, padding=0, dilation=(1, 1), group=1, has_bias=False
              (fake_quant_weight): SlbFakeQuantizerPerLayer<bit_num=1>
              >
            >
          >
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
        weight_quant_dtype = Validator.check_isinstance("weight quant dtype", weight_quant_dtype, QuantDtype)
        if weight_quant_dtype not in [QuantDtype.INT1, QuantDtype.INT2, QuantDtype.INT4]:
            raise NotImplementedError("Only supported if `weight_quant_dtype` is `QuantDtype.INT1`, " \
                                      "`QuantDtype.INT2` or `QuantDtype.INT4` yet. " \
                                      "But got {}".format(weight_quant_dtype))
        self._config.weight_quant_dtype = weight_quant_dtype

    def set_epoch_size(self, epoch_size):
        """
        Set value of epoch_size of `_config`

        Args:
            epoch_size (int): the epoch size of training, default: 100.

        Raises:
            TypeError: If `epoch_size` is not int.
            ValueError: If `epoch_size` is not greater than 0.
        """
        epoch_size = Validator.check_int(epoch_size, 0, Rel.GT, "epoch_size", self.__class__.__name__)
        self._config.epoch_size = epoch_size

    def set_has_trained_epoch(self, has_trained_epoch):
        """
        Set value of has_trained_epoch of `_config`

        Args:
            has_trained_epoch (int): the trained epochs of training, default: 0.

        Raises:
            TypeError: If `has_trained_epoch` is not int.
            ValueError: If `has_trained_epoch` is less than 0.
        """
        has_trained_epoch = Validator.check_int(has_trained_epoch, 0, Rel.GE, "has_trained_epoch", self.__class__.__name__)
        self._config.has_trained_epoch = has_trained_epoch

    def set_t_start_val(self, t_start_val):
        """
        Set value of t_start_val of `_config`

        Args:
            t_start_val (float): Initial value of temperature hyperparameters, default: 1.0.

        Raises:
            TypeError: If `t_start_val` is not float.
            ValueError: `t_start_val` is not greater than 0.
        """
        t_start_val = Validator.check_positive_float(t_start_val, "t_start_val", self.__class__.__name__)
        self._config.t_start_val = t_start_val

    def set_t_start_time(self, t_start_time):
        """
        Set value of t_start_time of `_config`

        Args:
            t_start_time (float): Fraction of epochs after which temperature hyperparameters starting changing, default: 0.2.

        Raises:
            TypeError: If `t_start_time` is not float.
            ValueError: If `t_start_time` is less than 0. or greater than 1.
        """
        t_start_time = Validator.check_float_range(t_start_time, 0.0, 1.0, Rel.INC_BOTH, \
                                                   "t_start_time", self.__class__.__name__)
        self._config.t_start_time = t_start_time

    def set_t_end_time(self, t_end_time):
        """
        Set value of t_end_time of `_config`

        Args:
            t_end_time (float): Fraction of epochs after which temperature hyperparameters stopping changing, default: 0.6.

        Raises:
            TypeError: If `t_end_time` is not float.
            ValueError: If `t_end_time` is less than 0. or greater than 1.
        """
        t_end_time = Validator.check_float_range(t_end_time, 0.0, 1.0, Rel.INC_BOTH, \
                                                 "t_end_time", self.__class__.__name__)
        self._config.t_end_time = t_end_time

    def set_t_factor(self, t_factor):
        """
        Set value of t_factor of `_config`

        Args:
            t_factor (float): Multiplicative factor of temperature hyperparameters changing, default: 1.2.

        Raises:
            TypeError: If `t_factor` is not float.
            ValueError: If `t_factor` is not greater than 0.
        """
        t_factor = Validator.check_positive_float(t_factor, "t_factor", self.__class__.__name__)
        self._config.t_factor = t_factor

    def _init_net_policy(self, config):
        return SlbNetPolicy(config)

    def _create_qconfig_by_dict(self, config: dict):
        """Create `_config` from a dict"""
        self._config = SlbQuantConfig()
        self.set_weight_quant_dtype(config.get("quant_dtype", QuantDtype.INT1))

        if "epoch_size" in config:
            self.set_epoch_size(config["epoch_size"])
        if "has_trained_epoch" in config:
            self.set_has_trained_epoch(config["has_trained_epoch"])
        self.set_t_start_val(config.get("t_start_val", 1.0))
        self.set_t_start_time(config.get("t_start_time", 0.2))
        self.set_t_end_time(config.get("t_end_time", 0.6))
        self.set_t_factor(config.get("t_factor", 1.2))

    def callbacks(self, model: Model) -> [Callback]:
        """
        Define TemperatureScheduler callback for SLB QAT-algorithm.

        Args:
            model (Model): Model to be used.

        Raises:
            RuntimeError: If `epoch_size` is not initialized!
            RuntimeError: If `has_trained_epoch` is not initialized!
            ValueError: If `epoch_size` is not greater than `has_trained_epoch`.
            ValueError: If `t_end_time` is less than `t_start_time`.
            TypeError: If `model` is not Model.

        Returns:
            List of instance of Callbacks.
        """

        if self._config.epoch_size == -1:
            raise RuntimeError("The `epoch_size` need to be initialized!")
        if self._config.has_trained_epoch == -1:
            raise RuntimeError("The `has_trained_epoch` need to be initialized!")

        if self._config.epoch_size <= self._config.has_trained_epoch:
            raise ValueError("The `epoch_size` should be greater than `has_trained_epoch`.")
        if self._config.t_end_time < self._config.t_start_time:
            raise ValueError("The `t_end_time` should not be less than `t_start_time`.")

        model = Validator.check_isinstance("model", model, Model)

        cb = []
        cb.append(TemperatureScheduler(model, self._config.epoch_size, self._config.has_trained_epoch,
                                       self._config.t_start_val, self._config.t_start_time,
                                       self._config.t_end_time, self._config.t_factor))
        return cb

    def apply(self, network: Cell) -> Cell:
        """
        Apply SLB quantization Algorithm on `network`, use the following steps to make `network` available for
        quantization aware training:

        1. Fuse certain cells in `network` using pattern engine which is defined by net policy.
        2. Propagate layer policies defined through cells.
        3. Reduce redundant fake quantizers when they are redundant.
        4. Apply layer policies to convert normal cell to `QuantizeWrapperCell`.

        Args:
            network (Cell): Network to be quantized.

        Returns:
            Quantized network.
        """

        self._qat_policy.build()
        return super(SlbQuantAwareTraining, self).apply(network)

    def __repr__(self):
        """Display instance object as string."""
        s = 'SlbQuantAwareTraining<weight_quant_dtype={}, epoch_size={}, has_trained_epoch={}, t_start_val={}, ' \
            't_start_time={}, t_end_time={}, t_factor={}>'.format(self._config.weight_quant_dtype, self._config.epoch_size,
                                                                  self._config.has_trained_epoch, self._config.t_start_val,
                                                                  self._config.t_start_time, self._config.t_end_time,
                                                                  self._config.t_factor)
        return s


class TemperatureScheduler(Callback):
    """
    Define TemperatureScheduler callback for SLB QAT-algorithm.
    """
    def __init__(self, model, epoch_size=100, has_trained_epoch=0,
                 t_start_val=1.0, t_start_time=0.2, t_end_time=0.6, t_factor=1.2):
        super().__init__()
        self.epochs = epoch_size
        self.has_trained_epoch = has_trained_epoch
        self.t_start_val = t_start_val
        self.t_start_time = t_start_time
        self.t_end_time = t_end_time
        self.t_factor = t_factor
        self.model = model

    def epoch_begin(self, run_context):
        """
        Epoch_begin.
        """
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num + self.has_trained_epoch
        # Compute temperature value
        t = self.t_start_val
        t_start_epoch = int(self.epochs*self.t_start_time)
        t_end_epoch = int(self.epochs*self.t_end_time)
        if epoch > t_start_epoch:
            t *= self.t_factor**(min(epoch, t_end_epoch) - t_start_epoch)
        # Assign new value to temperature parameter
        for _, cell in self.model.train_network.cells_and_names():
            if cell.cls_name == 'SlbFakeQuantizerPerLayer': # for SLB
                cell.set_temperature(t)
                if epoch >= t_end_epoch:
                    cell.set_temperature_end_flag()
