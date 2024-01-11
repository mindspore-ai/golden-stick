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
"""LayerQConfig."""
import abc
from typing import Optional
from mindspore.nn import Cell
from .fake_quantizer import FakeQuantizer

layer_policy_key = "layer_quant_policy"


class PerChannelArgs:
    def __init__(self, num_channels=-1, channel_axis=-1, rank=-1):
        self.num_channels = num_channels
        self.channel_axis = channel_axis
        self.rank = rank


class LayerPolicy(abc.ABC):
    """
    Base class for layer quantize configure.
    Configuration including:
        Which weights of layer to be fake-quantize and how they should be fake-quantized
        If input and output of activation of layer need to be fake-quantized and how they should be fake-quantized
        If output and layer need to be fake-quantized and how it should be fake-quantized

    Args:
        config (int): User config for QAT. Config specification is default by derived class.

    Supported Config:
        ``quant_delay`` ``quant_dtype`` ``per_channel`` ``symmetric`` ``narrow_range`` ``one_conv_fold``.

    Note:
        Derived class must override `get_weight_name_and_quantizers`, `get_act_name_and_quantizers`,
            `get_output_quantizers` and `wrapper_cell`.
    """

    def __init__(self):
        self._input_num = 0
        self._inputs_insert_fq = []
        self._output_insert_fq: bool = True

    @abc.abstractmethod
    def get_weight_quantizer(self, weight_name="", perchannel_args: PerChannelArgs = PerChannelArgs(),
                             **kwargs) -> FakeQuantizer:
        """
        Define how to fake-quantize weight data. This method must be overridden by all subclasses.

        Returns:
            Return a list of 2-tuple of weight_name and weight_quantizer.
            Return empty list if no need to fake-quant weight.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_input_quantizer(self, input_index=-1, perchannel_args: PerChannelArgs = PerChannelArgs(),
                             **kwargs) -> FakeQuantizer:
        """
        Define how to fake-quantize input data. This method must be overridden by all subclasses.

        Returns:
            Return an instance of quantizer as quantizer for inputs.
        """
        raise NotImplementedError

    def get_input_quantizer(self, input_index=-1, perchannel_args: PerChannelArgs = PerChannelArgs(), **kwargs) -> \
    Optional[FakeQuantizer]:
        """
        Define how to fake-quantize input data.

        Returns:
            Return an instance of quantizer as quantizer for inputs.
            Return None if all inputs don't need to fake-quant.
        """
        if input_index >= self._input_num:
            return None
        if not self._inputs_insert_fq[input_index]:
            return None
        return self._get_input_quantizer(input_index, perchannel_args, **kwargs)

    @abc.abstractmethod
    def _get_output_quantizer(self, perchannel_args: PerChannelArgs = PerChannelArgs(), **kwargs) -> FakeQuantizer:
        """
        Define how to fake-quantize output data. This method must be overridden by all subclasses.

        Returns:
            Return an instance of quantizer as quantizer for outputs.
        """
        raise NotImplementedError

    def get_output_quantizer(self, perchannel_args: PerChannelArgs = PerChannelArgs(),
                             **kwargs) -> Optional[FakeQuantizer]:
        """
        Define how to fake-quantize output data.

        Returns:
            Return an instance of quantizer as quantizer for outputs.
            Return None if all outputs don't need to fake-quant.
        """
        if self._output_insert_fq:
            return self._get_output_quantizer(perchannel_args, **kwargs)
        return None

    @abc.abstractmethod
    def wrap_cell(self, handler: Cell) -> Cell:
        """
        Define how to wrapper `handler`. This method must be overridden by all subclasses.

        Args:
            handler (Cell): cell to be wrapped.

        Returns:
            Wrapped cell.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_config(self):
        raise NotImplementedError

    def set_input_number(self, input_num: int):
        self._inputs_insert_fq.clear()
        self._input_num = input_num
        for _ in range(0, self._input_num):
            self._inputs_insert_fq.append(True)

    def set_input_not_insert_fq(self, index: Optional[int] = None):
        if index is None:
            for i in range(0, self._input_num):
                self._inputs_insert_fq[i] = False
        else:
            if 0 <= index < self._input_num:
                self._inputs_insert_fq[index] = False
            else:
                raise RuntimeError("Index out of range of input number")

    def get_input_need_insert_fq(self) -> list:
        return self._inputs_insert_fq

    def set_output_not_insert_fq(self):
        self._output_insert_fq = False
