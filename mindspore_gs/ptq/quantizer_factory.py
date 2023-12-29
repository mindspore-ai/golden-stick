from mindspore import QuantDtype

from mindspore_gs.ptq.fake_quantizer import MinMaxPerChannel, MinMaxPerLayer
from mindspore_gs.quantization.fake_quantizer import FakeQuantizer
from mindspore_gs.quantization.layer_policy import PerChannelArgs


class QuantizerFactory:
    """
    Derived class of LayerPolicy. Sim-QAT layer policy.
    Use linear perchannel fake quantizer as weight fake quantizer, linear perlayer fake quantizer as act fake quantizer.

    Supported Config:
        ``quant_delay`` ``quant_dtype`` ``per_channel`` ``symmetric`` ``narrow_range`` ``one_conv_fold``.
    """

    def __init__(self, weight_names=None, act_names=None, cfg=None):
        self._config = cfg
        if cfg.weight_quant_dtype == QuantDtype.INT8:
            self._num_bits = 8
        else:
            raise TypeError("Only support int8 weight quant now!")
        if cfg.act_per_channel:
            raise NotImplementedError("act quant only support perlayer now!")
        self._weight_names = weight_names
        self._act_names = act_names

    def get_weight_quantizer(self, weight_name="", perchannel_args: PerChannelArgs = PerChannelArgs(),
                             **kwargs) -> FakeQuantizer:
        strategy = kwargs.get('strategy', None)
        if self._config.weight_per_channel:
            channel_axis = perchannel_args.channel_axis
            num_channels = perchannel_args.num_channels
            rank = perchannel_args.rank
            if channel_axis == -1:
                raise RuntimeError("Please provide channel axis of weight for per-channel weight quantize.")
            if num_channels == -1:
                raise RuntimeError("Please provide channel number of weight for per-channel weight quantize.")
            weight_quantizer = MinMaxPerChannel(data_rank=rank,
                                                symmetric=self._config.weight_symmetric,
                                                quant_dtype=self._config.weight_quant_dtype,
                                                narrow_range=self._config.weight_narrow_range,
                                                axis=channel_axis, output_channel=num_channels, strategy=strategy)
        else:
            weight_quantizer = MinMaxPerLayer(symmetric=self._config.weight_symmetric,
                                              quant_dtype=self._config.weight_quant_dtype,
                                              narrow_range=self._config.weight_narrow_range, strategy=strategy)
        return weight_quantizer

    def get_input_quantizer(self, input_index=-1, perchannel_args: PerChannelArgs = PerChannelArgs(),
                             **kwargs) -> FakeQuantizer:
        return MinMaxPerLayer(symmetric=self._config.act_symmetric, quant_dtype=self._config.act_quant_dtype,
                              narrow_range=self._config.act_narrow_range, strategy=kwargs.get('strategy', None))

    def get_output_quantizer(self, perchannel_args: PerChannelArgs = PerChannelArgs(), **kwargs) -> FakeQuantizer:
        return MinMaxPerLayer(symmetric=self._config.act_symmetric, quant_dtype=self._config.act_quant_dtype,
                              narrow_range=self._config.act_narrow_range, strategy=kwargs.get('strategy', None))

    def create_observer_perchannel(self,
                                   perchannel_args: PerChannelArgs = PerChannelArgs(),
                                   **kwargs) -> FakeQuantizer:
        strategy = kwargs.get('strategy', None)
        channel_axis = perchannel_args.channel_axis
        num_channels = perchannel_args.num_channels
        rank = perchannel_args.rank
        if num_channels == -1:
            raise RuntimeError("Please provide channel number for observer.")
        perchannel_observer = MinMaxPerChannel(axis=channel_axis,
                                               output_channel=num_channels,
                                               data_rank=rank,
                                               strategy=strategy)
        return perchannel_observer
