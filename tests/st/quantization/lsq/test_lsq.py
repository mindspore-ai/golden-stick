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
"""test interfaces of lsq."""
from collections import OrderedDict
import pytest
from mindspore import nn
from mindspore.common.dtype import QuantDtype
from mindspore_gs.quantization.learned_step_size_quantization import LearnedStepSizeQuantizationAwareTraining as \
    LearnedQAT
from mindspore_gs.quantization.learned_step_size_quantization.learned_step_size_quantization_layer_policy import \
    LearnedStepSizeFakeQuantizerPerLayer, LearnedStepSizeFakeQuantizePerChannel
from mindspore_gs.quantization.quantize_wrapper_cell import QuantizeWrapperCell


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_construct():
    """
    Feature: LSQ(Learned Step Size Quantization) algorithm constructor.
    Description: Feed config dictionary into constructor functional interface.
    Expectation: Success.
    """

    config = {"quant_delay": 0, "quant_dtype": QuantDtype.INT8, "per_channel": False, "symmetric": True,
              "narrow_range": True, "enable_fusion": True, "freeze_bn": 0, "one_conv_fold": True, "bn_fold": False}
    LearnedQAT(config)
    # add assert for compare

    config = {"quant_delay": (0, 0), "quant_dtype": (QuantDtype.INT8, QuantDtype.INT8), "per_channel": (False, True),
              "symmetric": (True, True), "narrow_range": (True, True), "enable_fusion": True, "freeze_bn": 0,
              "one_conv_fold": True, "bn_fold": False}
    LearnedQAT(config)
    # add assert for compare

    config = {"quant_delay": [0, 0], "quant_dtype": [QuantDtype.INT8, QuantDtype.INT8], "per_channel": [False, True],
              "symmetric": [True, True], "narrow_range": [True, True], "enable_fusion": True, "freeze_bn": 0,
              "one_conv_fold": True, "bn_fold": False}
    LearnedQAT(config)
    # add assert for compare


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_bn_fold():
    """
    Feature: LSQ(Learned Step Size Quantization) set function set_bn_fold().
    Description: Feed data into set_bn_fold() functional interface.
    Expectation:
        If the input data is bool, config success.
        If the input data is not bool, except TypeError.
    """

    lsq = LearnedQAT()
    lsq.set_bn_fold(True)
    assert lsq._config.bn_fold

    with pytest.raises(TypeError) as e:
        lsq.set_bn_fold(1)
    assert "LearnedStepSizeQuantizationAwareTraining', the 'bn_fold' must be a bool" in str(e.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_one_conv_fold():
    """
    Feature: LSQ(Learned Step Size Quantization) set function set_one_conv_fold().
    Description: Feed data `one_conv_fold` into set_one_conv_fold() functional interface.
    Expectation:
        If the input data is bool, config success.
        If the input data is not bool, except TypeError.
    """

    lsq = LearnedQAT()
    lsq.set_one_conv_fold(True)
    assert lsq._config.one_conv_fold

    with pytest.raises(TypeError) as e:
        lsq.set_one_conv_fold(0.5)
    assert "LearnedStepSizeQuantizationAwareTraining', the 'one_conv_fold' must be a bool" in str(e.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_act_quant_delay():
    """
    Feature: LSQ(Learned Step Size Quantization) set function set_act_quant_delay().
    Description: Feed data `act_quant_delay` into set_act_quant_delay() functional interface.
    Expectation:
        If the input is int and 0, config success.
        If the input is int but not 0, except ValueError.
        If the input is not int, except TypeError.
    """

    lsq = LearnedQAT()
    lsq.set_act_quant_delay(0)
    assert lsq._config.act_quant_delay == 0

    with pytest.raises(ValueError) as e:
        lsq.set_act_quant_delay(100)
    assert "Learned step size quantization only support `act_quant_delay` is 0 currently" in str(e.value)

    with pytest.raises(TypeError) as e:
        lsq.set_act_quant_delay(0.1)
    assert "For 'LearnedStepSizeQuantizationAwareTraining', the type of 'act_quant_delay' must be int" in str(e.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_weight_quant_delay():
    """
    Feature: LSQ(Learned Step Size Quantization) set function set_weight_quant_delay().
    Description: Feed data `weight_quant_delay` into set_weight_quant_delay() functional interface.
    Expectation:
        If the input is int and 0, config success.
        If the input is int but not 0, except ValueError.
        If the input is not int, except TypeError.
    """

    lsq = LearnedQAT()
    lsq.set_weight_quant_delay(0)
    assert lsq._config.weight_quant_delay == 0

    with pytest.raises(ValueError) as e:
        lsq.set_weight_quant_delay(100)
    assert "Learned step size quantization only support `weight_quant_delay` is 0 currently" in str(e.value)

    with pytest.raises(TypeError) as e:
        lsq.set_weight_quant_delay(0.1)
    assert "For 'LearnedStepSizeQuantizationAwareTraining', the type of 'weight_quant_delay' must be int" \
           in str(e.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_act_per_channel():
    """
    Feature: LSQ(Learned Step Size Quantization) set function set_act_per_channel().
    Description: Feed data `act_per_channel` into set_act_per_channel() functional interface.
    Expectation:
        If the input is bool and False, config success.
        If the input is bool and True, except ValueError.
        If the input is not bool, except TypeError.
    """

    lsq = LearnedQAT()
    lsq.set_act_per_channel(False)
    assert not lsq._config.act_per_channel

    with pytest.raises(ValueError) as e:
        lsq.set_act_per_channel(True)
    assert "Only supported if `act_per_channel` is False yet." in str(e.value)

    with pytest.raises(TypeError) as e:
        lsq.set_act_per_channel(0.1)
    assert "LearnedStepSizeQuantizationAwareTraining', the 'act_per_channel' must be a bool" in str(e.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_weight_per_channel():
    """
    Feature: LSQ(Learned Step Size Quantization) set function set_weight_per_channel().
    Description: Feed data `weight_per_channel` into set_weight_per_channel() functional interface.
    Expectation:
        If the input is bool, config success.
        If the input is not bool, except TypeError.
    """

    lsq = LearnedQAT()
    lsq.set_weight_per_channel(True)
    assert lsq._config.weight_per_channel

    with pytest.raises(TypeError) as e:
        lsq.set_weight_per_channel(1)
    assert "LearnedStepSizeQuantizationAwareTraining', the 'weight_per_channel' must be a bool" in str(e.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_act_quant_dtype():
    """
    Feature: LSQ(Learned Step Size Quantization) set function set_act_quant_dtype().
    Description: Feed data `act_quant_dtype` into set_act_quant_dtype() functional interface.
    Expectation:
        If the input is QuantDtype and QuantDtype.INT8, config success.
        If the input is QuantDtype but not QuantDtype.INT8, except TypeError.
        If the input is not QuantDtype, except TypeError.
    """

    lsq = LearnedQAT()
    lsq.set_act_quant_dtype(QuantDtype.INT8)
    assert lsq._config.act_quant_dtype == QuantDtype.INT8

    with pytest.raises(ValueError) as e:
        lsq.set_act_quant_dtype(QuantDtype.INT4)
    assert "Only supported if `act_quant_dtype` is `QuantDtype.INT8` yet." in str(e.value)

    with pytest.raises(TypeError) as e:
        lsq.set_act_quant_dtype(1)
    assert "The parameter `act quant dtype` must be isinstance of QuantDtype" in str(e.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_weight_quant_dtype():
    """
    Feature: LSQ(Learned Step Size Quantization) set function set_weight_quant_dtype().
    Description: Feed data `weight_quant_dtype` into set_weight_quant_dtype() functional interface.
    Expectation:
        If the input is QuantDtype and QuantDtype.INT8, config success.
        If the input is QuantDtype but not QuantDtype.INT8, except TypeError.
        If the input is not QuantDtype, except TypeError.
    """

    lsq = LearnedQAT()
    lsq.set_weight_quant_dtype(QuantDtype.INT8)
    assert lsq._config.weight_quant_dtype == QuantDtype.INT8

    with pytest.raises(ValueError) as e:
        lsq.set_weight_quant_dtype(QuantDtype.INT4)
    assert "Only supported if `weight_quant_dtype` is `QuantDtype.INT8` yet." in str(e.value)

    with pytest.raises(TypeError) as e:
        lsq.set_weight_quant_dtype(1)
    assert "The parameter `weight quant dtype` must be isinstance of QuantDtype" in str(e.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_act_symmetric():
    """
    Feature: LSQ(Learned Step Size Quantization) set function set_act_symmetric().
    Description: Feed data `act_symmetric` into set_act_symmetric() functional interface.
    Expectation:
        If the input is bool and True, config success.
        If the input is bool and False, except ValueError.
        If the input is not bool, except TypeError.
    """

    lsq = LearnedQAT()
    lsq.set_act_symmetric(True)
    assert lsq._config.act_symmetric

    with pytest.raises(ValueError) as e:
        lsq.set_act_symmetric(False)
    assert "Learned step size quantization only support `act_symmetric` is True currently" in str(e.value)

    with pytest.raises(TypeError) as e:
        lsq.set_act_symmetric(1)
    assert "For 'LearnedStepSizeQuantizationAwareTraining', the 'act_symmetric' must be a bool" in str(e.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_weight_symmetric():
    """
    Feature: LSQ(Learned Step Size Quantization) set function set_weight_symmetric().
    Description: Feed data `weight_symmetric` into set_weight_symmetric() functional interface.
    Expectation:
        If the input is bool and True, config success.
        If the input is bool and False, except ValueError.
        If the input is not bool, except TypeError.
    """

    lsq = LearnedQAT()
    lsq.set_weight_symmetric(True)
    assert lsq._config.weight_symmetric

    with pytest.raises(ValueError) as e:
        lsq.set_weight_symmetric(False)
    assert "Learned step size quantization only support `weight_symmetric` is True currently" in str(e.value)

    with pytest.raises(TypeError) as e:
        lsq.set_weight_symmetric(1)
    assert "For 'LearnedStepSizeQuantizationAwareTraining', the 'weight_symmetric' must be a bool" in str(e.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_act_narrow_range():
    """
    Feature: LSQ(Learned Step Size Quantization) set function set_act_narrow_range().
    Description: Feed data `act_narrow_range` into set_act_narrow_range() functional interface.
    Expectation:
        If the input is bool and True, config success.
        If the input is bool and False, except ValueError.
        If the input is not bool, except TypeError.
    """

    lsq = LearnedQAT()
    lsq.set_act_narrow_range(True)
    assert lsq._config.act_narrow_range

    with pytest.raises(ValueError) as e:
        lsq.set_act_narrow_range(False)
    assert "Learned step size quantization only support `act_narrow_range` is True currently" in str(e.value)

    with pytest.raises(TypeError) as e:
        lsq.set_act_narrow_range(1)
    assert "For 'LearnedStepSizeQuantizationAwareTraining', the 'act_narrow_range' must be a bool" in str(e.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_weight_narrow_range():
    """
    Feature: LSQ(Learned Step Size Quantization) set function set_weight_narrow_range().
    Description: Feed data `weight_narrow_range` into set_weight_narrow_range() functional interface.
    Expectation:
        If the input is bool and True, config success.
        If the input is bool and False, except ValueError.
        If the input is not bool, except TypeError.
    """

    lsq = LearnedQAT()
    lsq.set_weight_narrow_range(True)
    assert lsq._config.weight_narrow_range

    with pytest.raises(ValueError) as e:
        lsq.set_weight_narrow_range(False)
    assert "Learned step size quantization only support `weight_narrow_range` is True currently" in str(e.value)

    with pytest.raises(TypeError) as e:
        lsq.set_weight_narrow_range(1)
    assert "For 'LearnedStepSizeQuantizationAwareTraining', the 'weight_narrow_range' must be a bool" in str(e.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_freeze_bn():
    """
    Feature: LSQ(Learned Step Size Quantization) set function set_freeze_bn().
    Description: Feed data `freeze_bn` into set_freeze_bn() functional interface.
    Expectation:
        If the input is int and 0, config success.
        If the input is int but not 0, except ValueError.
        If the input is not int, except TypeError.
    """

    lsq = LearnedQAT()
    lsq.set_freeze_bn(0)
    assert lsq._config.freeze_bn == 0

    with pytest.raises(ValueError) as e:
        lsq.set_freeze_bn(100)
    assert "Learned step size quantization only support `freeze_bn` is 0 currently" in str(e.value)

    with pytest.raises(TypeError) as e:
        lsq.set_freeze_bn(0.5)
    assert "For 'LearnedStepSizeQuantizationAwareTraining', the type of 'freeze_bn' must be int" in str(e.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_enable_fusion():
    """
    Feature: LSQ(Learned Step Size Quantization) set function set_enable_fusion().
    Description: Feed data `enable_fusion` into set_enable_fusion() functional interface.
    Expectation:
        If the input data is bool, config success.
        If the input data is not bool, except TypeError.
    """

    lsq = LearnedQAT()
    lsq.set_enable_fusion(True)
    assert lsq._config.enable_fusion

    with pytest.raises(TypeError) as e:
        lsq.set_enable_fusion(0.5)
    assert "For 'LearnedStepSizeQuantizationAwareTraining', the 'enable_fusion' must be a bool" in str(e.value)


class NetToQuant(nn.Cell):
    """
    Network with single conv2d to be quanted
    """

    def __init__(self):
        super(NetToQuant, self).__init__()
        self.conv = nn.Conv2d(5, 7, 3, pad_mode='valid')

    def construct(self, x):
        x = self.conv(x)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_apply():
    """
    Feature: LearnedQAT algorithm set functions.
    Description: Apply DefaultQuantAwareTraining on single conv2d net.
    Expectation: Apply success and coordinate attributes are same as config.
    """

    network = NetToQuant()
    config = {"quant_delay": (0, 0), "quant_dtype": (QuantDtype.INT8, QuantDtype.INT8), "per_channel": (False, True),
              "symmetric": (True, True), "narrow_range": (True, True), "enable_fusion": True, "freeze_bn": 0,
              "one_conv_fold": True, "bn_fold": False}
    lsq = LearnedQAT(config)
    new_network = lsq.apply(network)
    cells: OrderedDict = new_network.name_cells()

    assert cells.get("Conv2dQuant", None) is not None
    conv_quant: QuantizeWrapperCell = cells.get("Conv2dQuant")
    assert isinstance(conv_quant, QuantizeWrapperCell)
    conv_handler = conv_quant._handler
    weight_fake_quant: LearnedStepSizeFakeQuantizePerChannel = conv_handler.fake_quant_weight
    assert isinstance(weight_fake_quant, LearnedStepSizeFakeQuantizePerChannel)
    assert weight_fake_quant._symmetric
    assert weight_fake_quant._quant_delay == 0
    act_fake_quant = conv_quant._output_quantizer
    assert isinstance(act_fake_quant, LearnedStepSizeFakeQuantizerPerLayer)
    assert act_fake_quant._symmetric
    assert act_fake_quant._quant_delay == 0
