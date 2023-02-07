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
"""test slb qat."""

import os
import sys
from collections import OrderedDict
import pytest
from mindspore import nn, context
from mindspore import Model
from mindspore.common.dtype import QuantDtype
from mindspore.nn.metrics import Accuracy

from mindspore_gs.quantization.quantize_wrapper_cell import QuantizeWrapperCell
from mindspore_gs.quantization.slb import SlbQuantAwareTraining as SlbQAT
from mindspore_gs.quantization.slb.slb_fake_quantizer import SlbActQuantizer
from mindspore_gs.quantization.slb.slb_fake_quantizer import SlbFakeQuantizerPerLayer


class NetToQuant(nn.Cell):
    """
    Network with single conv2d to be quanted.
    """

    def __init__(self, num_channel=1):
        super(NetToQuant, self).__init__()
        self.conv = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.bn = nn.BatchNorm2d(6)

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("quant_bit", ["W4", "W2", "W1", "W4A8", "W2A8", "W1A8"])
@pytest.mark.parametrize("enable_bn_calibration", [True, False])
def test_set_config(quant_bit, enable_bn_calibration):
    """
    Feature: SLB(Searching for Low-Bit Weights) QAT-algorithm set functions.
    Description: Apply SlbQuantAwareTraining on lenet.
    Expectation: Apply success and coordinate attributes are same as config.
    """

    network = NetToQuant()
    qat = SlbQAT()
    if "W4" in quant_bit:
        qat.set_weight_quant_dtype(QuantDtype.INT4)
    elif "W2" in quant_bit:
        qat.set_weight_quant_dtype(QuantDtype.INT2)
    elif "W1" in quant_bit:
        qat.set_weight_quant_dtype(QuantDtype.INT1)
    if "A8" in quant_bit:
        qat.set_act_quant_dtype(QuantDtype.INT8)
        qat.set_enable_act_quant(True)
    else:
        qat.set_enable_act_quant(False)
    qat.set_enable_bn_calibration(enable_bn_calibration)
    qat.set_epoch_size(100)
    qat.set_has_trained_epoch(0)
    qat.set_t_start_val(1.0)
    qat.set_t_start_time(0.2)
    qat.set_t_end_time(0.6)
    qat.set_t_factor(1.2)
    new_network = qat.apply(network)
    cells: OrderedDict = new_network.name_cells()

    assert cells.get("Conv2dSlbQuant", None) is not None
    conv_quant: QuantizeWrapperCell = cells.get("Conv2dSlbQuant")
    assert isinstance(conv_quant, QuantizeWrapperCell)
    conv_handler = conv_quant._handler
    weight_fake_quant: SlbFakeQuantizerPerLayer = conv_handler.fake_quant_weight
    assert isinstance(weight_fake_quant, SlbFakeQuantizerPerLayer)
    assert qat._config.enable_bn_calibration == enable_bn_calibration
    assert qat._config.epoch_size == 100
    assert qat._config.has_trained_epoch == 0
    assert qat._config.t_start_val == 1.0
    assert qat._config.t_start_time == 0.2
    assert qat._config.t_end_time == 0.6
    assert qat._config.t_factor == 1.2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("enable_act_quant", [True, False])
def test_convert(enable_act_quant):
    """
    Feature: SLB convert function.
    Description: convert a compressed network to a standard network before exporting to MindIR.
    Expectation: convert success and structure of network as expect.
    """

    network = NetToQuant()
    config = {"quant_dtype": [QuantDtype.INT8, QuantDtype.INT1], "enable_act_quant": enable_act_quant,
              "enable_bn_calibration": False, "epoch_size": 10,
              "has_trained_epoch": 0, "t_start_val": 1.0,
              "t_start_time": 0.2, "t_end_time": 0.6, "t_factor": 3.2}
    qat = SlbQAT(config)
    new_network = qat.apply(network)
    new_network = qat.convert(new_network)

    cells: OrderedDict = new_network.name_cells()
    assert cells.get("Conv2dSlbQuant", None) is not None
    conv: QuantizeWrapperCell = cells.get("Conv2dSlbQuant")
    assert isinstance(conv, QuantizeWrapperCell)
    act_fake_quant = conv._output_quantizer
    assert not isinstance(act_fake_quant, SlbActQuantizer)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_weight_quant_dtype_type():
    """
    Feature: set_weight_quant_dtype api of SLB.
    Description: Feed int type `weight_quant_dtype` into set_weight_quant_dtype() functional interface.
    Expectation: Except TypeError.
    """

    qat = SlbQAT()
    with pytest.raises(TypeError, match="The parameter `weight quant dtype` must be isinstance of QuantDtype, but got "
                                        "3."):
        qat.set_weight_quant_dtype(weight_quant_dtype=3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_weight_quant_dtype_range():
    """
    Feature: set_weight_quant_dtype api of SLB.
    Description: Feed QuantDtype type `weight_quant_dtype` into set_weight_quant_dtype() functional interface.
    Expectation: Except ValueError.
    """

    qat = SlbQAT()
    with pytest.raises(ValueError, match="Only supported if `weight_quant_dtype` is `QuantDtype.INT1`, "
                                         "`QuantDtype.INT2` or `QuantDtype.INT4` yet."):
        qat.set_weight_quant_dtype(weight_quant_dtype=QuantDtype.INT8)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_act_quant_dtype_type():
    """
    Feature: set_act_quant_dtype api of SLB.
    Description: Feed int type `act_quant_dtype` into set_act_quant_dtype() functional interface.
    Expectation: Except TypeError.
    """

    qat = SlbQAT()
    with pytest.raises(TypeError, match="The parameter `act quant dtype` must be isinstance of QuantDtype, but got "
                                        "3."):
        qat.set_act_quant_dtype(act_quant_dtype=3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_act_quant_dtype_range():
    """
    Feature: set_act_quant_dtype api of SLB.
    Description: Feed QuantDtype type `act_quant_dtype` into set_act_quant_dtype() functional interface.
    Expectation: Except ValueError.
    """

    qat = SlbQAT()
    with pytest.raises(ValueError, match="Only supported if `act_quant_dtype` is `QuantDtype.INT8` yet."):
        qat.set_act_quant_dtype(act_quant_dtype=QuantDtype.INT1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_enable_act_quant():
    """
    Feature: set_enable_act_quant api of SlbQAT.
    Description: Check default value of enable_act_quant and value after called set_enable_act_quant.
    Expectation: Config success.
    """
    qat = SlbQAT()
    assert not qat._config.enable_act_quant
    qat.set_enable_act_quant(True)
    assert qat._config.enable_act_quant


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_enable_bn_calibration():
    """
    Feature: set_enable_bn_calibration api of SlbQAT.
    Description: Check default value of enable_bn_calibration and value after called set_enable_bn_calibration.
    Expectation: Config success.
    """
    qat = SlbQAT()
    assert not qat._config.enable_bn_calibration
    qat.set_enable_bn_calibration(True)
    assert qat._config.enable_bn_calibration


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_epoch_size_type():
    """
    Feature: set_epoch_size api of SlbQAT.
    Description: Feed float type `epoch_size` into set_epoch_size() functional interface.
    Expectation: Except TypeError.
    """

    qat = SlbQAT()
    with pytest.raises(TypeError, match="For 'SlbQuantAwareTraining', the 'epoch_size' must be int, but got 'float'"):
        qat.set_epoch_size(epoch_size=3.2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_epoch_size_range():
    """
    Feature: set_epoch_size api of SlbQAT.
    Description: Feed int type `epoch_size` into set_epoch_size() functional interface.
    Expectation: Except ValueError.
    """

    qat = SlbQAT()
    with pytest.raises(ValueError, match="For 'SlbQuantAwareTraining', the 'epoch_size' must be int and must > 0, but "
                                         "got '-1' with type 'int'."):
        qat.set_epoch_size(epoch_size=-1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_has_trained_epoch_type():
    """
    Feature: set_has_trained_epoch api of SlbQAT.
    Description: Feed float type `has_trained_epoch` into set_has_trained_epoch() functional interface.
    Expectation: Except TypeError.
    """

    qat = SlbQAT()
    with pytest.raises(TypeError, match="For 'SlbQuantAwareTraining', the 'has_trained_epoch' must be int, but got "
                                        "'float'"):
        qat.set_has_trained_epoch(has_trained_epoch=3.2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_has_trained_epoch_range():
    """
    Feature: set_has_trained_epoch api of SlbQAT.
    Description: Feed int type `has_trained_epoch` into set_has_trained_epoch() functional interface.
    Expectation: Except ValueError.
    """

    qat = SlbQAT()
    with pytest.raises(ValueError, match="For 'SlbQuantAwareTraining', the 'has_trained_epoch' must be int and must >= "
                                         "0, but got '-10' with type 'int'."):
        qat.set_has_trained_epoch(has_trained_epoch=-10)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_t_start_val_type():
    """
    Feature: set_t_start_val api of SlbQAT.
    Description: Feed int type `t_start_val` into set_t_start_val() functional interface.
    Expectation: Except TypeError.
    """

    qat = SlbQAT()
    with pytest.raises(TypeError, match="For 'SlbQuantAwareTraining', the 't_start_val' must be float, but got 'int'"):
        qat.set_t_start_val(t_start_val=2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_t_start_val_range():
    """
    Feature: set_t_start_val api of SlbQAT.
    Description: Feed float type `t_start_val` into set_t_start_val() functional interface.
    Expectation: Except ValueError.
    """

    qat = SlbQAT()
    with pytest.raises(ValueError, match="For 'SlbQuantAwareTraining', the 't_start_val' must be float and must > 0, "
                                         "but got '-2.1' with type 'float'."):
        qat.set_t_start_val(t_start_val=-2.1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_t_start_time_type():
    """
    Feature: set_t_start_time api of SlbQAT.
    Description: Feed int type `t_start_time` into set_t_start_time() functional interface.
    Expectation: Except TypeError.
    """

    qat = SlbQAT()
    with pytest.raises(TypeError, match="For 'SlbQuantAwareTraining', the 't_start_time' must be 'float',  but got "
                                        "'int'."):
        qat.set_t_start_time(t_start_time=2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_t_start_time_range():
    """
    Feature: set_t_start_time api of SlbQAT.
    Description: Feed float type `t_start_time` into set_t_start_time() functional interface.
    Expectation: Except ValueError.
    """

    qat = SlbQAT()
    with pytest.raises(ValueError, match="For 'SlbQuantAwareTraining', the 't_start_time' must be in range of "
                                         "\\[0.0, 1.0\\], but got -2.1 with type 'float'."):
        qat.set_t_start_time(t_start_time=-2.1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_t_end_time_type():
    """
    Feature: set_t_end_time api of SlbQAT.
    Description: Feed int type `t_end_time` into set_t_end_time() functional interface.
    Expectation: Except TypeError.
    """

    qat = SlbQAT()
    with pytest.raises(TypeError, match="For 'SlbQuantAwareTraining', the 't_end_time' must be 'float',  but got "
                                        "'int'."):
        qat.set_t_end_time(t_end_time=2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_t_end_time_range():
    """
    Feature: set_t_end_time api of SlbQAT.
    Description: Feed float type `t_end_time` into set_t_end_time() functional interface.
    Expectation: Except ValueError.
    """

    qat = SlbQAT()
    with pytest.raises(ValueError, match="For 'SlbQuantAwareTraining', the 't_end_time' must be in range of "
                                         "\\[0.0, 1.0\\], but got -2.1 with type 'float'."):
        qat.set_t_end_time(t_end_time=-2.1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_t_factor_type():
    """
    Feature: set_t_factor api of SlbQAT.
    Description: Feed int type `t_factor` into set_t_factor() functional interface.
    Expectation: Except TypeError.
    """

    qat = SlbQAT()
    with pytest.raises(TypeError, match="For 'SlbQuantAwareTraining', the 't_factor' must be float, but got 'int'"):
        qat.set_t_factor(t_factor=2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_t_factor_range():
    """
    Feature: set_t_factor api of SlbQAT.
    Description: Feed float type `t_factor` into set_t_factor() functional interface.
    Expectation: Except ValueError.
    """

    qat = SlbQAT()
    with pytest.raises(ValueError, match="For 'SlbQuantAwareTraining', the 't_factor' must be float and must > 0, but "
                                         "got '-2.1' with type 'float'."):
        qat.set_t_factor(t_factor=-2.1)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_callbacks_epoch_initial():
    """
    Feature: callbacks api of SlbQAT.
    Description: Not feed `epoch_size` and `has_trained_epoch`.
    Expectation: Except RuntimeError.
    """

    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../models/research/cv/'))
    from lenet.src.dataset import create_dataset as create_mnist_ds
    context.set_context(mode=context.GRAPH_MODE)
    data_path = "/home/workspace/mindspore_dataset/mnist/train"
    ds_train = create_mnist_ds(data_path, 32, 1)

    network = NetToQuant()
    qat = SlbQAT()
    new_network = qat.apply(network)
    model = Model(new_network)
    with pytest.raises(RuntimeError, match="The `epoch_size` need to be initialized!"):
        qat.callbacks(model=model, dataset=ds_train)
    qat.set_epoch_size(100)
    with pytest.raises(RuntimeError, match="The `has_trained_epoch` need to be initialized!"):
        qat.callbacks(model=model, dataset=ds_train)
    qat.set_has_trained_epoch(0)
    qat.callbacks(model=model, dataset=ds_train)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_callbacks_epoch_range_compare():
    """
    Feature: callbacks api of SlbQAT.
    Description: Feed incorrect `epoch_size`, `has_trained_epoch`, `t_start_time` and `t_end_time`.
    Expectation: Except ValueError.
    """

    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../models/research/cv/'))
    from lenet.src.dataset import create_dataset as create_mnist_ds
    context.set_context(mode=context.GRAPH_MODE)
    data_path = "/home/workspace/mindspore_dataset/mnist/train"
    ds_train = create_mnist_ds(data_path, 32, 1)

    network = NetToQuant()
    qat = SlbQAT()
    new_network = qat.apply(network)
    model = Model(new_network)
    qat.set_epoch_size(epoch_size=100)
    qat.set_has_trained_epoch(has_trained_epoch=120)
    with pytest.raises(ValueError, match="The `epoch_size` should be greater than `has_trained_epoch`."):
        qat.callbacks(model=model, dataset=ds_train)
    qat.set_epoch_size(epoch_size=100)
    qat.set_has_trained_epoch(has_trained_epoch=0)
    qat.set_t_start_time(t_start_time=0.7)
    qat.set_t_end_time(t_end_time=0.4)
    with pytest.raises(ValueError, match="The `t_end_time` should not be less than `t_start_time`."):
        qat.callbacks(model=model, dataset=ds_train)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_callbacks_model_dataset_type():
    """
    Feature: callbacks api of SlbQAT.
    Description: Feed int type `model` or int type `dataset` into callbacks() functional interface.
    Expectation: Except TypeError.
    """

    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../models/research/cv/'))
    from lenet.src.dataset import create_dataset as create_mnist_ds
    context.set_context(mode=context.GRAPH_MODE)
    data_path = "/home/workspace/mindspore_dataset/mnist/train"
    ds_train = create_mnist_ds(data_path, 32, 1)

    qat = SlbQAT()
    qat.set_epoch_size(epoch_size=100)
    qat.set_has_trained_epoch(has_trained_epoch=0)
    qat.set_t_start_time(t_start_time=0.2)
    qat.set_t_end_time(t_end_time=0.6)
    with pytest.raises(TypeError, match="The parameter `model` must be isinstance of mindspore.Model, but got 10."):
        qat.callbacks(model=10, dataset=ds_train)

    network = NetToQuant()
    new_network = qat.apply(network)
    model = Model(new_network)
    with pytest.raises(TypeError, match="The parameter `dataset` must be isinstance of mindspore.dataset.Dataset, but "
                                        "got 5."):
        qat.callbacks(model=model, dataset=5)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_convert_network_type():
    """
    Feature: convert api of SlbQAT.
    Description: Feed int type `net_opt`, int type `ckpt_path`, incorrect `ckpt_path` into convert() functional
                 interface.
    Expectation: Except TypeError.
    """

    qat = SlbQAT()
    with pytest.raises(TypeError, match="The parameter `net_opt` must be isinstance of Cell, but got <class 'int'>."):
        qat.convert(net_opt=10)

    network = NetToQuant()
    new_network = qat.apply(network)
    with pytest.raises(TypeError, match="The parameter `ckpt_path` must be isinstance of str, but got <class 'int'>."):
        qat.convert(net_opt=new_network, ckpt_path=5)

    with pytest.raises(ValueError, match="The parameter `ckpt_path` can only be empty or a valid file, but got "):
        qat.convert(net_opt=new_network, ckpt_path="an_invalid_test_path")
