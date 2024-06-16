# Copyright 2024 Huawei Technologies Co., Ltd
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
"""test KVCache Int8 algorithm."""
import os
import sys
from collections import OrderedDict

import pytest
import numpy as np
import mindspore
from mindspore import context, Parameter, dtype, GRAPH_MODE, PYNATIVE_MODE, Tensor, nn, QuantDtype
from mindspore.common.initializer import initializer

from mindspore_gs.quantization.fake_quantizer import FakeQuantParamCell, FakeQuantParam
from mindspore_gs.ptq import RoundToNearest as RTN
from mindspore_gs.ptq.quant_cells import KVCacheMgrQuant
from mindspore_gs.ptq.convert_utils import AntiQuantCell, QuantCell
from mindspore_gs.ptq.fake_quantizer import MinMaxPerChannel
from mindspore_gs.ptq.ptq_config import PTQConfig, PTQMode
from mindspore_gs.common.gs_enum import BackendTarget
from mindformers.modules import KVCacheMgr

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
# pylint: disable=wrong-import-position
from tests.st.models.llama2 import llama2, create_dummy_inputs
from tests.st.test_utils import check_network_contain_layer, relative_tolerance_acceptable, \
    absolute_tolerance_acceptable


class SimpleNet(nn.Cell):
    """
    Network with single linear to be quant
    """

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.kvcache = KVCacheMgr(8, 16, 2, 512)
        kv_shape = self.kvcache.key_past.shape
        kv_dtype = self.kvcache.key_past.dtype
        self.kvcache.key_past = Parameter(initializer('normal', kv_shape, kv_dtype), name=self.kvcache.key_past.name,
                                          requires_grad=False)
        self.kvcache.value_past = Parameter(initializer('normal', kv_shape, kv_dtype),
                                            name=self.kvcache.value_past.name, requires_grad=False)

    def construct(self, key, value, kvcache_inputs):
        return self.kvcache(key, value, kvcache_inputs)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_apply_convert():
    """
    Feature: RoundToNearestPTQ algorithm set functions.
    Description: Apply RoundToNearestPTQ on SimpleNet.
    Expectation: Apply success and coordinate attributes are same as config.
    """

    mindspore.set_context(device_target="CPU", mode=mindspore.GRAPH_MODE)
    network = SimpleNet()
    kv_shape = network.kvcache.key_past.shape
    kv_dtype = network.kvcache.key_past.dtype
    network.kvcache.key_past = Parameter(initializer('ones', kv_shape, kv_dtype), name=network.kvcache.key_past.name,
                                         requires_grad=False)
    network.kvcache.value_past = Parameter(initializer('ones', kv_shape, kv_dtype),
                                           name=network.kvcache.value_past.name, requires_grad=False)
    cfg = PTQConfig(mode=PTQMode.QUANTIZE,
                    backend=BackendTarget.NONE)
    ptq = RTN(config=cfg)
    # pylint: disable=W0212
    ptq._config.weight_only = False
    # pylint: disable=W0212
    ptq._config.enable_kvcache_int8 = True
    # apply
    new_network = ptq.apply(network)
    cells: OrderedDict = new_network.name_cells()
    quant_cell = cells.get("kvcache", None)
    assert isinstance(quant_cell, KVCacheMgrQuant)
    assert quant_cell.weight_quantizer() is None
    assert quant_cell.input_quantizer() is None
    assert quant_cell.output_quantizer() is None

    key_input_quantizer: MinMaxPerChannel = quant_cell._key_input_quantizer
    assert isinstance(key_input_quantizer, MinMaxPerChannel)
    assert key_input_quantizer.symmetric()
    assert key_input_quantizer.quant_dtype() == QuantDtype.INT8
    assert key_input_quantizer.is_per_channel()
    assert not key_input_quantizer.narrow_range()
    assert key_input_quantizer.num_bits() == 8

    key_output_quantizer: MinMaxPerChannel = quant_cell._key_output_quantizer
    assert isinstance(key_output_quantizer, MinMaxPerChannel)
    assert key_output_quantizer.symmetric()
    assert key_output_quantizer.quant_dtype() == QuantDtype.INT8
    assert key_output_quantizer.is_per_channel()
    assert not key_output_quantizer.narrow_range()
    assert key_output_quantizer.num_bits() == 8

    value_input_quantizer: MinMaxPerChannel = quant_cell._value_input_quantizer
    assert isinstance(value_input_quantizer, MinMaxPerChannel)
    assert value_input_quantizer.symmetric()
    assert value_input_quantizer.quant_dtype() == QuantDtype.INT8
    assert value_input_quantizer.is_per_channel()
    assert not value_input_quantizer.narrow_range()
    assert value_input_quantizer.num_bits() == 8

    value_output_quantizer: MinMaxPerChannel = quant_cell._value_output_quantizer
    assert isinstance(value_output_quantizer, MinMaxPerChannel)
    assert value_output_quantizer.symmetric()
    assert value_output_quantizer.quant_dtype() == QuantDtype.INT8
    assert value_output_quantizer.is_per_channel()
    assert not value_output_quantizer.narrow_range()
    assert value_output_quantizer.num_bits() == 8

    # calibrate
    # pylint: disable=W0212
    ptq._calibrate(network)

    quant_params = key_input_quantizer.quant_params()
    min_data = np.array(quant_params.get("min"))
    max_data = np.array(quant_params.get("max"))
    # BNSD: (2, 8, 512, 16) --> BSND: (2, 512, 8, 16) --> BSH: (2, 512, 128)
    assert min_data.shape == (1, 1, 128)
    assert max_data.shape == (1, 1, 128)
    for min_ in min_data.flatten().tolist():
        assert min_ == 1.
    for max_ in max_data.flatten().tolist():
        assert max_ == 1.

    # convert
    new_network = ptq.convert(new_network)
    cells: OrderedDict = new_network.name_cells()

    quant_cell = cells.get("kvcache", None)
    assert isinstance(quant_cell, KVCacheMgrQuant)
    key_input_quantizer: FakeQuantParamCell = quant_cell._key_input_quantizer
    assert isinstance(key_input_quantizer, FakeQuantParamCell)
    assert isinstance(key_input_quantizer.fq, FakeQuantParam)
    key_output_quantizer: FakeQuantParamCell = quant_cell._key_output_quantizer
    assert isinstance(key_output_quantizer, FakeQuantParamCell)
    assert isinstance(key_output_quantizer.fq, FakeQuantParam)
    value_input_quantizer: FakeQuantParamCell = quant_cell._value_input_quantizer
    assert isinstance(value_input_quantizer, FakeQuantParamCell)
    assert isinstance(value_input_quantizer.fq, FakeQuantParam)
    value_output_quantizer: FakeQuantParamCell = quant_cell._value_output_quantizer
    assert isinstance(value_output_quantizer, FakeQuantParamCell)
    assert isinstance(value_output_quantizer.fq, FakeQuantParam)

    assert quant_cell.weight_quantizer() is None
    assert quant_cell.input_quantizer() is None
    assert quant_cell.output_quantizer() is None


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", ["Ascend"])
@pytest.mark.parametrize("mode", [GRAPH_MODE])
def test_kvint8_predict_1stage(device, mode):
    """
    Feature: RoundToNearestPTQ algorithm set functions.
    Description: Apply, Convert and Predict RoundToNearestPTQ on SimpleNet.
    Expectation: Execute success.
    """

    context.set_context(device_target=device, mode=mode)
    network = SimpleNet()
    cfg = PTQConfig(mode=PTQMode.QUANTIZE,
                    backend=BackendTarget.ASCEND)
    ptq = RTN(config=cfg)
    # pylint: disable=W0212
    ptq._config.weight_only = False
    # pylint: disable=W0212
    ptq._config.enable_kvcache_int8 = True
    quant_network = ptq.apply(network)
    quant_network = ptq._calibrate(quant_network)
    ascend_network = ptq.convert(quant_network)
    for _, cell in ascend_network.name_cells().items():
        if not isinstance(cell, KVCacheMgrQuant):
            continue
        kvcache: KVCacheMgrQuant = cell
        assert not kvcache.input_quantizer()
        assert not kvcache.output_quantizer()
        assert not kvcache.weight_quantizer()
        assert isinstance(kvcache._key_input_quantizer, QuantCell)
        assert isinstance(kvcache._key_output_quantizer, AntiQuantCell)
        assert isinstance(kvcache._value_input_quantizer, QuantCell)
        assert isinstance(kvcache._value_output_quantizer, AntiQuantCell)
        kcache: Parameter = kvcache.handler().key_past
        vcache: Parameter = kvcache.handler().value_past
        assert isinstance(kcache, Parameter)
        assert isinstance(vcache, Parameter)
        assert kcache.dtype == dtype.int8
        assert kcache.value().dtype == dtype.int8
        assert vcache.dtype == dtype.int8
        assert vcache.value().dtype == dtype.int8
    # BNSD: (1, 8, 512, 16)
    key = Tensor(np.ones((1, 8, 100, 16), dtype=np.float32), dtype=dtype.float32)
    value = Tensor(np.ones((1, 8, 100, 16), dtype=np.float32), dtype=dtype.float32)
    batch_valid_length = Tensor([0], dtype=dtype.int64)
    zactivate_len = Tensor(np.ones((512,)), dtype=dtype.int64)
    batch_index = Tensor([0], dtype=dtype.int64)
    seq_length_tensor = Tensor([512], dtype=dtype.int64)
    kvcache_inputs = (batch_valid_length, zactivate_len, batch_index, seq_length_tensor)
    ascend_network(key, value, kvcache_inputs)
    kbuffer = ascend_network.kvcache.handler().key_past.asnumpy()
    vbuffer = ascend_network.kvcache.handler().value_past.asnumpy()
    assert kbuffer.shape == (2, 8, 512, 16)
    for b in range(1):
        for n in range(8):
            for s in range(100):
                for d in range(16):
                    assert kbuffer[b][n][s][d] == 127
    assert vbuffer.shape == (2, 8, 512, 16)
    for b in range(1):
        for n in range(8):
            for s in range(100):
                for d in range(16):
                    assert vbuffer[b][n][s][d] == 127


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", ["Ascend"])
@pytest.mark.parametrize("mode", [GRAPH_MODE])
def test_kvint8_predict_2stage(device, mode):
    """
    Feature: RoundToNearestPTQ algorithm set functions.
    Description: Apply, Convert and Predict RoundToNearestPTQ on SimpleNet.
    Expectation: Execute success.
    """

    context.set_context(device_target=device, mode=mode)

    def quant():
        network = SimpleNet()
        cfg = PTQConfig(mode=PTQMode.QUANTIZE,
                        backend=BackendTarget.ASCEND)
        ptq = RTN(config=cfg)
        # pylint: disable=W0212
        ptq._config.weight_only = False
        # pylint: disable=W0212
        ptq._config.enable_kvcache_int8 = True
        quant_network = ptq.apply(network)
        quant_network = ptq._calibrate(quant_network)
        ascend_network = ptq.convert(quant_network)
        for _, cell in ascend_network.name_cells().items():
            if not isinstance(cell, KVCacheMgrQuant):
                continue
            kvcache: KVCacheMgrQuant = cell
            assert not kvcache.input_quantizer()
            assert not kvcache.output_quantizer()
            assert not kvcache.weight_quantizer()
            assert isinstance(kvcache._key_input_quantizer, QuantCell)
            assert isinstance(kvcache._key_output_quantizer, AntiQuantCell)
            assert isinstance(kvcache._value_input_quantizer, QuantCell)
            assert isinstance(kvcache._value_output_quantizer, AntiQuantCell)
            kcache: Parameter = kvcache.handler().key_past
            vcache: Parameter = kvcache.handler().value_past
            assert isinstance(kcache, Parameter)
            assert isinstance(vcache, Parameter)
            assert kcache.dtype == dtype.int8
            assert kcache.value().dtype == dtype.int8
            assert vcache.dtype == dtype.int8
            assert vcache.value().dtype == dtype.int8
        mindspore.save_checkpoint(ascend_network, "test_kvint8_predict_2stage.ckpt")

    def infer():
        network = SimpleNet()
        cfg = PTQConfig(mode=PTQMode.DEPLOY,
                        backend=BackendTarget.ASCEND)
        ptq = RTN(config=cfg)
        # pylint: disable=W0212
        ptq._config.weight_only = False
        # pylint: disable=W0212
        ptq._config.enable_kvcache_int8 = True
        quant_network = ptq.apply(network)
        ascend_network = ptq.convert(quant_network)
        mindspore.load_checkpoint("test_kvint8_predict_2stage.ckpt", ascend_network)

        key = Tensor(np.ones((1, 8, 100, 16), dtype=np.float32), dtype=dtype.float32)
        value = Tensor(np.ones((1, 8, 100, 16), dtype=np.float32), dtype=dtype.float32)
        batch_valid_length = Tensor([0], dtype=dtype.int64)
        zactivate_len = Tensor(np.ones((512,)), dtype=dtype.int64)
        batch_index = Tensor([0], dtype=dtype.int64)
        seq_length_tensor = Tensor([512], dtype=dtype.int64)
        kvcache_inputs = (batch_valid_length, zactivate_len, batch_index, seq_length_tensor)
        ascend_network(key, value, kvcache_inputs)
        kbuffer = ascend_network.kvcache.handler().key_past.asnumpy()
        vbuffer = ascend_network.kvcache.handler().value_past.asnumpy()
        assert kbuffer.shape == (2, 8, 512, 16)
        for b in range(1):
            for n in range(8):
                for s in range(100):
                    for d in range(16):
                        assert kbuffer[b][n][s][d] == 127
        assert vbuffer.shape == (2, 8, 512, 16)
        for b in range(1):
            for n in range(8):
                for s in range(100):
                    for d in range(16):
                        assert vbuffer[b][n][s][d] == 127

    quant()
    infer()


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", ["Ascend", "CPU"])
@pytest.mark.parametrize("mode", [GRAPH_MODE, PYNATIVE_MODE])
def test_llama2_kvint8_apply_convert(device, mode):
    """
    Feature: RoundToNearestPTQ KVInt8 algorithm.
    Description: Apply KVInt8 quant on LLama2 and convert to ascend backend.
    Expectation: Execute successfully.

    Disabled because of miss of RMSNorm ops in mindspore2.3.
    """

    context.set_context(device_target=device, mode=mode)
    network = llama2(8, 512, 1024, 2, use_past=True)
    assert check_network_contain_layer(network, KVCacheMgr)
    cfg = PTQConfig(mode=PTQMode.QUANTIZE,
                    backend=BackendTarget.ASCEND)
    ptq = RTN(config=cfg)
    # pylint: disable=W0212
    ptq._config.weight_only = False
    # pylint: disable=W0212
    ptq._config.enable_kvcache_int8 = True
    cfg = ptq._config
    cfg.weight_only = False
    cfg.enable_kvcache_int8 = True

    quant_network = ptq.apply(network.model)
    quant_network = ptq._calibrate(quant_network)
    assert not check_network_contain_layer(quant_network, KVCacheMgr, (KVCacheMgrQuant,))
    assert check_network_contain_layer(quant_network, KVCacheMgrQuant)
    ascend_network = ptq.convert(quant_network)
    for _, cell in ascend_network.name_cells().items():
        if not isinstance(cell, KVCacheMgrQuant):
            continue
        kvcache: KVCacheMgrQuant = cell
        assert not kvcache.input_quantizer()
        assert not kvcache.output_quantizer()
        assert not kvcache.weight_quantizer()
        assert isinstance(kvcache._key_input_quantizer, QuantCell)
        assert isinstance(kvcache._key_output_quantizer, AntiQuantCell)
        assert isinstance(kvcache._value_input_quantizer, QuantCell)
        assert isinstance(kvcache._value_output_quantizer, AntiQuantCell)
        kcache: Parameter = kvcache.handler().key_past
        vcache: Parameter = kvcache.handler().value_past
        assert isinstance(kcache, Parameter)
        assert isinstance(vcache, Parameter)
        assert kcache.dtype == dtype.int8
        assert kcache.value().dtype == dtype.int8
        assert vcache.dtype == dtype.int8
        assert vcache.value().dtype == dtype.int8


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", ["Ascend"])
@pytest.mark.parametrize("mode", [GRAPH_MODE])
def test_llama2_kvint8_predict_1stage(device, mode):
    """
    Feature: RoundToNearestPTQ A16W8 algorithm.
    Description: Apply KVInt8 quant on LLama2 and convert to ascend backend.
    Expectation: Execute successfully.

    Disabled because of miss of RMSNorm ops in mindspore2.3.
    """

    context.set_context(device_target=device, mode=mode)
    inputs = create_dummy_inputs(8, 512, 512)
    network = llama2(8, 512, 2048, 2, use_past=True)
    fp_outputs = network(*inputs)

    cfg = PTQConfig(mode=PTQMode.QUANTIZE,
                    backend=BackendTarget.ASCEND)
    ptq = RTN(config=cfg)
    # pylint: disable=W0212
    ptq._config.weight_only = False
    # pylint: disable=W0212
    ptq._config.enable_kvcache_int8 = True

    quant_network = ptq.apply(network.model)
    # calibrate
    for _, cell in network.name_cells().items():
        if not isinstance(cell, KVCacheMgr):
            continue
        kvcachemgr: KVCacheMgr = cell
        kv_shape = kvcachemgr.key_past.shape
        kv_dtype = kvcachemgr.key_past.dtype
        kvcachemgr.key_past = Parameter(initializer('normal', kv_shape, kv_dtype), name=kvcachemgr.key_past.name,
                                        requires_grad=False)
        kvcachemgr.value_past = Parameter(initializer('normal', kv_shape, kv_dtype), name=kvcachemgr.value_past.name,
                                          requires_grad=False)
    quant_network = ptq._calibrate(quant_network)

    ascend_network = ptq.convert(quant_network)
    network.model = ascend_network
    quant_outputs = network(*inputs)

    assert len(fp_outputs) == 3
    assert fp_outputs[0].shape == (8, 32000)
    assert fp_outputs[0].dtype == dtype.float32
    assert fp_outputs[1].shape == (8, 512)
    assert fp_outputs[1].dtype == dtype.int32
    assert fp_outputs[2].shape == (8, 512)
    assert fp_outputs[2].dtype == dtype.float32

    assert len(quant_outputs) == 3
    assert quant_outputs[0].shape == (8, 32000)
    assert quant_outputs[0].dtype == dtype.float32
    assert quant_outputs[1].shape == (8, 512)
    assert quant_outputs[1].dtype == dtype.int32
    assert quant_outputs[2].shape == (8, 512)
    assert quant_outputs[2].dtype == dtype.float32

    context.set_context(device_target="CPU", mode=mode)
    assert relative_tolerance_acceptable(quant_outputs[0].asnumpy(), fp_outputs[0].asnumpy(), 5e-2)
    assert relative_tolerance_acceptable(quant_outputs[1].asnumpy(), fp_outputs[1].asnumpy(), 5e-2)
    assert relative_tolerance_acceptable(quant_outputs[2].asnumpy(), fp_outputs[2].asnumpy(), 5e-2)


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", ["Ascend"])
@pytest.mark.parametrize("mode", [GRAPH_MODE])
def test_llama2_kvint8_predict_2stage(device, mode):
    """
    Feature: RoundToNearestPTQ A16W8 algorithm.
    Description: Apply KVInt8 quant on LLama2 and convert to ascend backend.
    Expectation: Execute successfully.

    Disabled because of miss of RMSNorm ops in mindspore2.3.
    """

    context.set_context(device_target=device, mode=mode)

    def quant(inputs):
        network = llama2(8, 512, 2048, 2, use_past=True)
        fp_outputs = network(*inputs)

        cfg = PTQConfig(mode=PTQMode.QUANTIZE,
                        backend=BackendTarget.ASCEND)
        ptq = RTN(config=cfg)
        # pylint: disable=W0212
        ptq._config.weight_only = False
        # pylint: disable=W0212
        ptq._config.enable_kvcache_int8 = True
        quant_network = ptq.apply(network.model)
        # calibrate
        for _, cell in network.name_cells().items():
            if not isinstance(cell, KVCacheMgr):
                continue
            kvcachemgr: KVCacheMgr = cell
            kv_shape = kvcachemgr.key_past.shape
            kv_dtype = kvcachemgr.key_past.dtype
            kvcachemgr.key_past = Parameter(initializer('normal', kv_shape, kv_dtype), name=kvcachemgr.key_past.name,
                                            requires_grad=False)
            kvcachemgr.value_past = Parameter(initializer('normal', kv_shape, kv_dtype),
                                              name=kvcachemgr.value_past.name,
                                              requires_grad=False)
        quant_network = ptq._calibrate(quant_network)
        ascend_network = ptq.convert(quant_network)
        network.model = ascend_network
        mindspore.save_checkpoint(network, "test_llama2_kvint8_predict_2stage.ckpt")
        return fp_outputs

    def infer(inputs):
        network = llama2(8, 512, 2048, 2, use_past=True)
        cfg = PTQConfig(mode=PTQMode.DEPLOY,
                        backend=BackendTarget.ASCEND)
        ptq = RTN(config=cfg)
        # pylint: disable=W0212
        ptq._config.weight_only = False
        # pylint: disable=W0212
        ptq._config.enable_kvcache_int8 = True
        quant_network = ptq.apply(network.model)
        ascend_network = ptq.convert(quant_network)
        network.model = ascend_network
        mindspore.load_checkpoint("test_llama2_kvint8_predict_2stage.ckpt", network)
        return network(*inputs)

    inputs = create_dummy_inputs(8, 512, 512)
    fp_outputs = quant(inputs)
    quant_outputs = infer(inputs)
    assert len(fp_outputs) == 3
    assert fp_outputs[0].shape == (8, 32000)
    assert fp_outputs[0].dtype == dtype.float32
    assert fp_outputs[1].shape == (8, 512)
    assert fp_outputs[1].dtype == dtype.int32
    assert fp_outputs[2].shape == (8, 512)
    assert fp_outputs[2].dtype == dtype.float32

    assert len(quant_outputs) == 3
    assert quant_outputs[0].shape == (8, 32000)
    assert quant_outputs[0].dtype == dtype.float32
    assert quant_outputs[1].shape == (8, 512)
    assert quant_outputs[1].dtype == dtype.int32
    assert quant_outputs[2].shape == (8, 512)
    assert quant_outputs[2].dtype == dtype.float32

    context.set_context(device_target="CPU", mode=mode)
    assert absolute_tolerance_acceptable(quant_outputs[0].asnumpy(), fp_outputs[0].asnumpy(), 1e-2)
    assert relative_tolerance_acceptable(quant_outputs[1].asnumpy(), fp_outputs[1].asnumpy(), 5e-2)
    assert relative_tolerance_acceptable(quant_outputs[2].asnumpy(), fp_outputs[2].asnumpy(), 5e-2)
