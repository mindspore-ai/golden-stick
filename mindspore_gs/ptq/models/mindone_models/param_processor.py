# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Parameter processor of mindone quant model in deploy stage"""

from tqdm import tqdm
import numpy as np
import mindspore as ms
from mindspore import Tensor, dtype, Parameter
from mindspore import ops as msops
from mindspore.ops.operations.comm_ops import ReduceOp
from mindspore.communication.management import GlobalComm
from mindspore.communication import get_rank
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq.basic_functions.basic_quant_func import np_int4data_pack_to_int8
from mindspore_gs.ptq.utils import QuantType


class ParamProcessor:
    """parameter processor for different backend in deploy stage"""

    def __init__(self, backend: BackendTarget):
        self.backend = backend
        self.axw4_processor = AxW4ParamProcessor()
        self.static_a8wx_processor = StaticA8WXParamProcessor()
        self.static_a8wx_processed = set()

    def process_param_dict(self, param_dict: dict, param_desc: dict) -> dict:
        """process param dict in different backend."""
        if self.backend == BackendTarget.NONE:
            return param_dict
        if self.backend == BackendTarget.ASCEND:
            return self._deploy_ascend_param_dict(param_dict, param_desc)
        raise ValueError(f"Unsupported backend: {self.backend}")

    def process_param_desc(self, param_desc: dict) -> dict:
        """process param description."""
        if self.backend == BackendTarget.NONE:
            return param_desc
        if self.backend == BackendTarget.ASCEND:
            return self._deploy_ascend_param_desc(param_desc)
        raise ValueError(f"Unsupported backend: {self.backend}")

    def _deploy_ascend_param_dict(self, param_dict: dict, param_desc: dict) -> dict:
        """deploy parameter dictionary for ascend backend."""
        axw4_support_quant_types = [QuantType.W4A16.value,
                                    QuantType.W4A8_DYNAMIC.value]
        static_a8wx_support_quant_types = [QuantType.W8A8.value]

        for param_name, quant_type in tqdm(param_desc.items(),
                                           desc="Processing parameters dict for Ascend backend"):
            if quant_type in axw4_support_quant_types:
                # Process W4A8 quantized parameters
                new_param = self.axw4_processor.process_param(param_name,
                                                              param_dict[param_name])
                param_dict[param_name] = new_param
            elif quant_type in static_a8wx_support_quant_types:
                # param_name: model.layers.0.self_attn.o_proj.weight
                # param_prefix: model.layers.0.self_attn.o_proj
                # Process w8a8 by layer as the minimum process dimension.
                param_prefix = param_name.rsplit('.', 1)[0]
                if param_prefix in self.static_a8wx_processed:
                    continue
                self.static_a8wx_processed.add(param_prefix)
                self.static_a8wx_processor.process_param(param_prefix, param_dict)
            else:
                continue
        self.static_a8wx_processed.clear()
        return param_dict

    def _deploy_ascend_param_desc(self, param_desc: dict) -> dict:
        """deploy parameter dictionary for ascend backend."""
        static_a8wx_support_quant_types = [QuantType.W8A8.value]

        add_quantization_desc = {}
        for param_name, quant_type in tqdm(param_desc.items(),
                                           desc="Processing parameters description for Ascend backend"):
            if quant_type in static_a8wx_support_quant_types:
                # param_name: model.layers.0.self_attn.o_proj.weight
                # param_prefix: model.layers.0.self_attn.o_proj
                # Process w8a8 by layer as the minimum process dimension.
                param_prefix = param_name.rsplit('.', 1)[0]
                if param_prefix in self.static_a8wx_processed:
                    continue
                self.static_a8wx_processed.add(param_prefix)
                add_quantization_desc[param_prefix + ".deq_scale"] = \
                    param_desc[param_prefix + ".weight"]
                add_quantization_desc[param_prefix + ".quant_bias"] = \
                    param_desc[param_prefix + ".weight"]
            else:
                continue
        param_desc.update(add_quantization_desc)
        self.static_a8wx_processed.clear()
        return param_desc


class AxW4ParamProcessor:
    """AxW4 parameter processor"""

    def process_param(self, param_name: str, param: Parameter) -> Parameter:
        """Process parameter according to quantization type."""
        if param_name.endswith(".weight"):
            param = param.asnumpy().T
            pack_weight = self._pack_int4_weight(param)
            return Parameter(Tensor(pack_weight, dtype=dtype.qint4x2))
        return param

    def _pack_int4_weight(self, param: np.ndarray) -> np.ndarray:
        """pack int4 weight"""
        return np_int4data_pack_to_int8(param)


class StaticA8WXParamProcessor:
    """A8WX parameter processor"""

    def process_param(self, param_prefix: str, param_dict) -> Parameter:
        """Process parameter according to quantization type."""
        need_allreduce = False
        trans_b = True
        compute_dtype = param_dict[param_prefix + ".input_scale"].dtype
        self.get_dequant_scale(param_dict, param_prefix)
        self.correction_into_bias(param_dict, param_prefix, trans_b, compute_dtype, need_allreduce)
        self.process_input_scale_and_offset(param_dict, param_prefix, compute_dtype)

    @staticmethod
    def get_dequant_scale(param_dict, param_prefix):
        """_get_dequant_scale"""
        input_scale_name = param_prefix + ".input_scale"
        weight_scale_name = param_prefix + ".weight_scale"
        deq_scale_name = param_prefix + ".deq_scale"
        input_scale = param_dict[input_scale_name].astype(ms.float32).asnumpy()
        weight_scale = param_dict[weight_scale_name].astype(ms.float32).asnumpy()
        deq_scale = input_scale * weight_scale
        param_dict[deq_scale_name] = Parameter(Tensor(deq_scale, dtype=ms.float32))

    @staticmethod
    def correction_into_bias(param_dict, param_prefix, trans_b, compute_type, need_allreduce=False):
        """_correction_into_bias"""
        input_scale_name = param_prefix + ".input_scale"
        input_offset_name = param_prefix + ".input_offset"
        weight_name = param_prefix + ".weight"
        weight_scale_name = param_prefix + ".weight_scale"
        quant_bias_name = param_prefix + ".quant_bias"
        input_scale = param_dict[input_scale_name]
        input_offset = param_dict[input_offset_name]
        q_weight = param_dict[weight_name]
        weight_scale = param_dict[weight_scale_name]
        x_zp = input_offset.asnumpy()
        q_correction = -np.sum(x_zp.astype(np.int32) * q_weight.asnumpy().astype(np.int32),
                                axis=-1 if trans_b else -2).astype(np.int32)
        if need_allreduce:
            t_q_correction = Tensor(q_correction)
            t_q_correction = msops.AllReduce(op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)(t_q_correction)
            q_correction = t_q_correction.asnumpy()

        # for align precision
        deq_scale_np = (input_scale.asnumpy() * weight_scale.asnumpy()).astype(np.float64)
        q_correction = q_correction.astype(np.float64) * deq_scale_np
        q_correction_t = Tensor(q_correction, dtype=compute_type)
        deq_scale_t = input_scale.astype(np.float32) * weight_scale.astype(np.float32)
        q_correction_t = msops.round(q_correction_t / deq_scale_t).astype(dtype.int32)
        if need_allreduce and get_rank() != 0:
            q_correction_t = msops.zeros_like(q_correction_t)
        param_dict[quant_bias_name] = Parameter(q_correction_t)

    @staticmethod
    def process_input_scale_and_offset(param_dict, param_prefix, compute_dtype):
        """_process_input_scale_and_offset"""
        input_scale_name = param_prefix + ".input_scale"
        input_offset_name = param_prefix + ".input_offset"
        weight_name = param_prefix + ".weight"
        input_scale = param_dict[input_scale_name]
        intput_offset = param_dict[input_offset_name]
        input_scale = input_scale.astype(ms.float32).asnumpy()
        intput_offset = intput_offset.astype(ms.float32).asnumpy()
        ic = param_dict[weight_name].shape[1]
        input_scale = np.repeat(input_scale, ic)
        intput_offset = np.repeat(intput_offset, ic)
        param_dict[input_scale_name] = Parameter(Tensor(input_scale, dtype=compute_dtype))
        param_dict[input_offset_name] = Parameter(Tensor(intput_offset, dtype=ms.int8))
