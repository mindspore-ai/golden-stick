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
from mindspore import Tensor, dtype, Parameter

from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq.basic_functions.basic_quant_func import np_int4data_pack_to_int8
from mindspore_gs.ptq.utils import QuantType

class ParamProcessor:
    """parameter processor for different backend in deploy stage"""

    def __init__(self, backend: BackendTarget, quantization_desc: dict):
        self.backend = backend
        self.quantization_desc = quantization_desc
        self.axw4_processor = AxW4ParamProcessor()

    def deploy(self, param_dict: dict) -> dict:
        """Deploy parameter dictionary.
        """
        if self.backend == BackendTarget.NONE:
            return param_dict
        if self.backend == BackendTarget.ASCEND:
            return self._deploy_ascend(param_dict)
        raise ValueError(f"Unsupported backend: {self.backend}")

    def _deploy_ascend(self, param_dict: dict) -> dict:
        """Deploy parameter dictionary for ascend backend.
        """
        axw4_support_quant_types = [QuantType.W4A16.value,
                                    QuantType.W4A8_DYNAMIC.value]
        for param_name, quant_type in tqdm(self.quantization_desc.items(),
                                           desc="Processing parameters for Ascend backend"):
            if quant_type in axw4_support_quant_types:
                # Process W4A8 quantized parameters
                new_param = self.axw4_processor.process_param(param_name,
                                                              param_dict[param_name])
                param_dict[param_name] = new_param
            else:
                continue
        return param_dict


class AxW4ParamProcessor:
    """AxW4 parameter processor"""

    def process_param(self, param_name: str, param: Parameter) -> Parameter:
        """Process parameter according to quantization type."""
        if param_name.endswith(".weight"):
            param = param.asnumpy().T
            pack_weight = self._pack_int4_weight(param)
            pack_weight = pack_weight.T
            return Parameter(Tensor(pack_weight, dtype=dtype.qint4x2))
        return param

    def _pack_int4_weight(self, param: np.ndarray) -> np.ndarray:
        """pack int4 weight"""
        return np_int4data_pack_to_int8(param)
