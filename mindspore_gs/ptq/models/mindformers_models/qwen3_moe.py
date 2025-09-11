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
"""qwen3 quant model"""

from mindspore_gs.ptq.models.mindformers_models.mf_model import MFModel, MFModelEnableSafeTensors
from mindspore_gs.ptq.models.mindformers_models.param_processor import (MoeParamProcessor,
                                                                        QKVParamProcessor,
                                                                        FFNParamProcessor)
from mindspore_gs.ptq.utils import QuantType
from .qwen3 import QWen3


@MFModel.reg_model('qwen3_moe')
class QWen3MoE(QWen3):
    """QWen3"""
    def _process_params_dict_before_save(self, param_dict) -> tuple[dict, dict]:
        # pylint: disable=bad-super-call
        super(MFModelEnableSafeTensors, self)._process_params_dict_before_save(param_dict)

        # Apply MoE split
        moe_processor = MoeParamProcessor(self.network)
        param_dict, param_name_trace = moe_processor.split_param(param_dict)

        param_dict, super_trace = super()._process_params_dict_before_save(param_dict)
        param_name_trace.update(super_trace)

        return param_dict, param_name_trace

    def get_description_file(self, network):
        """
        Obtain the description of quantization type for each parameter in each layer of the network.
        Such as W8A8 or W4A8_DYNAMIC
        """
        results = self._get_quant_type(network)

        moe_processor = MoeParamProcessor(self.network)
        results = moe_processor.split_name(results)

        qkv_processor = QKVParamProcessor(self.network)
        results = qkv_processor.split_name(results)

        ffn_processor = FFNParamProcessor(self.network)
        results = ffn_processor.split_name(results)

        desc_info = {}
        param_dict = self.parameters_dict()
        for key, _ in param_dict.items():
            if key in results.keys():
                desc_info[key] = results[key]
            else:
                desc_info[key] = QuantType.FLOAT.value
        return desc_info
