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

    def _get_description_file(self, network):
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

        hf_results = {}
        for key, value in results.items():
            hf_results[self._convert_param_names_to_hf(key)] = value

        desc_info = {}
        param_dict = self.parameters_dict()
        for key, _ in param_dict.items():
            if key in hf_results:
                desc_info[key] = hf_results[key]
            else:
                desc_info[key] = QuantType.FLOAT.value
        return desc_info

    @classmethod
    def _convert_param_names_to_hf(cls, param_name):
        """Convert mcore name to huggingface name.
        One parameter may correspond to multiple parameters in huggingface,
        so return a list of names."""
        rules = {
            "model.": "",
            "decoder.layers.": "model.layers.",
            ".self_attention.": ".self_attn.",
            "embedding.word_embeddings.": "model.embed_tokens.",
            "decoder.final_layernorm.": "model.norm.",
            "output_layer.": "lm_head.",
            ".pre_mlp_layernorm.": ".post_attention_layernorm.",
            ".q_layernorm.": ".q_norm.",
            ".k_layernorm.": ".k_norm.",
            ".linear_proj.": ".o_proj.",
            ".linear_q.": ".q_proj.",
            ".linear_k.": ".k_proj.",
            ".linear_v.": ".v_proj.",
            ".gating.": ".gate_proj.",
            ".hidden.": ".up_proj.",
            ".linear_fc2.": ".down_proj."
        }

        for old, new in rules.items():
            param_name = param_name.replace(old, new)

        return param_name
