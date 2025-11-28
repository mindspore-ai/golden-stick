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
"""telechat2 quant model"""

from mindspore.nn.cell import Cell
from mindspore_gs.ptq.models.mindformers_models.mf_model import MFModel, MFModelEnableSafeTensors
from mindspore_gs.ptq.models.mindformers_models.param_processor import (QKVParamProcessor,
                                                                        FFNParamProcessor)
from mindspore_gs.ptq.utils import QuantType


@MFModel.reg_model('telechat2')
class Telechat2(MFModelEnableSafeTensors):
    """Telechat2"""

    def _process_params_dict_before_save(self, param_dict) -> tuple[dict, dict]:
        """_process_params_dict_before_save"""
        param_dict, param_name_trace = super()._process_params_dict_before_save(param_dict)

        # Apply QKV split
        qkv_processor = QKVParamProcessor(self.network)
        param_dict, qkv_trace = qkv_processor.split_param(param_dict)
        param_name_trace.update(qkv_trace)

        # Apply FFN split
        ffn_processor = FFNParamProcessor(self.network)
        param_dict, ffn_trace = ffn_processor.split_param(param_dict)
        param_name_trace.update(ffn_trace)

        return param_dict, param_name_trace

    def _get_quant_type(self, network):
        """_get_quant_type"""
        if not isinstance(network, Cell):
            raise TypeError(f"Input network should be a Cell, but got: {type(Cell)}.")
        results = {}
        def process(root: Cell, name_prefix):
            """Iterate the whole network and call callback function `process_cell`."""
            if root is None:
                return
            for name, cell in root.name_cells().items():
                full_cell_name = f"{name_prefix}.{name}"
                if not hasattr(cell, "quant_type_dict"):
                    process(cell, full_cell_name)
                    continue
                info = cell.quant_type_dict()
                results.update(info)
        process(network, 'network')
        return results

    def _get_description_file(self, network):
        """
        Obtain the description of quantization type for each parameter in each layer of the network.
        Such as W8A8 or W4A8_DYNAMIC
        """
        quant_types = self._get_quant_type(network)
        # Apply QKV parameter name splitting to match quantization types
        quant_types = QKVParamProcessor(self.network).split_name(quant_types)
        # Apply FFN parameter name splitting to match quantization types
        quant_types = FFNParamProcessor(self.network).split_name(quant_types)

        hf_quant_types = {}
        for key, value in quant_types.items():
            hf_quant_types[self._convert_param_names_to_hf(key)] = value

        param_dict = self.parameters_dict()
        desc_info = dict((key, hf_quant_types.get(key, QuantType.FLOAT.value)) for key in param_dict)
        return desc_info

    @classmethod
    def _convert_param_names_to_hf(cls, param_name):
        """Convert mcore name to huggingface name.
        One parameter may correspond to multiple parameters in huggingface,
        so return a list of names."""
        rules = {
            "model.": "",
            "decoder.layers": "transformer.h",
            "embedding.word_embeddings.": "transformer.word_embeddings.",
            ".pre_mlp_layernorm.": ".post_attention_layernorm.",
            ".linear_q.": ".query.",
            ".linear_k.": ".key.",
            ".linear_v.": ".value.",
            ".linear_proj.": ".dense.",
            ".gating.": ".gate_proj.",
            ".hidden.": ".up_proj.",
            ".linear_fc2.": ".down_proj.",
            "output_layer": "transformer.ln_f"
        }

        for old, new in rules.items():
            param_name = param_name.replace(old, new)

        return param_name
