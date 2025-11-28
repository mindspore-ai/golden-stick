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
"""deepseek v3 quant model"""

import os
import time
import json

from mindspore.nn.cell import Cell
from mindspore.communication import get_rank

import mindspore as ms
from mindspore_gs.ptq.models.mindformers_models.mf_model import MFModel, MFModelEnableSafeTensors
from mindspore_gs.ptq.models.mindformers_models.param_processor import (
    MLAParamProcessor, MoeParamProcessor, FFNParamProcessor)
from mindspore_gs.ptq.utils import QuantType
from mindspore_gs.common import logger
from mindspore_gs.common import BackendTarget


@MFModel.reg_model('deepseek_v3')
class DeepSeekV3(MFModelEnableSafeTensors):
    """DeepSeekV3"""

    def __init__(self, yaml_path):
        super().__init__(yaml_path)
        self.__parameter_dict = None

    def _after_network_load_weights(self):
        def process(root, name_prefix):
            """Iterate the whole network and call callback function `process_cell`."""
            if root is None:
                return
            for name, cell in root.name_cells().items():
                full_cell_name = f"{name_prefix}.{name}"
                if hasattr(cell, "weight1"):
                    del cell.weight1
                if hasattr(cell, "weight2"):
                    del cell.weight2
                process(cell, full_cell_name)
        process(self.network, 'network')

    def _convert_name(self, param_dict):
        """Convert mcore name to huggingface name.
        One parameter may correspond to multiple parameters in huggingface,
        so return a list of names."""
        new_param_dict = {}
        for key, value in param_dict.items():
            key = key.replace('model.', '')
            key = key.replace('decoder.layers.', 'model.layers.')
            key = key.replace('.self_attention.', '.self_attn.')
            key = key.replace('embedding.word_embeddings.', 'model.embed_tokens.')
            key = key.replace('decoder.final_layernorm.', 'model.norm.')
            key = key.replace('output_layer.', 'lm_head.')
            key = key.replace('.pre_mlp_layernorm.', '.post_attention_layernorm.')
            key = key.replace('.linear_q_down_proj.', '.q_a_proj.')
            key = key.replace('.linear_kv_down_proj.', '.kv_a_proj_with_mqa.')
            key = key.replace('.q_layernorm.', '.q_a_layernorm.')
            key = key.replace('.linear_q_up_proj.', '.q_b_proj.')
            key = key.replace('.kv_layernorm.', '.kv_a_layernorm.')
            key = key.replace('.linear_kv_up_proj.', '.kv_b_proj.')
            key = key.replace('.linear_proj.', '.o_proj.')
            key = key.replace('.gating.', '.gate_proj.')
            key = key.replace('.hidden.', '.up_proj.')
            key = key.replace('.linear_fc2.', '.down_proj.')
            key = key.replace('.router.', '.gate.')
            key = key.replace('.expert_bias', '.e_score_correction_bias')
            new_param_dict[key] = value
        return new_param_dict

    def _process_params_dict_before_save(self, param_dict) -> tuple[dict, dict]:
        param_dict, param_name_trace = super()._process_params_dict_before_save(param_dict)

        # Apply QKV split
        qkv_processor = MLAParamProcessor(self.network)
        param_dict, qkv_trace = qkv_processor.split_param(param_dict)
        param_name_trace.update(qkv_trace)

        # Apply MoE split
        moe_processor = MoeParamProcessor(self.network)
        param_dict, moe_trace = moe_processor.split_param(param_dict)
        param_name_trace.update(moe_trace)

        # Apply FFN split
        ffn_processor = FFNParamProcessor(self.network)
        param_dict, ffn_trace = ffn_processor.split_param(param_dict)
        param_name_trace.update(ffn_trace)

        return param_dict, param_name_trace

    def parameters_dict(self, scope="") -> dict:
        """parameters_dict"""
        # FIXME: Currently, calling this method will release the original network to save memory.
        if self.__parameter_dict is not None:
            return self.__parameter_dict
        param_dict = self.network.parameters_dict()
        del self.network.model
        param_dict, _ = self._process_params_dict_before_save(param_dict)
        param_dict = self._convert_name(param_dict)
        self.__parameter_dict = param_dict
        return param_dict

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
        Such as W8A8 or W8A8_DYNAMIC.
        """
        quant_types = self._get_quant_type(network)
        quant_types = MLAParamProcessor(self.network).split_name(quant_types)
        quant_types = MoeParamProcessor(self.network).split_name(quant_types)
        quant_types = FFNParamProcessor(self.network).split_name(quant_types)
        quant_types = self._convert_name(quant_types)
        param_dict = self.parameters_dict()
        desc_info = dict((key, quant_types.get(key, QuantType.FLOAT.value)) for key in param_dict)
        return desc_info

    def save_quantized(self, save_path, backend=BackendTarget.ASCEND):
        """save_pretrained"""
        if backend != BackendTarget.ASCEND:
            raise ValueError("Only support save quantized model for ASCEND backend "
                             "when not enable SafeTensors format in mindformers models.")
        _ = self._save_desc_json(save_path)
        self._save_safetenors(save_path)

    def _save_safetenors(self, save_path) -> str:
        """_save_safetenors"""
        start = time.time()
        logger.info("Saving checkpoint...", flush=True)
        param_dict = self.parameters_dict()
        try:
            rank_id = get_rank()
        except RuntimeError:
            rank_id = 0
        save_path = os.path.join(save_path, f"rank_{rank_id}")
        os.makedirs(save_path, exist_ok=True)
        final_path = os.path.join(save_path, 'quant')
        ms.save_checkpoint(param_dict, final_path, format="safetensors")
        logger.info(f'Checkpoint saved to {final_path}', flush=True)
        logger.info(f'Save checkpoint cost time is {time.time() - start} s.')

    def _save_desc_json(self, save_path) -> str:
        """_save_desc_json"""
        start = time.time()
        logger.info("Saving describle json file...", flush=True)
        desc_info = self._get_description_file(self._network())
        save_json_path = os.path.join(save_path, "quantization_description.json")
        os.makedirs(save_path, exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(desc_info, f, ensure_ascii=False, indent=4)
        logger.info(f'Describle json file saved to {save_json_path}', flush=True)
        logger.info(f'Save describle json cost time is {time.time() - start} s.')
        return save_json_path

    @classmethod
    def _convert_param_names_to_hf(cls, param_name):
        """Convert the parameter to huggingface format.
        Args:
            param_name: The parameter name to convert to huggingface format.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        # Have not implemented yet
        raise NotImplementedError
