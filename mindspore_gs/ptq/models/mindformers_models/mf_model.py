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
"""base class of mindformers quant model"""


import os
import time
import json
import mindspore as ms
from mindspore.communication import get_rank
from mindspore.nn.utils import no_init_parameters
from mindspore import load_param_into_net, load_checkpoint
from mindformers import AutoModel, MindFormerConfig, build_context, build_parallel_config
from mindformers.parallel_core.inference.tensor_parallel.layers import (RowParallelLinear,
                                                                        ColumnParallelLinear,
                                                                        MergedColumnParallelLinear,
                                                                        QKVParallelLinear)
from mindformers.parallel_core.inference.tensor_parallel.gemm_layers import (ColumnParallelGroupedLinear,
                                                                             RowParallelGroupedLinear)
from mindspore_gs.ptq.models.base_model_impl import BaseQuantForCausalLMImpl
from mindspore_gs.ptq.models.base_model import BaseQuantForCausalLM
from mindspore_gs.common import logger
from mindspore_gs.ptq import PTQ
from mindspore_gs.ptq.models.distributed_parameter import DistributedParameter
from mindspore_gs.ptq.processor import Processor
from mindspore_gs.ptq.ptq.wrappers.mindformers.mcore_linear_wrapper import McoreLinearInferCell
from mindspore_gs.ptq.models.safetensors_mgr import SafeTensorsMgr


@BaseQuantForCausalLM.reg_model_hub("mindformers")
class MFModel(BaseQuantForCausalLMImpl):
    """MFModel"""
    _model_registry: dict[str, type] = {}

    @staticmethod
    def _reg_model(name, model_clazz):
        cur = MFModel._model_registry.get(name)
        if cur:
            raise RuntimeError(f"Duplicated model reg, name: {name}, already reg class: {cur}, "
                               f"current reg class:{model_clazz}")
        logger.info(f"Register name {name} to model {model_clazz}")
        MFModel._model_registry[name] = model_clazz

    @staticmethod
    def reg_model(alias=None):
        def decorator(cls):
            """decorator"""
            register_key = alias if alias is not None else cls.__name__
            MFModel._reg_model(register_key, cls)
            return cls

        return decorator

    def __init__(self, yaml_path):
        config = MindFormerConfig(yaml_path)
        build_context(config)
        build_parallel_config(config)
        with no_init_parameters():
            self.network = AutoModel.from_config(yaml_path)

        self._original_sf_path = config.load_checkpoint
        if config.load_checkpoint:
            self.network.load_weights(config.load_checkpoint)

    # pylint: disable=arguments-differ
    @classmethod
    def from_pretrained(cls, yaml_path):
        # todo check
        logger.info('Creating mindformers network...', flush=True)
        config = MindFormerConfig(yaml_path)
        if not hasattr(config, 'trainer') or not hasattr(config.trainer, 'model_name'):
            raise ValueError(f"Not contain trainer.model_name in yaml-file: {yaml_path}")
        model_name = config.trainer.model_name
        model_cls = MFModel._model_registry.get(model_name, None)
        if model_cls is None:
            raise ValueError(f"Not supported model_name: {model_name} from yaml: {yaml_path}")
        return model_cls(yaml_path)

    def _original_safetensors_path(self):
        return self._original_sf_path

    def forward(self, input_ids, max_new_tokens=1):
        return self.network.generate(input_ids, do_sample=False, max_new_tokens=max_new_tokens)

    def _network(self):
        return self.network

    def _transformer_layers(self) -> tuple[type]:
        """_transformer_layers"""
        from mindformers.parallel_core.inference.transformer.transformer_layer import TransformerLayer
        return [TransformerLayer]

    def _process_params_dict_before_save(self, param_dict) -> tuple[dict, dict]:
        new_param_dict = {}
        for key, param in param_dict.items():
            if "key_cache" in key or "value_cache" in key or "float_weight" in key:
                continue
            new_param_dict[key] = param
        return new_param_dict, {}

    def _load_weights_to_fake_quant(self, quant_safetensors_path):
        raise NotImplementedError

    def fake_quant(self, ptq_config, layers_policy, quant_safetensors_path: str = ""):
        logger.info("Use ptq algo to fake-quant network and weight")
        ptq = PTQ(config=ptq_config, layer_policies=layers_policy)
        # pylint: disable=protected-access
        ptq._config.experimental = True
        ptq._config.use_fake_quant = True
        transformer_layers = self._transformer_layers()
        _ = [ptq.decoder_layer_types.append(layer) for layer in transformer_layers]
        ptq.fake_quant(self.network)
        self._load_weights_to_fake_quant(quant_safetensors_path)


class MFModelEnableSafeTensors(MFModel):
    """MFModelEnableSafeTensors"""
    def _load_weights_to_fake_quant(self, quant_safetensors_path):
        from .weight_loader import WeightProcessor
        processor = WeightProcessor()
        processor.load_safetensors_shard(quant_safetensors_path, self.network)

    def _process_params_dict_before_save(self, param_dict) -> tuple[dict, dict]:
        """_process_params_dict_before_save"""
        param_dict, param_name_trace = super()._process_params_dict_before_save(param_dict)
        # _del_experts_weight
        experts_dict = {k: v for k, v in param_dict.items()
                        if ".mlp.experts." in k}
        is_fc1_quant = any([".linear_fc1.weight_scale" in k for k in experts_dict.keys()])
        is_fc2_quant = any([".linear_fc2.weight_scale" in k for k in experts_dict.keys()])
        def process(root, name_prefix):
            """Iterate the whole network and call callback function `process_cell`."""
            if root is None:
                return
            for name, cell in root.name_cells().items():
                full_cell_name = f"{name_prefix}.{name}"
                if is_fc1_quant and hasattr(cell, "weight1"):
                    del cell.weight1
                    cell.weight1 = None
                if is_fc2_quant and hasattr(cell, "weight2"):
                    del cell.weight2
                    cell.weight2 = None
                process(cell, full_cell_name)
        process(self.network, 'network')
        return param_dict, param_name_trace

    def _shard_dict(self):
        """_shard_dict"""
        class Collector(Processor):
            """Collector"""
            def __init__(self):
                self.shard_axis = {}
                self.row_linears = ('linear_proj', 'linear_fc2')
                self.col_linears = {'linear_qkv', 'linear_fc1',
                                    'linear_q', 'linear_k', 'linear_v'}

            @staticmethod
            def _transpose_b(linear):
                if isinstance(linear, (RowParallelLinear, ColumnParallelLinear,
                                       QKVParallelLinear, MergedColumnParallelLinear)):
                    return linear.transpose_b
                if isinstance(linear, (ColumnParallelGroupedLinear, RowParallelGroupedLinear)):
                    return False
                raise ValueError(f"Not supported linear: {type(linear)}")

            def _try_append_shard_axis(self, linear, param_name, axis):
                if not hasattr(linear, param_name):
                    return
                self.shard_axis[getattr(linear, param_name).name] = axis

            def process_cell(self, cell_name, cell):
                if 'linear_proj' in cell_name:
                    # pylint: disable=protected-access
                    transpose_b = cell._transpose_b()
                    self._try_append_shard_axis(cell, 'weight', 1 if transpose_b else 0)
                    self._try_append_shard_axis(cell, 'weight_scale', None)
                    self._try_append_shard_axis(cell, 'weight_offset', None)
                    self._try_append_shard_axis(cell, 'input_scale', 0)
                    self._try_append_shard_axis(cell, 'input_offset', 0)
                    self._try_append_shard_axis(cell, 'smooth_scale', 0)
                    self._try_append_shard_axis(cell, 'dequant_scale', None)
                    self._try_append_shard_axis(cell, 'quant_bias', None)
                elif 'linear_fc2' in cell_name:
                    self._try_append_shard_axis(cell, 'weight', 0)
                    self._try_append_shard_axis(cell, 'weight_scale', None)
                    self._try_append_shard_axis(cell, 'weight_offset', None)
                    self._try_append_shard_axis(cell, 'input_scale', 0)
                    self._try_append_shard_axis(cell, 'input_offset', 0)
                    self._try_append_shard_axis(cell, 'smooth_scale', 0)
                    self._try_append_shard_axis(cell, 'dequant_scale', None)
                    self._try_append_shard_axis(cell, 'quant_bias', None)
                elif any(seg in cell_name for seg in ('linear_q', 'linear_k',
                                                      'linear_v', 'linear_qkv')):
                    # pylint: disable=protected-access
                    transpose_b = cell._transpose_b()
                    self._try_append_shard_axis(cell, 'weight', 0 if transpose_b else 1)
                    self._try_append_shard_axis(cell, 'weight_scale', 0)
                    self._try_append_shard_axis(cell, 'weight_offset', 0)
                    self._try_append_shard_axis(cell, 'input_scale', None)
                    self._try_append_shard_axis(cell, 'input_offset', None)
                    self._try_append_shard_axis(cell, 'smooth_scale', None)
                    self._try_append_shard_axis(cell, 'dequant_scale', 0)
                    self._try_append_shard_axis(cell, 'quant_bias', 0)
                elif 'linear_fc1' in cell_name:
                    self._try_append_shard_axis(cell, 'weight', 1)
                    self._try_append_shard_axis(cell, 'weight_scale', 0)
                    self._try_append_shard_axis(cell, 'weight_offset', 0)
                    self._try_append_shard_axis(cell, 'input_scale', None)
                    self._try_append_shard_axis(cell, 'input_offset', None)
                    self._try_append_shard_axis(cell, 'smooth_scale', None)
                    self._try_append_shard_axis(cell, 'dequant_scale', 0)
                    self._try_append_shard_axis(cell, 'quant_bias', 0)
                elif 'output_layer' in cell_name:
                    self._try_append_shard_axis(cell, 'weight', 0)
                elif 'embedding.word_embeddings' in cell_name:
                    self._try_append_shard_axis(cell, 'weight', 0)
                else:
                    pass
                if isinstance(cell, McoreLinearInferCell):
                    return cell, True
                return cell, False

        collector = Collector()
        collector.process(self.network)
        return collector.shard_axis

    def parameters_dict(self, scope="") -> dict[str, DistributedParameter]:
        """parameters_dict"""
        param_dict = self.network.parameters_dict()
        param_dict, param_name_trace = self._process_params_dict_before_save(param_dict)
        shard_info = self._shard_dict()
        dis_param_dict = {}
        for name, param in param_dict.items():
            shard_axis = shard_info.get(name)
            old_name = name
            while shard_axis is None:
                old_name = param_name_trace.get(old_name)
                if old_name is None:
                    break
                shard_axis = shard_info.get(old_name)
            logger.debug("shard axis for ", name, ' is ', shard_axis)
            if shard_axis is None:
                dis_param_dict[name] = DistributedParameter(param)
            else:
                dis_param_dict[name] = DistributedParameter(param, shard_axis)
        return dis_param_dict

    def save_quantized(self, save_path):
        """save_pretrained"""
        sf_mgr = SafeTensorsMgr()
        sf_mgr.save(self._original_sf_path,
                    save_path,
                    self.parameters_dict(),
                    self.get_description_file(self._network()))

    def get_description_file(self, network):
        raise NotImplementedError

class MFModelNotEnableSafeTensors(MFModel):
    """MFModelNotEnableSafeTensors"""
    @staticmethod
    def _find_unique_file(directory, suffix):
        """_find_unique_file"""
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"directory not exist: {directory}")

        matching_files = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and filename.endswith(suffix):
                matching_files.append(file_path)

        if not matching_files:
            raise ValueError(f"not found any 'xxx.{suffix}' file under {directory}")
        if len(matching_files) > 1:
            error_msg = f"found multi 'xxx.{suffix}' file under {directory}: {matching_files}"
            raise ValueError(error_msg)

        return matching_files[0]

    def _concat_route_moe_weight(self, param_dict) -> dict:
        """_concat_route_moe_weight"""
        new_param_dict = {}
        experts_dict = {k: v for k, v in param_dict.items()
                        if ".mlp.experts." in k}
        other_dict = dict(param_dict.items() - experts_dict.items())
        new_param_dict.update(other_dict)

        is_fc1_quant = any([".linear_fc1.weight_scale" in k for k in experts_dict.keys()])
        is_fc2_quant = any([".linear_fc2.weight_scale" in k for k in experts_dict.keys()])

        experts_fc1_dict = {k: v for k, v in experts_dict.items()
                            if ".mlp.experts" in k and ".linear_fc1" in k}
        experts_fc1_dict = self._concat_experts(experts_fc1_dict, is_fc1_quant, "weight1")

        experts_fc2_dict = {k: v for k, v in experts_dict.items()
                            if ".mlp.experts" in k and ".linear_fc2" in k}
        experts_fc2_dict = self._concat_experts(experts_fc2_dict, is_fc2_quant, "weight2")

        new_param_dict.update(experts_fc1_dict)
        new_param_dict.update(experts_fc2_dict)
        return new_param_dict, is_fc1_quant, is_fc2_quant

    def _concat_experts(self, param_dict, is_quant, weight_name):
        """_concat_experts"""
        new_param_dict = {}
        for key, _ in param_dict.items():
            key_split = key.split('.')
            prefix_str = '.'.join(key_split[:6])
            suffix_str = '.'.join(key_split[7:])
            if is_quant:
                new_name = f"{prefix_str}.{suffix_str}"
            else:
                new_name = f"{prefix_str}.{weight_name}"
            if new_name in new_param_dict.keys():
                continue
            experts_dict = {k: v for k, v in param_dict.items()
                            if k.startswith(prefix_str) and k.endswith(suffix_str)}
            num_experts = len(experts_dict.keys())
            value_list = []
            for i in range(num_experts):
                key_ = f"{prefix_str}.{i}.{suffix_str}"
                value_ = experts_dict[key_]
                if key_.endswith('.weight'):
                    value_ = msops.transpose(value_, (1, 0))
                value_ = value_.expand_dims(0)
                value_list.append(value_)
            new_value = msops.cat(tuple(value_list), axis=0)
            new_param_dict[new_name] = Parameter(new_value)
        return new_param_dict

    def _del_experts_weight(self, network, is_fc1_quant, is_fc2_quant):
        """_del_experts_weight"""
        def process(root, name_prefix):
            """Iterate the whole network and call callback function `process_cell`."""
            if root is None:
                return
            for name, cell in root.name_cells().items():
                full_cell_name = f"{name_prefix}.{name}"
                if is_fc1_quant and hasattr(cell, "weight1"):
                    del cell.weight1
                if is_fc2_quant and hasattr(cell, "weight2"):
                    del cell.weight2
                process(cell, full_cell_name)
        process(network, 'network')

    def _process_params_dict_before_load(self, param_dict) -> dict:
        """_process_params_dict_before_load"""
        param_dict, is_fc1_quant, is_fc2_quant = self._concat_route_moe_weight(param_dict)
        self._del_experts_weight(self.network, is_fc1_quant, is_fc2_quant)
        return param_dict

    def _load_weights_to_fake_quant(self, quant_safetensors_path):
        """_load_tp_splited_safetensors"""
        if not quant_safetensors_path:
            return
        try:
            rank_id = get_rank()
        except RuntimeError:
            rank_id = 0
        param_dict_path = os.path.join(quant_safetensors_path, f"rank_{rank_id}")
        param_dict_path = MFModelNotEnableSafeTensors._find_unique_file(param_dict_path, ".safetensors")
        param_dict = load_checkpoint(param_dict_path, format="safetensors")
        new_param_dict = self._process_params_dict_before_load(param_dict)
        param_not_load, ckpt_not_load = load_param_into_net(self.network, new_param_dict)
        logger.info(f"Network has but not in ckpt: {param_not_load}", flush=True)
        logger.info(f"CKPT has but not in network: {ckpt_not_load}", flush=True)

    def parameters_dict(self, scope=""):
        param_dict = self.network.parameters_dict()
        param_dict, _ = self._process_params_dict_before_save(param_dict)
        return param_dict

    def save_quantized(self, save_path):
        """save_pretrained"""
        self._save_safetenors(save_path)
        _ = self._save_desc_json(save_path)

    def _save_safetenors(self, save_path) -> str:
        """_save_safetenors"""
        start = time.time()
        logger.info(f"Saving checkpoint...", flush=True)
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
        logger.info(f"Saving describle json file...", flush=True)
        desc_info = self.get_description_file(self._network())
        save_json_path = os.path.join(save_path, f"quantization_description.json")
        os.makedirs(save_path, exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(desc_info, f, ensure_ascii=False, indent=4)
        logger.info(f'Describle json file saved to {save_json_path}', flush=True)
        logger.info(f'Save describle json cost time is {time.time() - start} s.')
        return save_json_path

    def get_description_file(self, network):
        raise NotImplementedError
