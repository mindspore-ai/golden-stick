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
"""FA3 algorithm"""
import os
from typing import Optional, Tuple
import time
import warnings
import copy
from dataclasses import dataclass
import gc
import tqdm

from mindspore import get_context, PYNATIVE_MODE

from mindspore import mint
from mindspore import ops
from mindspore import dtype as msdtype
from mindspore.communication import get_rank
from mindspore.nn import Cell
from mindspore.ops import operations as P

from mindspore_gs.common import BackendTarget
from mindspore_gs.common import logger
from mindspore_gs.common.utils import value_check, offload_network
from mindspore_gs.comp_algo import CompAlgo
from mindspore_gs.common.json_cache import JSONCache
from mindspore_gs.ptq.network_helpers import NetworkHelper
from mindspore_gs.ptq.ptq.quant import InputCatcher
from mindspore_gs.ptq.basic_functions.processor import Processor

from mindformers.parallel_core.inference.utils import get_tp_world_size
from research.deepseek3.deepseek3_config import DeepseekV3Config
from research.deepseek3.deepseek3_model_infer import MLAPagedAttentionMgr, MLAInferAttention
from research.deepseek3.infer.layers import ColumnParallelLinear

@dataclass
class FA3Config:
    export_params_path: str = ''
    backend: BackendTarget = BackendTarget.ASCEND
    dsk_config: DeepseekV3Config = None

class FA3(CompAlgo):
    """
    Class for fa3 calibration algorithm.
    Please note that the Flash Attention 3
    (FA3) calibration algorithm is currently a demo feature.
    """
    def __init__(self, config=None):
        super().__init__()
        if config is not None:
            if not isinstance(config, FA3Config):
                raise TypeError(f'Shall init with FA3Config, bug got {type(config)}')
            self._config = config
        else:
            raise TypeError(f'Shall init with FA3Config, bug got {type(config)}')
        if self._config.backend != BackendTarget.ASCEND:
            raise ValueError("FA3 only support ASCEND as BackendTarget now, "
                             f"but got {self._config.backend}.")
        self.tensor_parallel_group_size = get_tp_world_size()
        self.n_local_heads = self._config.dsk_config.num_heads // self.tensor_parallel_group_size
        self.kv_lora_rank = self._config.dsk_config.kv_lora_rank
        self.qk_rope_head_dim = self._config.dsk_config.qk_rope_head_dim
        self.qk_nope_head_dim = self._config.dsk_config.qk_nope_head_dim
        self.hooked_name = ["attention.infer_attention.paged_attention_mgr", "attention.infer_attention", \
                            "attention.lkv2kv_k_nope"]
        self.decoder_layers: list[Cell] = []
        self.decoder_layer_types: list = []
        self.absorb_mm_inputs = []
        self.key_samples = []
        self.absorb_linear = None
        self.modules = []
        self.qabsorb_matmul = P.BatchMatMul()
        self._load_mindformers_plugin()
        if self._config.export_params_path:
            cache_file_path = os.path.join(self._config.export_params_path, f'rank_{get_rank()}', \
                                           'perhead.json')
        else:
            cache_file_path = ''
        self.cache: Optional[JSONCache] = JSONCache(cache_file_path)

    def apply(self, network, **kwargs):
        pass

    def _load_mindformers_plugin(self):
        from research.deepseek3.deepseek3_model_infer import DeepseekV3DecodeLayer
        self.decoder_layer_types.append(DeepseekV3DecodeLayer)

    def _get_first_layer_input(self, network: Cell, network_helper: NetworkHelper = None, ds=None):
        """get first layer input"""
        catcher = InputCatcher()
        catcher.patch(self.decoder_layers[0][1])
        if not ds:
            raise ValueError("PTQ need dataset to calibrate, please provide dataset.")
        total_count = ds.get_dataset_size()
        data_count = 1
        for _, ds_item in enumerate(ds.create_dict_iterator()):
            logger.info(f"Calibrating: dataset count: {data_count}/{total_count}")
            input_ids = ds_item['input_ids'].asnumpy()
            try:
                network_helper.generate(network, input_ids, do_sample=False, max_new_tokens=1)
            except GeneratorExit:
                if hasattr(network, "block_mgr") and network.block_mgr:
                    network.block_mgr.clear_cache()
            data_count += 1
        catcher.recover()
        offload_network(network)
        return catcher, network

    def _get_decoder_layers(self, network: Cell):
        """
        Get decoder layers from network.

        Args:
            network (nn.Cell): Network to get decoder layers.

        Returns:
            A list of tuples (cell_name, `Cell`) as decoder layers of network.
        """
        value_check('network', network, Cell)

        class NetworkWalker(Processor):
            def __init__(self, decoder_layer_types_):
                self.layers = []
                self._decoder_layer_types = decoder_layer_types_

            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                if isinstance(cell, self._decoder_layer_types):
                    self.layers.append((cell_name, cell))
                    return cell, True
                return cell, False

        walker = NetworkWalker(tuple(self.decoder_layer_types))
        walker.process(network)
        if walker.layers:
            self.decoder_layers = walker.layers
            return
        self.decoder_layers = [("network", network)]
        logger.warning(
            f"No decoder layer found in network. Visible decoder layer types: {self.decoder_layer_types}, "
            "please modify PTQ.decoder_layer_types before invoking apply method. If not, PTQ will take lots of memory.")

    def find_attribute(self, layer, attr_name):
        attrs = attr_name.split('.')
        current_layer = layer
        for attr in attrs:
            if hasattr(current_layer, attr):
                current_layer = getattr(current_layer, attr)
            else:
                print("Attribute '{attr}' not found in the current layers.")
                return None
        return current_layer

    def add_hook(self, layer):
        """Add hook for fa3 calibration."""
        self.key_samples.clear()
        self.absorb_mm_inputs.clear()
        class HookMLAPagedAttention(MLAPagedAttentionMgr):
            """To obtain intermediate activation values, the auxiliary class of hook algorithm."""
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.kv_lora_rank = 0
                self.qk_rope_head_dim = 0
            def construct(self, key, slot_mapping, key_cache=None):
                """The forward compute of single cache for Paged Attention."""
                #split for q_nope
                k_nope, _ = mint.split(key, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
                self.key_samples.append(ops.squeeze(k_nope))
                if self.npu_mem_size > 0:
                    return self.reshape_and_cache(key, None, self.key_cache, None, slot_mapping)
                return self.reshape_and_cache(key, None, key_cache, None, slot_mapping)

        class HookMLAInferAttention(MLAInferAttention):
            """To obtain intermediate activation values, the auxiliary class of hook algorithm."""
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.qk_nope_head_dim = 0
                self.qk_rope_head_dim = 0
            def construct(self, query, key, value, batch_valid_length, block_tables,
                          attn_mask=None, alibi_mask=None, q_seq_lens=None, key_cache=None):
                """ Forward process of the MLA Infer Attention Cell """
                q_nope, _ = mint.split(query, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
                self.absorb_mm_inputs.append(ops.squeeze(q_nope))
                if self.is_first_iteration:
                    return self._prefill_attention(query, key, value, attn_mask, alibi_mask, q_seq_lens,
                                                   batch_valid_length)
                return self._incre_attention(query, batch_valid_length, block_tables,
                                             attn_mask, q_seq_lens, key_cache=key_cache)

        self.modules.clear()
        for name in self.hooked_name:
            module = self.find_attribute(layer, name)
            self.modules.append(module)
            if module is None:
                warn_str = f"Not found {name} of layer to quant."
                warnings.warn(warn_str, RuntimeError)
                continue

            if isinstance(module, MLAPagedAttentionMgr):
                module.__class__ = HookMLAPagedAttention
                module.__class__.kv_lora_rank = self.kv_lora_rank
                module.__class__.qk_rope_head_dim = self.qk_rope_head_dim
                module.key_samples = self.key_samples
            elif isinstance(module, MLAInferAttention):
                module.__class__ = HookMLAInferAttention
                module.__class__.qk_nope_head_dim = self.qk_nope_head_dim
                module.__class__.qk_rope_head_dim = self.qk_rope_head_dim
                module.absorb_mm_inputs = self.absorb_mm_inputs
            elif isinstance(module, ColumnParallelLinear):
                self.absorb_linear = module
            else:
                raise RuntimeError(f"Unsupported MLA type for hook: {type(self.layer.matmul)}")

    def remove_hook(self):
        """Remove hook for fa3 calibration."""
        for module in self.modules:
            # for perhead
            if isinstance(module, MLAPagedAttentionMgr):
                module.__class__ = MLAPagedAttentionMgr
                if hasattr(module, 'key_samples'):
                    del module.key_samples
            elif isinstance(module, MLAInferAttention):
                module.__class__ = MLAInferAttention
                if hasattr(module, 'absorb_mm_inputs'):
                    del module.absorb_mm_inputs
            elif isinstance(module, ColumnParallelLinear):
                continue
            else:
                raise RuntimeError(f"Unsupported MLA type for hook: {type(self.layer.matmul)}")

    def params_key(self, layer_name):
        kv_scales_name = layer_name + '.kv_scales'
        kv_offsets_name = layer_name + '.kv_offsets'
        q_scales_name = layer_name + '.q_scales'
        q_offsets_name = layer_name + '.q_offsets'
        return kv_scales_name, kv_offsets_name, q_scales_name, q_offsets_name

    def compute_per_head_quantization_params(self, origin_tensor, have_perhead_dim=True, bit_width=8, symmetric=True):
        """calculate params for fa3 calibration."""
        origin_tensor = ops.cast(origin_tensor, msdtype.float32)

        reduce_max = ops.ReduceMax(keep_dims=True)
        reduce_min = ops.ReduceMin(keep_dims=True)

        if have_perhead_dim:
            max_vals = reduce_max(origin_tensor, axis=(0, 2))
            min_vals = reduce_min(origin_tensor, axis=(0, 2))
        else:
            max_vals = reduce_max(origin_tensor)
            min_vals = reduce_min(origin_tensor)

        if symmetric:
            range_vals = ops.maximum(max_vals.abs(), min_vals.abs())
            scales = range_vals / (2 ** (bit_width - 1) - 1)
            offsets = ops.zeros_like(scales)
        else:
            range_vals = max_vals - min_vals
            scales = range_vals / (2 ** bit_width - 1)
            offsets = min_vals

        return scales, offsets

    # calculate scale and offset for each head and allgather them
    def calculate_scale_and_offset(self, layer_name):
        """calculate params for fa3 calibration."""
        q_absorb = self.absorb_linear.weight.view(self.n_local_heads, self.qk_nope_head_dim, self.kv_lora_rank)
        self.new_q_nope = []
        for absorb_mm_input in self.absorb_mm_inputs:
            new_q_nope_ = self.qabsorb_matmul(absorb_mm_input.transpose(1, 0, 2), q_absorb).transpose(1, 0, 2)
            self.new_q_nope.append(new_q_nope_)
        cat_key_samples = ops.cat(tuple(self.key_samples), axis=0)
        cat_new_q_nope = ops.cat(tuple(self.new_q_nope), axis=0)
        kv_scale, kv_offset = self.compute_per_head_quantization_params(cat_key_samples, False)
        q_scale, q_offset = self.compute_per_head_quantization_params(cat_new_q_nope)
        allgather = ops.AllGather()
        kv_scales_all = allgather(kv_scale.transpose(1, 0)).transpose(1, 0)
        kv_offsets_all = allgather(kv_offset.transpose(1, 0)).transpose(1, 0)
        q_scales_all = allgather(q_scale.transpose(1, 0, 2)).transpose(1, 0, 2)
        q_offsets_all = allgather(q_offset.transpose(1, 0, 2)).transpose(1, 0, 2)
        kv_scales_name, kv_offsets_name, q_scales_name, q_offsets_name = self.params_key(layer_name)

        self.cache.put(kv_scales_name, kv_scales_all.asnumpy())
        self.cache.put(kv_offsets_name, kv_offsets_all.asnumpy())
        self.cache.put(q_scales_name, q_scales_all.asnumpy())
        self.cache.put(q_offsets_name, q_offsets_all.asnumpy())

    def observe(self, network: Cell,
                network_helper: NetworkHelper = None,
                datasets=None, **kwargs):
        """main function for fa3 calibration."""
        self._get_decoder_layers(network)
        if get_context("mode") != PYNATIVE_MODE:
            raise ValueError("In FA3 quantize phase, please set mode=PYNATIVE_MODE.")
        if not network_helper:
            raise ValueError("Please provide network_helper when PTQ in observe phase.")
        if not datasets:
            raise ValueError("please provide dataset when use FA3 quant to quantize network.")
        logger.info(f"Visible decoder layer types: {self.decoder_layer_types}. If decoder layer type of target network "
                    "not in list, please modify PTQ.decoder_layer_types before invoking apply method.")
        logger.info("Analysis network structure.")
        start_time = time.time()
        logger.info(f"Catching inputs for first decoder layer with {datasets.get_dataset_size()} datasets samples.")
        catcher, network = self._get_first_layer_input(network, network_helper, datasets)
        all_args = catcher.args
        all_kwargs = catcher.kwargs

        for i in tqdm.tqdm(range(len(self.decoder_layers)), desc="Running perhead obersving..."):
            logger.info(f"Observe {i}th decoder layer.")
            layer_name, layer = self.decoder_layers[i]
            kv_scales_name, kv_offsets_name, q_scales_name, q_offsets_name = self.params_key(layer_name)
            kv_scales = self.cache.get(kv_scales_name)
            kv_offsets = self.cache.get(kv_offsets_name)
            q_scales = self.cache.get(q_scales_name)
            q_offsets = self.cache.get(q_offsets_name)
            if kv_scales is not None and kv_offsets is not None and q_scales is not None and q_offsets is not None:
                offload_network(layer)
                gc.collect()
                continue

            cur_args, cur_kwargs = copy.deepcopy(all_args), copy.deepcopy(all_kwargs)
            for index, (args, kwargs) in enumerate(zip(cur_args, cur_kwargs)):
                output = layer(*args, **kwargs)
                if len(self.decoder_layers) > 1:
                    all_args[index][0] = output[0] if isinstance(output, tuple) else output
            self.add_hook(layer)
            for args, kwargs in zip(cur_args, cur_kwargs):
                output = layer(*args, **kwargs)
            self.remove_hook()
            logger.info(f"{i}th layer output refresh time cost {time.time() - start_time}")
            self.calculate_scale_and_offset(layer_name)
            offload_network(layer)
            gc.collect()
