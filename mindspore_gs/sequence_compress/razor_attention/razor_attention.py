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
"""RazorAttention algorithm"""
import os
import json
import gc
import time
from typing import Tuple
import secrets
import numpy as np

from mindspore.nn import Cell
import mindspore
from mindspore import ops, mint
from mindone.transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from mindspore_gs.common import BackendTarget, logger
from mindspore_gs.common.utils import value_check
from mindspore_gs.comp_algo import CompAlgo
from mindspore_gs.sequence_compress.processor import Processor
from mindspore_gs.sequence_compress.razor_attention.ra_config import RAMode, RAConfig

DUMMY_INPUT_LENGTH = 2500
REPET_TIMES = 4


class RazorAttention(CompAlgo):
    """RazorAttention algorithm"""

    def __init__(self, config=None):
        super().__init__()
        if config is not None:
            if not isinstance(config, RAConfig):
                raise TypeError(f'Shall init RazorAttention with RAConfig, bug got {type(config)}')
            self._config = config
        else:
            self._config = RAConfig()
        RazorAttention._ra_config_check(self._config)
        if self._config.backend != BackendTarget.ASCEND:
            raise ValueError("RazorAttention only support ASCEND as BackendTarget now, "
                             f"but got {self._config.backend}.")
        mode = self._config.mode
        self._is_deploy = mode == RAMode.DEPLOY
        self.attention_layers: list[Cell] = []
        self.attention_layer_types: list = []
        self._load_mindone_plugin()
        self._origin_fas = ops.flash_attention_score
        self._current_layer_idx = 0
        self._echo_head = {}
        self._induction_head = {}
        self._num_layers = 0

    def _hooked_flash_attention_score(self, *args, **kwargs):
        """replace fa score ops using hook."""
        logger.info("=== Intercepted flash_attention_score call ===")
        logger.info(f"saving qk for layer {self._current_layer_idx}......")
        q, k = args[0], args[1]

        self._process_single_layer(self._current_layer_idx, q, k)
        self._current_layer_idx += 1
        out = self._origin_fas(*args, **kwargs)
        logger.info("=== Output shape:", out.shape if hasattr(out, 'shape') else out)
        del args, kwargs, q, k
        gc.collect()
        mindspore.runtime.empty_cache()
        return out

    def _load_mindone_plugin(self):
        """load mindone model"""
        self.attention_layer_types.append(Qwen3Attention)

    @staticmethod
    def _ra_config_check(config):
        """_ptq_config_check"""
        if (config.echo_head_ratio < 0 and config.echo_head_ratio > 1) or \
            (config.induction_head_ratio < 0 and config.induction_head_ratio > 1):
            raise ValueError("echo_head_ratio and induction_head_ratio must be >=0 or <=1.")
        if not config.retrieval_head_path.endswith(".json"):
            raise ValueError("please set path to json file of retrieval_head_path, ex. /path/to/retrieval.json")

    def _get_attention_layers(self, network: Cell):
        """
        Get attention layers from network.

        Args:
            network (nn.Cell): Network to get attention layers.

        Returns:
            A list of tuples (cell_name, `Cell`) as attention layers of network.
        """
        value_check('network', network, Cell)

        class NetworkWalker(Processor):
            """NetworkWalker"""
            def __init__(self, attention_layer_types_):
                self.layers = []
                self._attention_layer_types = attention_layer_types_

            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                """"process_cell"""
                if isinstance(cell, self._attention_layer_types):
                    self.layers.append((cell_name, cell))
                    return cell, True
                return cell, False

        walker = NetworkWalker(tuple(self.attention_layer_types))
        walker.process(network)
        if walker.layers:
            self.attention_layers = walker.layers
            return
        self.attention_layers = [("root", network)]
        logger.warning(
            f"No attention layer found in network. Visible attention layer types: {self.attention_layer_types}.")

    # pylint: disable=arguments-differ
    def apply(self, network: Cell, is_saved: bool) -> dict:
        """
        Apply compression or deployment transformation to the given network.

        Args:
            network (Cell): The model network to process.
            is_saved (bool): Whether to save the retrieval head.

        Returns:
            dict: The retrieval head.

        Raises:
            RuntimeError: If network is invalid or processing fails.
        """
        # get layer info and replace fa score using hook
        self._get_attention_layers(network)
        ops.flash_attention_score = self._hooked_flash_attention_score
        logger.info("Start replace network...")
        start_time = time.time()
        logger.info(f"Replace network cost time: {time.time() - start_time:.2f}s")

        # initialize variables
        self._current_layer_idx = 0
        self._echo_head = {}
        self._induction_head = {}
        self._num_layers = len(self.attention_layers) if self.attention_layers else 1

        # input setting
        logger.info("Using random input to eval (layer-by-layer search)...")
        start_time = time.time()

        secrets_gen = secrets.SystemRandom()
        prompt_token_ids = [
            secrets_gen.randint(10000, 10000 + DUMMY_INPUT_LENGTH)
            for _ in range(DUMMY_INPUT_LENGTH)
        ] * REPET_TIMES

        bos_token_id = getattr(getattr(network, "config", None), "bos_token_id", None)
        if bos_token_id is not None:
            prompt_token_ids.insert(0, bos_token_id)

        # generate
        input_ids = mindspore.tensor([prompt_token_ids], mindspore.int32)  # shape [1, seq_len]
        model_inputs = {"input_ids": input_ids}
        _ = network.generate(
                            **model_inputs,
                            max_new_tokens=1,
                            do_sample=False,
                            use_cache=True,
                        )

        logger.info(f"Generate cost time: {time.time() - start_time:.2f}s")

        # search for retrieval head
        logger.info("Start finalize compress head...")
        start_time = time.time()
        retrieval_heads = self._finalize_compress_head(is_saved)
        logger.info(
            f"Finalize process cost time: {time.time() - start_time:.2f}s, "
            f"the head info saved in {self._config.retrieval_head_path}"
        )

        return retrieval_heads

    def _repeat_kv(self, x, rep):
        """Expand key/value along num_head dimension."""
        if rep == 1:
            return x
        bs, num_groups, seq_length, head_dim = x.shape
        x = x.reshape((bs, num_groups, 1, seq_length * head_dim))
        x = x.tile((1, 1, rep, 1))
        x = x.reshape((bs, num_groups * rep, seq_length, head_dim))
        return x

    def _generate_attn_mask(self, bsz, q_len, kv_len):
        """Upper-triangular causal mask."""
        mask = mint.ones((bsz, 1, q_len, kv_len), dtype=mindspore.uint8)
        mask = mint.triu(mask, diagonal=1)
        return mask

    def _attn_dot_product(self, query, key):
        """Compute attention probabilities with causal mask."""
        bs, _, seq_len, head_size = query.shape
        scale_factor = 1 / np.sqrt(head_size)
        score = ops.bmm(query, ops.transpose(key, (0, 1, 3, 2)))  # (B, N, S, S)
        score = mint.mul(score, scale_factor)
        attn_mask = self._generate_attn_mask(bs, seq_len, seq_len)
        causal_mask = attn_mask * -1e5
        masked_input = score + causal_mask
        probs = mint.nn.functional.softmax(masked_input, dim=-1)
        return probs

    def _get_score_mask(self, seq_len):
        """Compute echo and induction masks."""
        echo_mask = mint.zeros((seq_len, seq_len), dtype=mindspore.uint8)
        induction_mask = mint.zeros((seq_len, seq_len), dtype=mindspore.uint8)
        for i in range(1, seq_len):
            if i // DUMMY_INPUT_LENGTH == 0:
                continue
            for j in range(i % DUMMY_INPUT_LENGTH, i, DUMMY_INPUT_LENGTH):
                echo_mask[i, j] = 1
                induction_mask[i, j+1] = 1
        return echo_mask, induction_mask

    def _calc_score(self, head_i_score):
        """Compute echo/induction score for one head."""
        seq_len = head_i_score.shape[-2]
        echo_mask, induction_mask = self._get_score_mask(seq_len)
        echo_score = float(ops.mean(head_i_score * echo_mask))
        induction_score = float(ops.mean(head_i_score * induction_mask))
        return echo_score, induction_score

    def _max_every_group(self, data, n):
        """Take max in every kv-group."""
        max_values = [max(data[i:i+n]) for i in range(0, len(data), n)]
        return max_values

    def _process_single_layer(self, layer_idx, q, k):
        """
        Process single layer and offloading qk attn_p after calculation.
        
        This function computes attention scores for echo and induction heads,
        then stores the results for later head selection.
        
        Args:
            layer_idx (int): Layer index in the transformer model
            q (Tensor): Query tensor with shape [batch_size, seq_len, num_heads, head_dim]
            k (Tensor): Key tensor with shape [batch_size, seq_len, num_kv_heads, head_dim]
        """
        logger.info(f"Start process layer {layer_idx} echo head and induction head...")
        start_time = time.time()

        _, seq_len, num_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k.shape

        kv_groups = num_heads // num_kv_heads
        q = ops.transpose(q.reshape(-1, seq_len, num_heads, head_dim), (0, 2, 1, 3))
        k = ops.transpose(k.reshape(-1, seq_len, num_kv_heads, head_dim), (0, 2, 1, 3))
        key_states = self._repeat_kv(k, kv_groups)
        attn_p = self._attn_dot_product(q, key_states)
        echo_scores, induction_scores = [], []
        for h in range(num_heads):
            echo_score, induction_score = self._calc_score(attn_p[:, h, :, :].squeeze(0))
            echo_scores.append(echo_score)
            induction_scores.append(induction_score)
        echo_scores = self._max_every_group(echo_scores, kv_groups)
        induction_scores = self._max_every_group(induction_scores, kv_groups)
        # save to the dict
        self._echo_head[layer_idx] = echo_scores
        self._induction_head[layer_idx] = induction_scores
        # per layer offloading
        del q, k, key_states, attn_p
        gc.collect()
        mindspore.runtime.empty_cache()

        logger.info(f"End of processing layer {layer_idx} echo head and induction head, "
                    f"cost time is {time.time() - start_time:.2f}s")
        used_mem = mindspore.runtime.memory_allocated()
        logger.info(f"[Layer {layer_idx}] used memory: {used_mem / 1024**2:.2f} MB")


    # pylint: disable=W0212
    def _finalize_compress_head(self, is_saved):
        """
        Combine all echo/induction scores for the whole network and select retrieval heads.
        
        This function aggregates attention scores from all layers, selects top heads
        based on configured ratios, and optionally saves the results to a file.
        
        Args:
            is_saved (bool): Whether to save the retrieval head information to file
            
        Returns:
            dict: Dictionary containing selected retrieval heads for each layer
        """
        logger.info("Select top heads...")
        selected_echo_head = self._select_top_heads(self._echo_head,
                                                    self._config.echo_head_ratio)
        select_induction_head = self._select_top_heads(self._induction_head,
                                                       self._config.induction_head_ratio)
        logger.info("Remove empty list from heads info...")
        selected_echo_head = self._remove_empty_list_keys(selected_echo_head)
        select_induction_head = self._remove_empty_list_keys(select_induction_head)
        logger.info("Concat echo heads and induction heads...")
        retrieval_heads = self._get_retrieval_heads(selected_echo_head, select_induction_head)

        logger.info("saving head info...")
        if is_saved:
            dir_path = '/'.join(self._config.retrieval_head_path.split('/')[:-1])
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(self._config.retrieval_head_path, 'w+', encoding='utf8') as file_path:
                json.dump(retrieval_heads, file_path, indent=3)
        return retrieval_heads

    def _get_retrieval_heads(self, echo_head, induction_head):
        """_get_retrieval_heads"""
        if not echo_head:
            return induction_head
        for key, val in echo_head.items():
            if key in induction_head.keys():
                induction_head[key] = list(set(induction_head[key]) | set(val))
            else:
                induction_head[key] = val
            return induction_head

    def _remove_empty_list_keys(self, dictionary):
        """_remove_empty_list_keys"""
        dictionary = {k: v for k, v in dictionary.items() if v != []}
        return dictionary

    def _select_top_heads(self, data, ratio):
        """_select_top_heads"""
        all_values = [
            value
            for key in data
            for value in data[key]
        ]
        sorted_values = sorted(all_values, reverse=True)
        percent_index = round(len(sorted_values) * ratio)
        percent_values = sorted_values[:percent_index]
        result = {}
        for key in data:
            percent_index_in_original_list = [i for i, value in enumerate(data[key]) if value in percent_values]
            result[key] = percent_index_in_original_list
        return result
