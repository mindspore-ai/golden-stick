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
import time
from functools import partial
from typing import Tuple
import random
import tqdm

from mindspore.nn import Cell
from vllm_mindspore.attention import Attention

from mindspore_gs.common import BackendTarget
from mindspore_gs.common import logger
from mindspore_gs.common.utils import value_check
from mindspore_gs.comp_algo import CompAlgo

from mindspore_gs.long_context_compress.processor import (
    Processor,
    network_replace)
from mindspore_gs.long_context_compress.razor_attention.ra_config import RAMode, RAConfig
from mindspore_gs.long_context_compress.razor_attention.wrappers.vllm_mindspore import DeployRACompressCell, RACompressCell

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
        self._load_vllm_ms_plugin()

    def _load_vllm_ms_plugin(self):
        """
        Load quant cells, layer policy for vllm_mindspore as plugin so that `RazorAttention` can support network from
        vllm_mindspore. Invoking this static method before creating `RazorAttention`.
        """
        # pylint: disable=unused-import
        from vllm_mindspore.model_executor.models.llama import LlamaAttention
        self.attention_layer_types.append(LlamaAttention)

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
            def __init__(self, attention_layer_types_):
                self.layers = []
                self._attention_layer_types = attention_layer_types_

            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
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
    def apply(self, net_helper) -> Cell:
        """

        Args:

        Returns:

        Raises:
        """
        network = net_helper.get_network()
        self._get_attention_layers(network)
        if self._config.mode == RAMode.DEPLOY:
            return self._get_deploy_network(network)

        logger.info("Start replace network...")
        start_time = time.time()
        network = self._get_compress_network(network)
        logger.info(f"Replace network cost time is {time.time() - start_time}")

        logger.info("Using random input to eval...")
        start_time = time.time()
        random.seed(2025)
        prompt_token_ids = [random.randint(10000, 10000+DUMMY_INPUT_LENGTH) \
                            for _ in range(DUMMY_INPUT_LENGTH)]*REPET_TIMES
        bos_token_id = net_helper.llm.llm_engine.get_model_config().hf_config.bos_token_id
        prompt_token_ids.insert(0, bos_token_id)
        _ = net_helper.generate(prompt_token_ids=prompt_token_ids)
        logger.info(f"Generate cost time is {time.time() - start_time}")

        logger.info("Start search compress head...")
        start_time = time.time()
        self._get_compress_head(network)
        logger.info(f"Search process cost time is {time.time() - start_time}, \
                    the head info saved in {self._config.retrieval_head_path}")
        return network

    def _get_deploy_network(self, network):
        """_get_deploy_network"""
        with open(self._config.retrieval_head_path, 'r', encoding='utf-8') as f:
            head_dict = json.load(f)
        for i in tqdm.tqdm(range(len(self.attention_layers)), desc="Running RazorAttention Deploy..."):
            # TODO 更新config中的echo_head和induction_head
            _, layer = self.attention_layers[i]
            retrieval_head = head_dict.get(str(i), [])
            attention_deploy_creator = partial(DeployRACompressCell,
                                               cfg=self._config,
                                               retrieval_head=retrieval_head)
            network_replace(layer, Attention, DeployRACompressCell, attention_deploy_creator, [])
            network.update_parameters_name()
        return network

    def _get_compress_network(self, network):
        """_get_compress_network"""
        for i in tqdm.tqdm(range(len(self.attention_layers)), desc="Running RazorAttention..."):
            _, layer = self.attention_layers[i]
            attention_compress_creator = partial(RACompressCell, cfg=self._config)
            network_replace(layer, Attention, RACompressCell, attention_compress_creator, [])
            network.update_parameters_name()
        return network

    # pylint: disable=W0212
    def _get_compress_head(self, network):
        """_get_compress_head"""
        echo_head = {}
        induction_head = {}
        self._get_attention_layers(network)
        for i, layer in enumerate(self.attention_layers):
            logger.info(f"Start get layer {i} echo head and induction head...")
            start_time = time.time()
            layer[-1].attn.process()
            echo_head[i] = layer[-1].attn._echo_score
            induction_head[i] = layer[-1].attn._induction_score
            logger.info(f"End of getting layer {i} echo head and induction head, \
                        cost time is {time.time() - start_time}")
        logger.info("Select top heads...")
        selected_echo_head = self._select_top_heads(echo_head,
                                                    self._config.echo_head_ratio)
        select_induction_head = self._select_top_heads(induction_head,
                                                       self._config.induction_head_ratio)
        logger.info("Remove empty list from heads info...")
        selected_echo_head = self._remove_empty_list_keys(selected_echo_head)
        select_induction_head = self._remove_empty_list_keys(select_induction_head)
        logger.info("Concat echo heads and induction heads...")
        retrieval_heads = self._get_retrieval_heads(selected_echo_head, select_induction_head)

        logger.info("saving head info...")
        dir_path = '/'.join(self._config.retrieval_head_path.split('/')[:-1])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(self._config.retrieval_head_path, 'w+', encoding='utf8') as file_path:
            json.dump(retrieval_heads, file_path, indent=3)

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
        # 将所有列表里的值汇总
        all_values = [
            value
            for key in data
            for value in data[key]
        ]
        # 对汇总后的值进行排序
        sorted_values = sorted(all_values, reverse=True)
        # 计算前%的索引
        percent_index = round(len(sorted_values) * ratio)
        # 获取前%的值
        percent_values = sorted_values[:percent_index]
        # 创建一个新字典
        result = {}
        for key in data:
            # 获取前%的值在原列表中的索引
            percent_index_in_original_list = [i for i, value in enumerate(data[key]) if value in percent_values]
            result[key] = percent_index_in_original_list
        return result
