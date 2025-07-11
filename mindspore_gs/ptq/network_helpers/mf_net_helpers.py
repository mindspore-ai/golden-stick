# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Network helper for network from MindFormers."""
import os.path

from typing import Union, List

import math
from collections import OrderedDict
import numpy as np
import mindspore as ms
from mindspore import dtype as mstype
from mindspore import Tensor, Model
from mindspore.nn.utils import no_init_parameters

from mindformers import MindFormerConfig, build_context, AutoModel, build_parallel_config
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.models.build_tokenizer import build_tokenizer
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindspore_gs.common.utils import value_check
from .network_helper import NetworkHelper


class MFNetworkHelper(NetworkHelper):
    """
    Network helper for network from MindFormers.

    Args:
        config (Union[str, MindFormerConfig]): MindFormerConfig or path of config file for network.

    Raises:
        TypeError: If input `config` is not an instance of `MindFormerConfig` neither a str.
        ValueError: If input `config` is not a valid file path when input `config` is a str.
    """
    def __init__(self, config: Union[str, MindFormerConfig] = None):
        value_check("config", config, (MindFormerConfig, str))
        if isinstance(config, MindFormerConfig):
            self.mf_config = config
        else:
            if not os.path.isfile(config):
                raise ValueError(f"Input `config`({config}) is not a valid file path.")
            self.mf_config = MindFormerConfig(config)
        build_parallel_config(self.mf_config)
        self.mf_config.model.model_config.parallel_config = self.mf_config.parallel_config
        self.mf_config.model.model_config.moe_config = self.mf_config.moe_config

    def create_network(self):
        """
        Create network of type LlamaForCasualLM.

        Returns:
            Network of type LlamaForCasualLM.
        """
        build_context(self.mf_config)
        with no_init_parameters():
            network = AutoModel.from_config(self.mf_config, download_checkpoint=False)
        network.set_train(False)
        ckpt_path = self.mf_config.load_checkpoint
        if ckpt_path:
            self._load_ckpt(network)
        ms.ms_memory_recycle()
        return network

    def get_spec(self, name: str):
        """
        Get network specific, such as batch_size, seq_length and so on.

        Args:
            name (str): Name of specific.

        Returns:
            Object as network specific.
        """
        value_check('name', name, str)
        if name == 'vocab_file':
            return self.mf_config.processor.tokenizer.vocab_file
        model_config = self.mf_config.model.model_config
        if hasattr(model_config, name):
            return getattr(model_config, name)
        raise KeyError(f"Can not find network specific: {name}.")

    # pylint: disable=arguments-differ
    def create_tokenizer(self):
        """
        Get network tokenizer.

        Args:
            kwargs (Dict): Extensible parameter for subclasses.

        Returns:
            Object as network tokenizer.

        Examples:
            >>> from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper
            >>> from mindformers.tools.register.config import MindFormerConfig
            >>> mf_yaml_config_file = "/path/to/mf_yaml_config_file"
            >>> mfconfig = MindFormerConfig(mf_yaml_config_file)
            >>> helper = MFLlama2Helper(mfconfig)
            >>> helper.create_tokenizer()
            LlamaTokenizer(name_or_path='', vocab_size=32000, model_max_length=100000,  added_tokens_decoder={
                0: AddedToken("<unk>", rstrip=False, lstrip=False, normalized=True, special=True),
                1: AddedToken("<s>", rstrip=False, lstrip=False, normalized=True, special=True),
                2: AddedToken("</s>", rstrip=False, lstrip=False, normalized=True, special=True),
            })
        """
        return build_tokenizer(self.mf_config.processor.tokenizer)

    # pylint: disable=arguments-differ
    def generate(self, mf_network, input_ids: Union[np.ndarray, List[int], List[List[int]]],
                 max_new_tokens=None, **kwargs):
        """
        Invoke `network` and generate tokens.

        Args:
            network (Cell): Network to generate tokens.
            input_ids (numpy.ndarray): Input tokens for generate.
            max_new_tokens (int): Max number of tokens to be generated, default 1.
            kwargs (Dict): Extensible parameter for subclasses.

        Returns:
            A list as generated tokens.

        Examples:
            >>> import numpy as np
            >>> from mindspore import context
            >>> from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper
            >>> from mindformers import LlamaForCausalLM, LlamaConfig
            >>> from mindformers.tools.register.config import MindFormerConfig
            >>> context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
            >>> mf_yaml_config_file = "/path/to/mf_yaml_config_file"
            >>> mfconfig = MindFormerConfig(mf_yaml_config_file)
            >>> helper = MFLlama2Helper(mfconfig)
            >>> network = LlamaForCausalLM(LlamaConfig(**mfconfig.model.model_config))
            >>> input_ids = np.array([[1, 10000]], dtype = np.int32)
            >>> helper.generate(network, input_ids)
            array([[    1, 10000, 10001]], dtype=int32)
        """
        value_check('mf_network', mf_network, PreTrainedModel)
        value_check('input_ids', input_ids, (np.ndarray, List))
        if max_new_tokens:
            value_check('max_new_tokens', max_new_tokens, int)
        do_sample = self.get_spec('do_sample')
        seq = self.get_spec('seq_length')
        top_p = self.get_spec('top_p')
        top_k = self.get_spec('top_k')
        return mf_network.generate(input_ids, do_sample=do_sample, max_length=seq, max_new_tokens=max_new_tokens,
                                   top_p=top_p, top_k=top_k)

    def assemble_inputs(self, input_ids: np.ndarray, **kwargs):
        raise NotImplementedError

    def _load_ckpt(self, network):
        raise NotImplementedError


class MFLlama2Helper(MFNetworkHelper):
    """
    Derived from 'NetworkHelper', a utility class for the MindFormers framework Llama2 network.

    Args:
        config (MindFormerConfig): MindFormerConfig for network.

    Raises:
        TypeError: If input `config` is not an instance of `MindFormerConfig`.
    """
    def __init__(self, config: Union[str, MindFormerConfig] = None):
        super().__init__(config)
        self._decoder_infos = OrderedDict()

    @staticmethod
    def _get_slots(bs, block_size, prefill_max_len, is_prefill, block_tables, valid_length_example):
        """get_slots."""
        slot_mapping = []
        for i in range(bs):
            block_table = block_tables[i]
            if is_prefill:
                slots = [block_table[k // block_size] * block_size + k % block_size
                         for k in range(valid_length_example[i])]
                null_slot_idx = -1
                num_elements_to_add = prefill_max_len - valid_length_example[i]
                for _ in range(num_elements_to_add):
                    slots.append(null_slot_idx)
            else:
                current_idx = valid_length_example[i] - 1
                slots = [block_table[current_idx // block_size] * block_size + current_idx % block_size]
            slot_mapping = slot_mapping + slots

        return np.array(slot_mapping, copy=False, dtype=np.int32)

    @staticmethod
    def _get_pa_inputs(bs, seq, block_size, valid_length):
        """_get_pa_inputs"""
        valid_length_each_example = np.array([valid_length])
        prefill_max_len = max(valid_length_each_example)
        required_block_num = math.ceil(seq / block_size)
        block_tables = np.arange(required_block_num, dtype=np.int32).reshape(bs, -1)
        slot_mapping = MFLlama2Helper._get_slots(bs, block_size, prefill_max_len, True, block_tables,
                                                 valid_length_each_example)
        block_tables = Tensor(block_tables, mstype.int32)
        slot_mapping = Tensor(slot_mapping, mstype.int32)
        return block_tables, slot_mapping

    def assemble_inputs(self, input_ids: np.ndarray, **kwargs):
        """
        Assemble network inputs for predict from input tokens in numpy ndarray format.

        Args:
            input_ids (numpy.ndarray): Input tokens.
            kwargs (Dict): Extensible parameter for subclasses.

        Returns:
            A list of `mindspore.Tensor` as inputs of network predict.
        """
        value_check('input_ids', input_ids, np.ndarray)
        shape = input_ids.shape
        if len(shape) > 2:
            raise ValueError(f"Only support two-dimension(bs, seq_length) input_ids, got: {shape}.")
        bs = self.mf_config.model.model_config.batch_size
        seq = self.mf_config.model.model_config.seq_length
        if shape[0] > bs or shape[1] > seq:
            raise ValueError(f"Input input_ids shape({shape}) out of max shape({bs}, {seq}).")
        pad_token_id = self.mf_config.model.model_config.pad_token_id
        use_past = self.mf_config.model.model_config.use_past
        input_ids = np.pad(input_ids, ((0, bs - shape[0]), (0, seq - shape[1])), 'constant',
                           constant_values=pad_token_id)
        t_input_ids = Tensor(input_ids)
        if not use_past:
            return t_input_ids, None, None, None, None, None, None, None, None, None, None, None
        block_size = self.mf_config.model.model_config.block_size

        block_tables, slot_mapping = MFLlama2Helper._get_pa_inputs(bs, seq, block_size, shape[1])
        return t_input_ids, None, None, None, None, None, None, None, None, None, block_tables, slot_mapping

    def _load_ckpt(self, network):
        """_load_ckpt"""
        model = Model(network)
        input_ids = np.ones(shape=[self.get_spec('batch_size'), self.get_spec('seq_length')], dtype=np.int32)
        infer_data = network.prepare_inputs_for_predict_layout(input_ids)
        if self.mf_config.use_parallel:
            network.phase = 'infer_predict_layout'
            model.infer_predict_layout(*infer_data)
        transform_and_load_checkpoint(self.mf_config, model, network, infer_data, do_predict=True)


class MFParallelLlama2Helper(MFLlama2Helper):
    """
    Derived from 'NetworkHelper', a utility class for the MindFormers framework ParrallelLlamaForCasualLM network.

    Args:
        config (MindFormerConfig): A MindFormerConfig object indicates the network configuration.

    Raises:
        TypeError: If input `config` is not an instance of `MindFormerConfig`.

    Examples:
        >>> from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFParallelLlama2Helper
        >>> from mindformers.tools.register.config import MindFormerConfig
        >>> mf_yaml_config_file = "/path/to/mf_yaml_config_file"
        >>> mfconfig = MindFormerConfig(mf_yaml_config_file)
        >>> helper = MFParallelLlama2Helper(mfconfig)
        >>> network = helper.create_network()
    """
    def __init__(self, config: Union[str, MindFormerConfig] = None):
        super().__init__(config)
        try:
            # pylint: disable=unused-import
            from research.llama3_1.llama import ParallelLlamaForCausalLM
        except ImportError as e:
            raise ImportError('Please add mindformers repo root dir into PYTHONPATH.') from e

    def _load_ckpt(self, network):
        """_load_ckpt"""
        transform_and_load_checkpoint(self.mf_config, None, network, None)


class Qwen3Helper(MFLlama2Helper):
    """
    Derived from 'NetworkHelper', a utility class for the MindFormers framework ParrallelLlamaForCasualLM network.

    Args:
        config (MindFormerConfig): A MindFormerConfig object indicates the network configuration.

    Raises:
        TypeError: If input `config` is not an instance of `MindFormerConfig`.

    Examples:
        >>> from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFParallelLlama2Helper
        >>> from mindformers.tools.register.config import MindFormerConfig
        >>> mf_yaml_config_file = "/path/to/mf_yaml_config_file"
        >>> mfconfig = MindFormerConfig(mf_yaml_config_file)
        >>> helper = MFParallelLlama2Helper(mfconfig)
        >>> network = helper.create_network()
    """
    def _load_ckpt(self, network):
        """_load_ckpt"""
        transform_and_load_checkpoint(self.mf_config, None, network, None)


class MFParallelTeleChat2Helper(MFParallelLlama2Helper):
    """MFParallelTeleChat2Helper"""

    def create_network(self):
        try:
            from research.telechat2.infer.telechat import ParallelTelechatForCausalLM
            from research.telechat2.telechat_config import TelechatConfig
        except ImportError as e:
            raise ImportError('Please add mindformers repo root dir into PYTHONPATH.') from e
        build_context(self.mf_config)
        model_config = TelechatConfig(**self.mf_config.model.model_config)
        with no_init_parameters():
            network = ParallelTelechatForCausalLM(model_config)
        network.set_train(False)
        network.phase = 'predict'
        ckpt_path = self.mf_config.load_checkpoint
        if ckpt_path:
            self._load_ckpt(network)
        ms.ms_memory_recycle()
        network.phase = 'predict'
        return network

    def create_tokenizer(self):
        """create_tokenizer."""
        from research.telechat2.telechat_tokenizer import TelechatTokenizer
        return TelechatTokenizer(vocab_file=self.get_spec('vocab_file'))


class MFDSV3Helper(MFNetworkHelper):
    """MFDSV3Helper"""
    def __init__(self, config: Union[str, MindFormerConfig] = None):
        super().__init__(config)
        self._decoder_infos = OrderedDict()

    def create_network(self):
        try:
            from research.deepseek3.deepseek3 import DeepseekV3ForCausalLM
            from research.deepseek3.deepseek3_config import DeepseekV3Config
        except ImportError as e:
            raise ImportError('Please add mindformers repo root dir into PYTHONPATH.') from e
        build_context(self.mf_config)
        model_config = DeepseekV3Config(**self.mf_config.model.model_config)
        network = DeepseekV3ForCausalLM(model_config)
        network.set_train(False)
        network.phase = 'predict'
        ckpt_path = self.mf_config.load_checkpoint
        if ckpt_path:
            self._load_ckpt(network)
        ms.ms_memory_recycle()
        network.phase = 'predict'
        return network

    def create_tokenizer(self):
        """create_tokenizer."""
        from mindformers.models.llama.llama_tokenizer_fast import LlamaTokenizerFast
        return LlamaTokenizerFast(vocab_file=self.get_spec('vocab_file'),
                                  tokenizer_file=self.get_spec('tokenizer_file'))

    @staticmethod
    def _get_slots(bs, block_size, prefill_max_len, is_prefill, block_tables, valid_length_example):
        """get_slots."""
        slot_mapping = []
        for i in range(bs):
            block_table = block_tables[i]
            if is_prefill:
                slots = [block_table[k // block_size] * block_size + k % block_size
                         for k in range(valid_length_example[i])]
                null_slot_idx = -1
                num_elements_to_add = prefill_max_len - valid_length_example[i]
                for _ in range(num_elements_to_add):
                    slots.append(null_slot_idx)
            else:
                current_idx = valid_length_example[i] - 1
                slots = [block_table[current_idx // block_size] * block_size + current_idx % block_size]
            slot_mapping = slot_mapping + slots

        return np.array(slot_mapping, copy=False, dtype=np.int32)

    @staticmethod
    def _get_pa_inputs(bs, seq, block_size, valid_length):
        """_get_pa_inputs"""
        valid_length_each_example = np.array([valid_length])
        prefill_max_len = max(valid_length_each_example)
        required_block_num = math.ceil(seq / block_size)
        block_tables = np.arange(required_block_num, dtype=np.int32).reshape(bs, -1)
        slot_mapping = MFDSV3Helper._get_slots(bs, block_size, prefill_max_len, True, block_tables,
                                               valid_length_each_example)
        block_tables = Tensor(block_tables, mstype.int32)
        slot_mapping = Tensor(slot_mapping, mstype.int32)
        return block_tables, slot_mapping

    def assemble_inputs(self, input_ids: np.ndarray, **kwargs):
        """
        Assemble network inputs for predict from input tokens in numpy ndarray format.

        Args:
            input_ids (numpy.ndarray): Input tokens.
            kwargs (Dict): Extensible parameter for subclasses.

        Returns:
            A list of `mindspore.Tensor` as inputs of network predict.
        """
        value_check('input_ids', input_ids, np.ndarray)
        shape = input_ids.shape
        if len(shape) > 2:
            raise ValueError(f"Only support two-dimension(bs, seq_length) input_ids, got: {shape}.")
        bs = self.mf_config.model.model_config.batch_size
        seq = self.mf_config.model.model_config.seq_length
        if shape[0] > bs or shape[1] > seq:
            raise ValueError(f"Input input_ids shape({shape}) out of max shape({bs}, {seq}).")
        pad_token_id = self.mf_config.model.model_config.pad_token_id
        use_past = self.mf_config.model.model_config.use_past
        input_ids = np.pad(input_ids, ((0, bs - shape[0]), (0, seq - shape[1])), 'constant',
                           constant_values=pad_token_id)
        t_input_ids = Tensor(input_ids)
        if not use_past:
            return t_input_ids, None, None, None, None, None, None, None, None, None, None, None
        block_size = self.mf_config.model.model_config.block_size

        block_tables, slot_mapping = MFDSV3Helper._get_pa_inputs(bs, seq, block_size, shape[1])
        return t_input_ids, None, None, None, None, None, None, None, None, None, block_tables, slot_mapping

    def _load_ckpt(self, network):
        """_load_ckpt"""
        transform_and_load_checkpoint(self.mf_config, None, network, None)
