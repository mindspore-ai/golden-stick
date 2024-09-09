# Copyright 2022 Huawei Technologies Co., Ltd
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
"""util class about NetworkHelper"""

import os
from typing import Union, List

import numpy as np
from mindspore.dataset import GeneratorDataset
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper
from mindformers import MindFormerConfig, LlamaConfig, init_context, TransformerOpParallelConfig


def set_config(config_path):
    """setup MindFormerConfig"""
    mfconfig = MindFormerConfig(config_path)
    mfconfig.model.model_config = LlamaConfig(**mfconfig.model.model_config)
    init_context(use_parallel=mfconfig.use_parallel, context_config=mfconfig.context, parallel_config=mfconfig.parallel)
    if mfconfig.use_parallel:
        parallel_config = TransformerOpParallelConfig(**mfconfig.parallel_config)
        mfconfig.model.model_config.parallel_config = parallel_config
    mfconfig.model.model_config.checkpoint_name_or_path = mfconfig.load_checkpoint
    device_id = int(os.environ.get('DEVICE_ID', '0'))
    mfconfig.context.device_id = device_id
    print(f"---- use device_id: {device_id}", flush=True)
    return mfconfig


class MFLlama2HelloNetworkHelper(MFLlama2Helper):
    """SimpleNetworkHelper"""

    def generate(self, mf_network, input_ids: Union[np.ndarray, List[int], List[List[int]]],
                 max_new_tokens=1, **kwargs):
        do_sample = self.mf_config.model.model_config.do_sample
        seq = self.mf_config.model.model_config.seq_length
        top_p = self.mf_config.model.model_config.top_p
        top_k = self.mf_config.model.model_config.top_k
        return mf_network.generate(input_ids, do_sample=do_sample, max_length=seq, top_p=top_p, top_k=top_k)


def create_hello_ds(tokenizer, repeat=1):
    """create_hello_ds"""
    class SimpleIterable:
        """SimpleIterable"""
        def __init__(self, tokenizer, repeat=1):
            self._index = 0
            self.data = []
            for _ in range(repeat):
                input_ids = tokenizer("Hello")['input_ids']
                self.data.append(input_ids)

        def __next__(self):
            if self._index >= len(self.data):
                raise StopIteration
            item = (self.data[self._index],)
            self._index += 1
            return item

        def __iter__(self):
            self._index = 0
            return self

        def __len__(self):
            return len(self.data)

    return GeneratorDataset(source=SimpleIterable(tokenizer, repeat), column_names=["input_ids"])
