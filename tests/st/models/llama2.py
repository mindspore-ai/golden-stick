# Copyright 2023 Huawei Technologies Co., Ltd
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
"""
Create llama2 from mindformers.
"""

import numpy as np
from mindspore import Tensor, dtype
from mindformers import LlamaForCausalLM, LlamaConfig


def llama2(batch_size: int = 32,
           seq_length: int = 4096,
           hidden_size: int = 8192,
           num_layers: int = 80,
           num_heads: int = 64,
           checkpoint_name_or_path=""):
    """Create a Llama2 network for test from mindformers."""
    llama2_config = LlamaConfig()
    llama2_config.batch_size = batch_size
    llama2_config.seq_length = seq_length
    llama2_config.hidden_size = hidden_size
    llama2_config.num_layers = num_layers
    llama2_config.num_heads = num_heads
    llama2_config.vocab_size = 32000
    llama2_config.multiple_of = 256
    llama2_config.rms_norm_eps = 1.0e-5
    llama2_config.bos_token_id = 1
    llama2_config.eos_token_id = 2
    llama2_config.pad_token_id = 0
    llama2_config.ignore_token_id = -100
    llama2_config.compute_dtype = dtype.float16
    llama2_config.layernorm_compute_type = dtype.float32
    llama2_config.softmax_compute_type = dtype.float16
    llama2_config.rotary_dtype = dtype.float32
    llama2_config.param_init_type = dtype.float32
    llama2_config.use_past = False
    llama2_config.pretrain_seqlen = 4096
    llama2_config.compute_in_2d = True
    llama2_config.use_flash_attention = True
    llama2_config.offset = 0
    llama2_config.use_past_shard = False
    llama2_config.checkpoint_name_or_path = checkpoint_name_or_path
    llama2_config.repetition_penalty = 1
    llama2_config.max_decode_length = 512
    llama2_config.top_k = 3
    llama2_config.top_p = 1
    return LlamaForCausalLM(llama2_config)


def _dummy_tensor(shape, dt):
    """create dummy tensor"""
    if None in shape:
        return Tensor(shape=shape, dtype=dt)
    return Tensor(np.ones(shape=tuple(shape)), dtype=dt)


def create_dummy_inputs(bs=None, seqlen=None, activate_len_shape=None):
    input_ids = _dummy_tensor(shape=[bs, seqlen], dt=dtype.int32)
    input_position = _dummy_tensor(shape=[bs], dt=dtype.int32)
    batch_valid_length = _dummy_tensor(shape=[bs], dt=dtype.int64)
    batch_index = _dummy_tensor(shape=[bs], dt=dtype.int64)
    activate_len = _dummy_tensor(shape=[activate_len_shape], dt=dtype.int64)
    return [input_ids, None, input_position, None, None, None, None, batch_valid_length, batch_index, activate_len]
