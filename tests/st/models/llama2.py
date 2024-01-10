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

from mindformers.modules.transformer.transformer import default_transformer_config, TransformerOpParallelConfig
from mindformers import LlamaForCausalLM, LlamaConfig


def llama2(batch_size: int = 32,
           seq_length: int = 4096,
           hidden_size: int = 8192,
           num_layers: int = 80,
           num_heads: int = 64,
           vocab_size: int = 32000,
           n_kv_heads: int = 8,
           bos_token_id: int = 1,
           eos_token_id: int = 2,
           pad_token_id: int = 0,
           ignore_token_id: int = -100,
           parallel_config: TransformerOpParallelConfig = default_transformer_config,
           checkpoint_name_or_path=""):
    """Create a Llama2 network for test from mindformers."""
    llama2_config = LlamaConfig(batch_size, seq_length, hidden_size, num_layers, num_heads, n_kv_heads,
                                vocab_size=vocab_size, multiple_of=256, ffn_dim_multiplier=1.3, rms_norm_eps=1.0e-5,
                                bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id,
                                ignore_token_id=ignore_token_id, compute_dtype="float16",
                                layernorm_compute_type="float32", softmax_compute_type="float16",
                                rotary_dtype="float16", param_init_type="float16", parallel_config=parallel_config)
    llama2_config.use_past = True
    llama2_config.use_flash_attention = False
    llama2_config.is_dynamic = True
    llama2_config.use_past_shard = True
    llama2_config.use_rope_slice = True
    llama2_config.use_kvcache_mgr = True
    llama2_config.checkpoint_name_or_path = checkpoint_name_or_path
    return LlamaForCausalLM(llama2_config)
