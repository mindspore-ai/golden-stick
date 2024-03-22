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
"""Quant llama2."""


import mindspore as ms
from mindformers import LlamaForCausalLM, MindFormerConfig, LlamaConfig, init_context, TransformerOpParallelConfig, \
    LlamaTokenizer, BaseModel
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQMode, PTQConfig
from mindspore_gs.ptq.smooth_quant.smooth_quant import SmoothQuant


def _set_config(config_path, device_id):
    """setup MindFormerConfig"""
    mfconfig = MindFormerConfig(config_path)
    if device_id != -1:
        mfconfig.context.device_id = device_id
    mfconfig.model.model_config = LlamaConfig(**mfconfig.model.model_config)

    init_context(use_parallel=mfconfig.use_parallel, context_config=mfconfig.context, parallel_config=mfconfig.parallel)
    if mfconfig.use_parallel:
        parallel_config = TransformerOpParallelConfig(**mfconfig.parallel_config)
        mfconfig.model.model_config.parallel_config = parallel_config
    mfconfig.model.model_config.checkpoint_name_or_path = mfconfig.load_checkpoint
    print(mfconfig)
    return mfconfig


def create_mfconfig(config_path, device_id, bs, seq_len, tokenizer_path="", ckpt_path=""):
    """Create mindformers config for llama2 network for example."""
    config = _set_config(config_path, device_id)
    config.model.model_config.batch_size = bs
    config.model.model_config.seq_length = seq_len
    config.model.model_config.compute_dtype = ms.float16
    config.model.model_config.layernorm_compute_type = ms.float32
    config.model.model_config.softmax_compute_type = ms.float16
    config.model.model_config.rotary_dtype = ms.float16
    config.model.model_config.param_init_type = ms.float16
    config.processor.tokenizer.vocab_file = tokenizer_path
    config.load_checkpoint = ckpt_path
    config.model.model_config.checkpoint_name_or_path = ckpt_path
    return config


def calibrate(net: BaseModel, tokenizer_: LlamaTokenizer, max_length, prompts):
    for prompt in prompts:
        input_ids = tokenizer_(prompt)['input_ids']
        _ = net.generate(input_ids, do_sample=False, max_length=max_length, top_p=1, top_k=3)


def quant_llama2(network: LlamaForCausalLM, backend: BackendTarget = BackendTarget.ASCEND, is_deploy: bool = False,
                 tokernizer=None, max_length=1024, prompts=None):
    """Quant llama2 model to w8a8 with smooth_quant algorithm."""
    if not is_deploy:
        print("Use RTN algo to quant network and weight.", flush=True)
        cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=backend)
    else:
        print("Use RTN algo to quant network.", flush=True)
        cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=backend)
    ptq = SmoothQuant(cfg)
    qnet = ptq.apply(network.model)
    if not is_deploy:
        if prompts is None:
            prompts = [
                "what's apple",
                "I like China, China is very Great",
                "Is Huawei a Great Company"
            ]
        network.model = qnet
        calibrate(network, tokernizer, max_length, prompts)
        qnet = network.model
    qnet = ptq.convert(qnet)
    network.model = qnet
    return network
