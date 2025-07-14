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
"""OmniQuant algorithm."""
import argparse
import gc
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.nn import Cell
from mindspore import Tensor
from mindformers import (LlamaForCausalLM, MindFormerConfig, LlamaConfig,
                         init_context, TransformerOpParallelConfig, LlamaTokenizer)
from mindformers.modules import Linear
from mindspore_gs.experimental.omniquant.oq_linear_wrappers import OqLinearWrapper


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--fp_ckpt_path', '-k', type=str, required=True)
    parser.add_argument('--device_id', '-d', type=int, required=True)
    parser.add_argument('--tokenizer_path', '-t', type=str, required=True)
    args = parser.parse_args()
    print(f"-------------------------------------------------quant args: {args}", flush=True)
    return args


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


def apply(net: Cell) -> Cell:
    """Apply"""
    op_types = [nn.Dense, nn.Conv2d, Linear]

    def _replace(root: Cell):
        if root is None:
            return
        for name, cell in root.name_cells().items():
            if isinstance(cell, tuple(op_types)):
                cell_wrapper = OqLinearWrapper(cell)
                root.insert_child_to_cell(name, cell_wrapper)
            else:
                _replace(cell)

    _replace(net)
    net.update_parameters_name()
    return net


def calibrate(model, tokenizer_: LlamaTokenizer, max_length, prompts):
    """calibrate"""
    ms.set_context(mode=ms.PYNATIVE_MODE)
    for prompt in prompts:
        input_ids = tokenizer_(prompt)['input_ids']
        pad_length = max_length - len(input_ids)
        token = np.pad(input_ids, (0, pad_length), "constant", constant_values=(0, 0),)
        tokens = model.reshape(Tensor(token), (1, -1))
        bs, seq_len = model.shape(tokens)
        freqs_cis = model.freqs_mgr(seq_len)
        mask = model.casual_mask(tokens)  # mask: [bs, seq, seq]
        h = model.tok_embeddings(tokens)
        h = model.reshape(h, (bs, seq_len, model.hidden_size))
        fpin = h
        quantin = h
        for i in range(model.num_layers):
            layer = model.layers[i]
            all_params1 = layer.get_parameters()
            all_params2 = layer.get_parameters()
            all_params3 = layer.get_parameters()
            lwc_low_params = list(filter(lambda x: "lowbound" in x.name, all_params1))
            lwc_up_params = list(filter(lambda x: "upbound" in x.name, all_params2))
            let_params = list(filter(lambda x: "smoothscale" in x.name, all_params3))
            omi_params = [{'params': let_params, 'lr': 0.00001},
                          {'params': lwc_low_params, 'lr': 0.0001},
                          {'params': lwc_up_params, 'lr': 0.0001}]
            optimizer = nn.AdamWeightDecay(params=omi_params)
            fpinfpout = layer(fpin, freqs_cis, mask)
            quantinfpout = layer(quantin, freqs_cis, mask)
            loss_fn = nn.MSELoss() # loss function

            def setflag(root):
                if root is None:
                    return
                for _, cell in root.name_cells().items():
                    if isinstance(cell, OqLinearWrapper):
                        cell.set_use_temporary_parameter()
                    else:
                        setflag(cell)

            def paramstore(root):
                if root is None:
                    return
                for _, cell in root.name_cells().items():
                    if isinstance(cell, OqLinearWrapper):
                        cell.paramstore()
                    else:
                        paramstore(cell)

            def paramconfirm(root):
                if root is None:
                    return
                for _, cell in root.name_cells().items():
                    if isinstance(cell, OqLinearWrapper):
                        cell.paramconfirm()
                    else:
                        paramconfirm(cell)

            # pylint: disable=W0640
            def forword_fn(inputs, targets1, targets2):
                setflag(layer)
                layer.set_train(True)
                logits = layer(inputs, freqs_cis, mask)
                loss1 = loss_fn(logits, targets1)
                loss2 = loss_fn(logits, targets2)
                loss = loss1+loss2
                del loss1
                del loss2
                return loss, logits

            # get grad function
            grad_fn = ms.value_and_grad(forword_fn, None, optimizer.parameters)

            # pylint: disable=W0640
            def train_step(inputs, targets1, targets2):
                layer.set_train(True)
                (loss, logits), grads = grad_fn(inputs, targets1, targets2) # get values and gradients
                optimizer(grads)
                del grads
                return loss, logits

            minloss = float("inf")
            for _ in range(10):
                loss, logits = train_step(quantin, quantinfpout, fpinfpout)
                if loss < minloss:
                    minloss = loss
                    quantin2 = logits
                    paramstore(layer)

            paramconfirm(layer)
            print(f'第{i}层量化完成')
            quantin = quantin2
            fpin = fpinfpout

            del optimizer
            del omi_params
            del all_params1
            del all_params2
            del all_params3
            del lwc_low_params
            del lwc_up_params
            del let_params
            del loss_fn
            del layer
            del loss
            del minloss
            del logits
            gc.collect()


def omniquant(model, tokernizer=None, max_length=None, prompts=None):
    """omniquant"""
    qnet = apply(model)
    if prompts is None:
        prompts = [
            "I like China, China is very Great",
        ]
    calibrate(qnet, tokernizer, max_length, prompts)
    model = qnet
    return model

if __name__ == "__main__":
    uargs = get_args()
    cfg = create_mfconfig(uargs.config_path, uargs.device_id, 1, 512, ckpt_path=uargs.fp_ckpt_path)
    network = LlamaForCausalLM(cfg.model.model_config)
    tokenizer = LlamaTokenizer(vocab_file=uargs.tokenizer_path)
    network.set_train(False)
    network.phase = 'predict'
    testprompts = [
                "what's apple",
                "I like China, China is very Great",
                "I love Beijing, because"
    ]
    for testprompt in testprompts:
        testinput_ids = tokenizer(testprompt)['input_ids']
        _ = network.generate(testinput_ids, do_sample=False, max_length=512)
        print(f'量化前结果为{_}')
        print(f'量化前结果为{tokenizer.decode(_)}')

    print('------------ quant llama2 to W8A16 ------------', flush=True)

    modelinput = network.model
    quantmodel = omniquant(model=modelinput, tokernizer=tokenizer, max_length=512)
    network.model = quantmodel

    print('------------ quant finished ------------', flush=True)
    for testprompt in testprompts:
        testinput_ids = tokenizer(testprompt)['input_ids']
        _ = network.generate(testinput_ids, do_sample=False, max_length=512)
        print(f'量化后结果为{_}')
        print(f'量化后结果为{tokenizer.decode(_)}')

    ms.save_checkpoint(network, f"llama2-w8a8{uargs.device_id}.ckpt")
