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
"""test interfaces of smooth quant."""
import argparse

import os
import sys
import numpy as np

import mindspore as ms
from mindspore.communication import get_rank
from mindspore import save_checkpoint
from mindspore import dtype
from mindspore.dataset import GeneratorDataset
from mindformers.trainer.utils import transform_and_load_checkpoint

from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq.ptq_config import (PTQConfig, PTQMode,
                                         OutliersSuppressionType,
                                         PrecisionRecovery, GPTQQuantConfig,
                                         QuantGranularity, AWQConfig)
from mindspore_gs.ptq.context import LayerQuantizeAlgo
from mindspore_gs.ptq.ptq import PTQ
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFParallelLlama2Helper
from mindspore_gs.common.utils import offload_network

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../mindformers")))


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


def create_cfg(quant_algo_, mode):
    """create_cfg"""
    if quant_algo_ == 'A8W8':
        cfg = PTQConfig(mode=mode,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        act_quant_dtype=dtype.int8,
                        weight_quant_dtype=dtype.int8,
                        outliers_suppression=OutliersSuppressionType.SMOOTH)
    elif quant_algo_ == 'A16W8':
        cfg = PTQConfig(mode=mode,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["lm_head"])
    elif quant_algo_ == 'A16W4_GPTQ':
        algorithm_config = GPTQQuantConfig(block_size=32)
        cfg = PTQConfig(mode=mode,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        weight_quant_dtype=dtype.qint4x2,
                        precision_recovery=PrecisionRecovery.GPTQ,
                        algo_args=algorithm_config)
    elif quant_algo_ == 'A16W4_GPTQ_per_group':
        algorithm_config = GPTQQuantConfig(block_size=128)
        cfg = PTQConfig(mode=mode,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        weight_quant_dtype=dtype.qint4x2,
                        precision_recovery=PrecisionRecovery.GPTQ,
                        weight_quant_granularity=QuantGranularity.PER_GROUP,
                        group_size=128,
                        algo_args=algorithm_config)
    elif quant_algo_ == 'A8W4_GPTQ':
        algorithm_config = GPTQQuantConfig(block_size=32, desc_act=True)
        cfg = PTQConfig(mode=mode,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        weight_quant_dtype=dtype.qint4x2,
                        act_quant_dtype=dtype.int8,
                        act_quant_granularity=QuantGranularity.PER_TOKEN,
                        precision_recovery=PrecisionRecovery.GPTQ,
                        algo_args=algorithm_config)
    elif quant_algo_ == 'A16W4_AWQ':
        algorithm_config = AWQConfig()
        # FIXME yrf:add per_group when use 1decoder
        cfg = PTQConfig(mode=mode,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        outliers_suppression=OutliersSuppressionType.AWQ,
                        weight_quant_dtype=dtype.qint4x2,
                        algo_args=algorithm_config)
    elif quant_algo_ == 'C8':
        cfg = PTQConfig(mode=mode,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        weight_quant_dtype=None,
                        kvcache_quant_dtype=dtype.int8)
    elif quant_algo_ == 'A8W8C8':
        cfg = PTQConfig(mode=mode,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        act_quant_dtype=dtype.int8,
                        weight_quant_dtype=dtype.int8,
                        outliers_suppression=OutliersSuppressionType.SMOOTH,
                        kvcache_quant_dtype=dtype.int8)
    elif quant_algo_ == 'A16W8C8':
        cfg = PTQConfig(mode=mode,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["lm_head"],
                        kvcache_quant_dtype=dtype.int8)
    elif quant_algo_ == 'A8W8_FallBack':
        cfg = PTQConfig(mode=mode,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["lm_head"],
                        act_quant_dtype=dtype.int8,
                        weight_quant_dtype=dtype.int8,
                        outliers_suppression=OutliersSuppressionType.SMOOTH)
    elif quant_algo_ == 'A8W8_Dynamic':
        cfg = PTQConfig(mode=mode,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["lm_head"],
                        act_quant_dtype=dtype.int8,
                        weight_quant_dtype=dtype.int8,
                        act_quant_granularity=QuantGranularity.PER_TOKEN,
                        outliers_suppression=OutliersSuppressionType.SMOOTH)
    elif quant_algo_ == 'C8_Dynamic':
        cfg = PTQConfig(mode=mode,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["lm_head"],
                        weight_quant_dtype=None,
                        kvcache_quant_granularity=QuantGranularity.PER_TOKEN,
                        kvcache_quant_dtype=dtype.int8)
    elif quant_algo_ == 'Quant_A8W16_Deploy_A8W8_Dynamic':
        if mode == PTQMode.DEPLOY:
            cfg = PTQConfig(mode=mode,
                            backend=BackendTarget.ASCEND,
                            opname_blacklist=["lm_head"],
                            act_quant_dtype=dtype.int8,
                            weight_quant_dtype=dtype.int8,
                            act_quant_granularity=QuantGranularity.PER_TOKEN,
                            outliers_suppression=OutliersSuppressionType.SMOOTH)
        else:
            cfg = PTQConfig(mode=mode,
                            backend=BackendTarget.ASCEND,
                            opname_blacklist=["lm_head"],
                            outliers_suppression=OutliersSuppressionType.SMOOTH)
    elif quant_algo_ == 'OSL_A8W8':
        cfg = PTQConfig(mode=mode,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        weight_quant_dtype=dtype.int8, act_quant_dtype=dtype.int8,
                        outliers_suppression=OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE,
                        act_quant_granularity=QuantGranularity.PER_TENSOR,
                        weight_quant_granularity=QuantGranularity.PER_CHANNEL,
                        kvcache_quant_granularity=QuantGranularity.PER_CHANNEL)
    elif quant_algo_ == 'OSPQuant_A8W8':
        os.environ["MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST"] = "RmsNormQuant,PagedAttention"
        cfg = PTQConfig(mode=mode,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        weight_quant_dtype=dtype.int8, act_quant_dtype=dtype.int8,
                        outliers_suppression=OutliersSuppressionType.OUTLIER_SUPPRESSION_PLUS,
                        act_quant_granularity=QuantGranularity.PER_TENSOR,
                        weight_quant_granularity=QuantGranularity.PER_CHANNEL,
                        kvcache_quant_granularity=QuantGranularity.PER_CHANNEL)
    else:
        raise ValueError(f"Unsupported quant_algo : {quant_algo_}")
    return cfg


def quant_llama2(config_path_, ckpt_path, output_dir_, quant_algo_):
    """quant llama2 using PTQ"""
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
    os.environ['FORCE_EAGER'] = "true"
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    cur_dir_ = os.path.dirname(os.path.abspath(__file__))
    config_path_ = os.path.join(cur_dir_, config_path_)
    vocab_file = os.path.join(cur_dir_, "../../../data/llama2-tokenizer.model")

    helper = MFParallelLlama2Helper(config_path_)
    helper.mf_config.load_checkpoint = os.path.join(cur_dir_, ckpt_path)
    helper.mf_config.output_dir = os.path.join(cur_dir_, output_dir_)
    helper.mf_config.processor.tokenizer.vocab_file = vocab_file
    device_id = int(os.environ.get('DEVICE_ID', '0'))
    helper.mf_config.context.device_id = device_id
    config = helper.mf_config

    network = helper.create_network()
    tokenizer = helper.create_tokenizer()

    cfg = create_cfg(quant_algo_, PTQMode.QUANTIZE)
    ptq = PTQ(config=cfg)
    if quant_algo_ == "A8W8_FallBack":
        # pylint: disable=W0212
        ptq._config.fallback_blacklist = {'w2': LayerQuantizeAlgo.A16W8}
    if quant_algo_ == "A16W4_AWQ":
        # pylint: disable=W0212
        ptq._config.weight_symmetric = False
    if quant_algo_ == "OSPQuant_A8W8":
        # pylint: disable=W0212
        ptq._config.use_inner_osp = True
    # pylint: disable=W0212
    ptq._config.enable_deploy_fusion = False
    ds = create_hello_ds(tokenizer, 1)
    network = ptq.apply(network, datasets=ds)
    network = ptq.convert(network)
    try:
        rank_id = get_rank()
    except RuntimeError:
        rank_id = 0
    save_path = os.path.join(config.output_dir, f"rank_{rank_id}")
    os.makedirs(save_path, exist_ok=True)
    save_checkpoint(network.parameters_dict(), os.path.join(save_path, "quant.ckpt"),
                    choice_func=lambda x: "key_cache" not in x and "value_cache" not in x and "float_weight" not in x)
    offload_network(network)
    os.environ.pop('FORCE_EAGER', None)


def eval_llama2(input_, is_quant, config_path_, ckpt_path_, quant_algo_=""):
    """eval llama2 by float ckpt and int ckpt"""
    ms.set_context(mode=0)
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
    os.environ['MS_INTERNAL_ENABLE_CUSTOM_KERNAL_LIST'] = "QbmmAllReduceAdd,QbmmAdd"
    os.environ.pop('FORCE_EAGER', None)
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    cur_dir_ = os.path.dirname(os.path.abspath(__file__))
    config_path_ = os.path.join(cur_dir_, config_path_)
    vocab_file = os.path.join(cur_dir_, "../../../data/llama2-tokenizer.model")

    helper = MFParallelLlama2Helper(config_path_)
    helper.mf_config.load_checkpoint = "" #os.path.join(cur_dir_, ckpt_path_)
    helper.mf_config.processor.tokenizer.vocab_file = vocab_file
    helper.mf_config.context.mode = 0

    device_id = int(os.environ.get('DEVICE_ID', '0'))
    helper.mf_config.context.device_id = device_id
    config = helper.mf_config
    network = helper.create_network()
    os.environ['MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST'] = "PagedAttention"
    if is_quant:
        cfg = create_cfg(quant_algo_, PTQMode.DEPLOY)
        ptq = PTQ(config=cfg)
        if quant_algo_ == "A8W8_FallBack":
            # pylint: disable=W0212
            ptq._config.fallback_blacklist = {'w2': LayerQuantizeAlgo.A16W8}
        if quant_algo_ == "OSPQuant_A8W8":
            # pylint: disable=W0212
            ptq._config.use_inner_osp = True
        # pylint: disable=W0212
        ptq._config.enable_deploy_fusion = False
        network = ptq.apply(network)
        network = ptq.convert(network)
    config.load_checkpoint = os.path.join(cur_dir_, ckpt_path_)
    transform_and_load_checkpoint(config, None, network, None)
    tokenizer = helper.create_tokenizer()

    seq_len = 100
    input_ids = tokenizer(input_)['input_ids']
    os.environ['MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST'] = "PagedAttention"
    outputs = network.generate(input_ids, do_sample=False, max_length=seq_len, top_p=1, top_k=3)
    answer = tokenizer.decode(outputs, skip_special_tokens=True)
    return outputs, answer


def print_output(qoutput_, foutput_):
    print(f"qoutput: {qoutput_}", flush=True)
    print(f"foutput: {foutput_}", flush=True)
    print(f"First not equal index {np.min(np.where((qoutput_ - foutput_) != 0))}", flush=True)


def ptq_llama2_predict_2stage(config_path_, fp16_ckpt_path_, quant_ckpt_path_, output_dir_, model_parallel_,
                              quant_algo_):
    """ptq_llama2_predict_2stage"""
    example = "Hello"
    quant_llama2(config_path_, fp16_ckpt_path_, output_dir_, quant_algo_)
    qoutput, _ = eval_llama2(input_=example, is_quant=True,
                             config_path_=config_path_, ckpt_path_=quant_ckpt_path_,
                             quant_algo_=quant_algo_)
    foutput, _ = eval_llama2(input_=example, is_quant=False,
                             config_path_=config_path_, ckpt_path_=fp16_ckpt_path_)
    qoutput = np.array(qoutput)
    foutput = np.array(foutput)
    if model_parallel_ == 1:
        if quant_algo_ == 'A8W8':
            ret = np.allclose(qoutput[:, :69], foutput[:, :69], 0, 0)
        elif quant_algo_ == 'A16W8':
            ret = np.allclose(qoutput, foutput, 0, 0)
        elif quant_algo_ == 'A16W4_GPTQ':
            ret = np.allclose(qoutput[:, :3], foutput[:, :3], 0, 0)
        elif quant_algo_ == 'A8W4_GPTQ':
            ret = np.allclose(qoutput[:, :10], foutput[:, :10], 0, 0)
        elif quant_algo_ == 'A16W4_GPTQ_per_group':
            ret = np.allclose(qoutput[:, :3], foutput[:, :3], 0, 0)
        elif quant_algo_ == 'A16W4_AWQ':
            ret = np.allclose(qoutput[:, :3], foutput[:, :3], 0, 0)
        elif quant_algo_ == 'A16W8C8':
            ret = np.allclose(qoutput[:, :5], foutput[:, :5], 0, 0)
        elif quant_algo_ == 'A8W8C8':
            ret = np.allclose(qoutput[:, :3], foutput[:, :3], 0, 0)
        elif quant_algo_ == 'C8':
            ret = np.allclose(qoutput[:, :5], foutput[:, :5], 0, 0)
        elif quant_algo_ == 'A8W8_Dynamic':
            ret = np.allclose(qoutput[:, :100], foutput[:, :100], 0, 0)
        elif quant_algo_ == 'Quant_A8W16_Deploy_A8W8_Dynamic':
            ret = np.allclose(qoutput[:, :100], foutput[:, :100], 0, 0)
        elif quant_algo_ == 'OSL_A8W8':
            ret = np.allclose(qoutput[:, :3], foutput[:, :3], 0, 0)
        elif quant_algo_ == 'OSPQuant_A8W8':
            ret = np.allclose(qoutput[:, :3], foutput[:, :3], 0, 0)
        else:
            assert False
        if not ret:
            print_output(qoutput, foutput)
        return ret
    if quant_algo_ == 'A8W8':
        ret = np.allclose(qoutput[:, :3], foutput[:, :3], 0, 0)
    elif quant_algo_ == 'A16W8':
        ret = np.allclose(qoutput, foutput, 0, 0)
    elif quant_algo_ == 'A16W4_GPTQ':
        ret = np.allclose(qoutput[:, :10], foutput[:, :10], 0, 0)
    elif quant_algo_ == 'A8W4_GPTQ':
        ret = np.allclose(qoutput[:, :9], foutput[:, :9], 0, 0)
    elif quant_algo_ == 'A16W4_GPTQ_per_group':
        ret = np.allclose(qoutput[:, :7], foutput[:, :7], 0, 0)
    elif quant_algo_ == 'A16W4_AWQ':
        ret = np.allclose(qoutput[:, :10], foutput[:, :10], 0, 0)
    elif quant_algo_ == 'A16W8C8':
        ret = np.allclose(qoutput[:, :7], foutput[:, :7], 0, 0)
    elif quant_algo_ == 'A8W8C8':
        ret = np.allclose(qoutput[:, :3], foutput[:, :3], 0, 0)
    elif quant_algo_ == 'C8':
        ret = np.allclose(qoutput[:, :3], foutput[:, :3], 0, 0)
    elif quant_algo_ == "A8W8_FallBack":
        ret = np.allclose(qoutput[:, :13], foutput[:, :13], 0, 0)
    elif quant_algo_ == "A8W8_Dynamic":
        ret = np.allclose(qoutput[:, :58], foutput[:, :58], 0, 0)
    elif quant_algo_ == 'Quant_A8W16_Deploy_A8W8_Dynamic':
        ret = np.allclose(qoutput[:, :100], foutput[:, :100], 0, 0)
    elif quant_algo_ == 'OSL_A8W8':
        ret = np.allclose(qoutput[:, :3], foutput[:, :3], 0, 0)
    elif quant_algo_ == 'OSPQuant_A8W8':
        ret = np.allclose(qoutput[:, :3], foutput[:, :3], 0, 0)
    else:
        assert False
    if not ret:
        print_output(qoutput, foutput)
    return ret


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_parallel', '-m', type=int, default=1)
    parser.add_argument('--quant_algo', '-a', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    uargs = get_args()
    model_parallel = uargs.model_parallel
    quant_algo = uargs.quant_algo

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    if model_parallel == 1:
        config_path = os.path.join(cur_dir, "../../../data/test_llama2/predict_parallelLlama2_13b_1p.yaml")
        config_path_per_group = os.path.join(cur_dir,
                                             "../../../data/test_llama2/predict_parallelLlama2_13b_1p_per_group.yaml")
        fp16_ckpt_path = os.path.join(cur_dir, "../../../data/test_llama2/parallelLlama2-fp16-1decoder-1p")
        quant_ckpt_path = f"../../../data/test_llama2/parallelLlama2-quant-1decoder-1p-{quant_algo}/rank_0/quant.ckpt"
        quant_ckpt_path = os.path.join(cur_dir, quant_ckpt_path)
        output_dir = os.path.join(cur_dir, f"../../../data/test_llama2/parallelLlama2-quant-1decoder-1p-{quant_algo}")
        if quant_algo == "A16W4_GPTQ_per_group":
            assert ptq_llama2_predict_2stage(config_path_per_group, fp16_ckpt_path, quant_ckpt_path, output_dir,
                                             model_parallel, quant_algo)
        else:
            assert ptq_llama2_predict_2stage(config_path, fp16_ckpt_path, quant_ckpt_path, output_dir,
                                             model_parallel, quant_algo)
    elif model_parallel == 2:
        config_path = os.path.join(cur_dir, "../../../data/test_llama2/predict_parallelLlama2_13b_2p.yaml")
        config_path_per_group = os.path.join(cur_dir,
                                             "../../../data/test_llama2/predict_parallelLlama2_13b_2p_per_group.yaml")
        fp16_ckpt_path = os.path.join(cur_dir, "../../../data/test_llama2/parallelLlama2-fp16-1decoder-2p")
        quant_ckpt_path = os.path.join(cur_dir,
                                       f"../../../data/test_llama2/parallelLlama2-quant-1decoder-2p-{quant_algo}")
        output_dir = os.path.join(cur_dir, f"../../../data/test_llama2/parallelLlama2-quant-1decoder-2p-{quant_algo}")
        if quant_algo == "A16W4_GPTQ_per_group":
            assert ptq_llama2_predict_2stage(config_path_per_group, fp16_ckpt_path, quant_ckpt_path, output_dir,
                                             model_parallel, quant_algo)
        else:
            assert ptq_llama2_predict_2stage(config_path, fp16_ckpt_path, quant_ckpt_path, output_dir,
                                             model_parallel, quant_algo)
    else:
        raise ValueError(f"Unsupported model_parallel: {model_parallel}")
