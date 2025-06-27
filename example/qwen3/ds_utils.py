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
"""ds infer."""

from collections import OrderedDict

import mindspore as ms
from mindspore import dtype as msdtype
from mindspore import Model, Tensor
from mindspore.common import initializer
from mindspore.nn.utils import no_init_parameters
from mindformers import MindFormerConfig
from mindformers import build_context, LlamaConfig
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.core.parallel_config import build_parallel_config

from mindspore_gs.ptq import PTQ
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQConfig, PTQMode, OutliersSuppressionType, QuantGranularity, PrecisionRecovery
from qwen3_weight_processor import Qwen3WeightProcessor
from qwen3_sq_weight_processor import Qwen3SQWeightProcessor
from research.qwen3.qwen3 import ParallelQwen3ForCausalLM
from transformers import AutoTokenizer


def create_ptq(quant_type: str, quant_mode: PTQMode):
    """create_ptq"""
    if quant_type.lower() == 'awq-a16w4':
        cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.qint4x2,
                        act_quant_dtype=None, outliers_suppression=OutliersSuppressionType.AWQ,
                        opname_blacklist=['lm_head'], weight_quant_granularity=QuantGranularity.PER_GROUP,
                        group_size=128)
        layer_policies = OrderedDict()
    elif quant_type.lower() == 'smoothquant':
        cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                        act_quant_dtype=msdtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH,
                        opname_blacklist=['lm_head'])
        w2_config = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                              act_quant_dtype=msdtype.int8,
                              outliers_suppression=OutliersSuppressionType.NONE,
                              precision_recovery=PrecisionRecovery.NONE,
                              act_quant_granularity=QuantGranularity.PER_TOKEN,
                              weight_quant_granularity=QuantGranularity.PER_CHANNEL)
        layer_policies = OrderedDict({r'.*\.w2.*': w2_config})
    else:
        raise RuntimeError(f'Input unsupported quant type: {quant_type}.')
    ptq = PTQ(config=cfg, layer_policies=layer_policies)

    if 'awq' in quant_type.lower():
        # pylint: disable=protected-access
        ptq._config.weight_symmetric = False
    ptq._config.algorithm_cache_path = {}
    from research.qwen3.qwen3_transformers import Qwen3ParallelTransformerLayer
    ptq.decoder_layer_types.append(Qwen3ParallelTransformerLayer)
    return ptq


def create_network(yaml_file, quant_type=None):
    """create_tokenizer"""
    config = MindFormerConfig(yaml_file)
    build_context(config)
    build_parallel_config(config)
    model_config = config.model.model_config
    model_config.parallel_config = config.parallel_config
    model_config.moe_config = config.moe_config
    auto_online_trans = config.auto_trans_ckpt
    print('=' * 50, f"if using auto_online_trans: {auto_online_trans}", flush=True)
    model_config = LlamaConfig(**model_config)

    with no_init_parameters():
        network = ParallelQwen3ForCausalLM(model_config)
    if quant_type:
        ptq = create_ptq(quant_type, PTQMode.DEPLOY)
        ptq.apply(network)
        ptq.convert(network)

    if config.load_checkpoint:
        if auto_online_trans:
            if not quant_type:
                model_parallelism = Qwen3WeightProcessor(config, network, None)
                model_parallelism.load_safetensors_shard(config.load_checkpoint)
            elif quant_type == 'smoothquant':
                model_parallelism = Qwen3SQWeightProcessor(config, network, quant_type)
                model_parallelism.load_safetensors_shard(config.load_checkpoint)
            else:
                raise NotImplementedError(f'Not supported quant_type: {quant_type}')
        else:
            ms_model = Model(network)
            seq_length = model_config.seq_length
            input_ids = Tensor(shape=(model_config.batch_size, seq_length), dtype=ms.int32, init=initializer.One())
            infer_data = network.prepare_inputs_for_predict_layout(input_ids)
            transform_and_load_checkpoint(config, ms_model, network, infer_data, do_predict=True)

    tokenizer = AutoTokenizer.from_pretrained(config.load_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, network
