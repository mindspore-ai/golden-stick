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
import numpy as np

import mindspore as ms
from mindspore import dtype as msdtype
from mindspore import Model, Tensor
from mindspore.common import initializer
from mindspore.nn.utils import no_init_parameters
from mindformers import MindFormerConfig
from mindformers import build_context
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.core.parallel_config import build_parallel_config
from mindformers.models.qwen3.configuration_qwen3 import Qwen3Config

from mindspore_gs.ptq import PTQ
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQConfig, PTQMode, OutliersSuppressionType, QuantGranularity
from transformers import AutoTokenizer


def create_ptq(quant_type: str, quant_mode: PTQMode):
    """create_ptq"""
    if quant_type.lower() == 'awq-a16w4':
        cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.qint4x2,
                        act_quant_dtype=None, outliers_suppression=OutliersSuppressionType.AWQ,
                        opname_blacklist=['output_layer'], weight_quant_granularity=QuantGranularity.PER_GROUP,
                        group_size=128)
        layer_policies = OrderedDict()
    elif quant_type.lower() == 'smoothquant':
        cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                        act_quant_dtype=msdtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH,
                        opname_blacklist=['output_layer', 'linear_fc2'])
        layer_policies = OrderedDict()
    else:
        raise RuntimeError(f'Input unsupported quant type: {quant_type}.')
    ptq = PTQ(config=cfg, layer_policies=layer_policies)

    if 'awq' in quant_type.lower():
        # pylint: disable=protected-access
        ptq._config.weight_symmetric = False
    ptq._config.algorithm_cache_path = ""
    from mindformers.parallel_core.inference.transformer.transformer_layer import TransformerLayer
    ptq.decoder_layer_types.append(TransformerLayer)
    return ptq


def prepare_inputs_for_predict_layout(input_ids, **kwargs):
    """ Get deepseekv3 model input tuple for transform ckpt. """
    input_ids = Tensor(input_ids, ms.int32)
    labels = Tensor(kwargs["labels"]) if "labels" in kwargs else None
    bs, seq = input_ids.shape[0], input_ids.shape[1]
    slot_mapping = Tensor(np.ones(shape=tuple([bs * seq])), ms.int32)
    return input_ids, labels, None, None, None, None, None, None, None, None, None, \
        slot_mapping


def create_network(yaml_file, quant_type=None):
    """create_tokenizer"""
    config = MindFormerConfig(yaml_file)
    build_context(config)
    build_parallel_config(config)
    auto_online_trans = config.auto_trans_ckpt
    print('=' * 50, f"if using auto_online_trans: {auto_online_trans}", flush=True)
    model_config = Qwen3Config(**config.model.model_config)

    with no_init_parameters():
        from mindformers import AutoModel
        network = AutoModel.from_config(yaml_file)
    if quant_type:
        ptq = create_ptq(quant_type, PTQMode.DEPLOY)
        ptq.apply(network)
        ptq.convert(network)

    if config.load_checkpoint:
        if auto_online_trans:
            if not quant_type or quant_type == 'smoothquant':
                network.load_weights(config.load_checkpoint)
            else:
                raise NotImplementedError(f'Not supported quant_type: {quant_type}')
        else:
            ms_model = Model(network)
            seq_length = model_config.seq_length
            input_ids = Tensor(shape=(model_config.batch_size, seq_length), dtype=ms.int32, init=initializer.One())
            infer_data = prepare_inputs_for_predict_layout(input_ids)
            transform_and_load_checkpoint(config, ms_model, network, infer_data, do_predict=True)

    tokenizer = AutoTokenizer.from_pretrained(config.load_checkpoint)
    return tokenizer, network
