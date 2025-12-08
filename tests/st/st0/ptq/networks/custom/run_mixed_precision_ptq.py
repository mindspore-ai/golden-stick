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
"""Run mixed precision network PTQ test with configurable parameters via args"""
import argparse
import os
from collections import OrderedDict

import numpy as np
import mindspore as ms
from mindspore import Tensor, dtype as msdtype
from mindspore.dataset import GeneratorDataset

from linear_specs_loader import load_linear_specs_from_config
from layer_policies_loader import create_layer_policies
from mixed_precision_network import create_mixed_precision_network
from mindspore_gs.ptq import PTQ, PTQConfig, PTQMode
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq.ptq_config import (
    QuantGranularity, PrecisionRecovery, GPTQQuantConfig
)


def create_test_input(hidden_size):
    """Create fixed test input with shape (10, hidden_size), data range [-1, 1]"""
    np.random.seed(42)
    data = np.zeros((10, hidden_size), dtype=np.float32)
    for i in range(10):
        for j in range(hidden_size):
            value = ((i * 1000 + j) % 2000 - 1000) / 1000.0
            data[i, j] = value

    return Tensor(data, dtype=msdtype.float32)


def create_linear_ds(batch_size, seq_length, repeat=1, is_parallel=False):
    """Create linear dataset with fixed deterministic values, data range [-1, 1]"""
    class LinearIterable:
        """Iterable dataset for linear layers with fixed deterministic values"""
        def __init__(self, batch_size, seq_length, repeat=1, is_parallel=False):
            self.index = 0
            self.data = []
            for i in range(repeat):
                if not is_parallel:
                    data = np.zeros((batch_size, seq_length), dtype=np.float16)
                    for b in range(batch_size):
                        for s in range(seq_length):
                            value = ((i * 10000 + b * 1000 + s) % 2000 - 1000) / 1000.0
                            data[b, s] = value
                    self.data.append(data)
                else:
                    data = np.zeros((batch_size, 1, seq_length), dtype=np.float16)
                    for b in range(batch_size):
                        for s in range(seq_length):
                            value = ((i * 10000 + b * 1000 + s) % 2000 - 1000) / 1000.0
                            data[b, 0, s] = value
                    self.data.append(data)

        def __next__(self):
            if self.index >= len(self.data):
                raise StopIteration
            item = (self.data[self.index],)
            self.index += 1
            return item

        def __iter__(self):
            self.index = 0
            return self

        def __len__(self):
            return len(self.data)

    return GeneratorDataset(
        source=LinearIterable(batch_size, seq_length, repeat, is_parallel),
        column_names=["input_ids"])


def get_save_file_name(save_name):
    """Get the save file name"""
    return save_name


def quant_net(hidden_size, num_experts, linear_specs):
    """Quantize network: save quantized weights to ./mixed_precision_quant.ckpt, return original float output"""
    np.random.seed(42)
    ms.set_seed(42)

    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = 'on'
    os.environ['ENFORCE_EAGER'] = 'true'
    os.environ["RUN_MODE"] = "predict"
    os.environ['MS_ENABLE_LCCL'] = 'off'
    ascend_path = os.environ.get('ASCEND_HOME_PATH', '')
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = '/usr/local/Ascend/latest'

    dataset = create_linear_ds(
        batch_size=2,
        seq_length=512,
        repeat=10,
        is_parallel=False
    )

    network = create_mixed_precision_network(
        linear_specs=linear_specs,
        hidden_dim=hidden_size,
        num_layers=len(linear_specs),
        num_experts=num_experts,
        tensor_model_parallel_size=1
    )

    test_input = create_test_input(hidden_size)
    fp_outputs_dict = network.get_outputs_dict(test_input)

    ms.set_context(device_target="Ascend",
                   mode=ms.PYNATIVE_MODE,
                   jit_config={"jit_level": "O0", "infer_boost": "on"},
                   deterministic="ON")
    ms.set_seed(42)
    base_config = PTQConfig(
        mode=PTQMode.QUANTIZE,
        backend=BackendTarget.ASCEND,
        weight_quant_dtype=msdtype.int8,
        act_quant_dtype=msdtype.int8,
        act_quant_granularity=QuantGranularity.PER_TOKEN,
    )

    layer_policies = create_layer_policies(linear_specs)

    ptq = PTQ(config=base_config, layer_policies=layer_policies)
    # pylint: disable=protected-access
    ptq._config.experimental = True
    ptq._config.always_use_fp_input_in_processer = True
    ptq._config.skip_offload_in_processing = True
    ptq._config.algorithm_cache_path = {}
    ptq._config.fake_quant = True

    try:
        # pylint: disable=import-outside-toplevel,protected-access
        from mindspore_gs.ptq.plugins import MFModelHubPlugin
        MFModelHubPlugin()._load_quant_cells()
        MFModelHubPlugin()._load_algo_modules()
    except ImportError:
        pass

    network = ptq.apply(network, datasets=dataset)
    network = ptq.convert(network)
    ms.save_checkpoint(network.parameters_dict(), get_save_file_name('mixed_precision_quant.ckpt'),
                       choice_func=lambda x: all(i not in x for i in ['key_cache', 'value_cache', 'float_weight']))
    return fp_outputs_dict


def infer_net(hidden_size, num_experts, linear_specs):
    """Infer: load quantized weights from ./mixed_precision_quant.ckpt, return inference output"""
    np.random.seed(42)
    ms.set_seed(42)

    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = 'on'
    os.environ['MS_INTERNAL_ENABLE_CUSTOM_KERNEL_LIST'] = 'QbmmAllReduceAdd,QbmmAdd'
    os.environ['MS_ENABLE_LCCL'] = 'off'
    os.environ.pop('ENFORCE_EAGER', None)
    ascend_path = os.environ.get('ASCEND_HOME_PATH', '')
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = '/usr/local/Ascend/latest'

    create_linear_ds(
        batch_size=2,
        seq_length=512,
        repeat=10,
        is_parallel=False
    )

    network = create_mixed_precision_network(
        linear_specs=linear_specs,
        hidden_dim=hidden_size,
        num_layers=len(linear_specs),
        num_experts=num_experts,
        tensor_model_parallel_size=1
    )

    ms.set_context(device_target="Ascend",
                   mode=ms.GRAPH_MODE,
                   jit_config={"jit_level": "O0", "infer_boost": "on"},
                   deterministic="ON")
    base_config = PTQConfig(
        mode=PTQMode.DEPLOY,
        backend=BackendTarget.ASCEND,
        weight_quant_dtype=msdtype.int8,
        act_quant_dtype=msdtype.int8,
        act_quant_granularity=QuantGranularity.PER_TOKEN,
    )

    quant_layer_policies = create_layer_policies(linear_specs)
    layer_policies = OrderedDict()
    for key, config in quant_layer_policies.items():
        algo_args = config.algo_args
        if isinstance(algo_args, dict) and config.precision_recovery == PrecisionRecovery.GPTQ:
            algo_args = GPTQQuantConfig(
                block_size=algo_args.get('block_size', 128),
                desc_act=algo_args.get('desc_act', True),
                static_groups=algo_args.get('static_groups', True),
                damp_percent=algo_args.get('damp_percent', 0.1)
            )

        deploy_config = PTQConfig(
            mode=PTQMode.DEPLOY,
            backend=config.backend,
            weight_quant_dtype=config.weight_quant_dtype,
            act_quant_dtype=config.act_quant_dtype,
            kvcache_quant_dtype=config.kvcache_quant_dtype,
            outliers_suppression=config.outliers_suppression,
            precision_recovery=config.precision_recovery,
            act_quant_granularity=config.act_quant_granularity,
            weight_quant_granularity=config.weight_quant_granularity,
            kvcache_quant_granularity=config.kvcache_quant_granularity,
            group_size=config.group_size,
            opname_blacklist=config.opname_blacklist,
            weight_clip=config.weight_clip,
            algo_args=algo_args,
        )
        layer_policies[key] = deploy_config

    ptq = PTQ(config=base_config, layer_policies=layer_policies)
    # pylint: disable=protected-access
    ptq._config.experimental = True
    ptq._config.fake_quant = True
    ptq._config.algorithm_cache_path = {}

    try:
        # pylint: disable=import-outside-toplevel,protected-access
        from mindspore_gs.ptq.plugins import MFModelHubPlugin
        MFModelHubPlugin()._load_quant_cells()
        MFModelHubPlugin()._load_algo_modules()
    except ImportError:
        pass

    ptq.fake_quant(network)

    param_dict = ms.load_checkpoint(get_save_file_name('mixed_precision_quant.ckpt'))
    ms.load_param_into_net(network, param_dict)

    test_input = create_test_input(hidden_size)
    qoutputs_dict = network.get_outputs_dict(test_input)
    return qoutputs_dict


def main():
    """Main function to run mixed precision PTQ test"""
    parser = argparse.ArgumentParser(description='Run mixed precision network PTQ test')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for results (npz file)')
    args = parser.parse_args()

    hidden_size = 512
    num_experts = 2

    linear_specs = load_linear_specs_from_config(hidden_size=hidden_size, num_experts=num_experts)

    fp_outputs_dict = quant_net(hidden_size, num_experts, linear_specs)
    qoutputs_dict = infer_net(hidden_size, num_experts, linear_specs)

    # Save outputs to npz file
    output_dict = {}
    for layer_idx in range(len(linear_specs)):
        fp_output = fp_outputs_dict[layer_idx]
        qoutput = qoutputs_dict[layer_idx]

        fp_output_np = fp_output.asnumpy() if hasattr(fp_output, 'asnumpy') else fp_output
        qoutput_np = qoutput.asnumpy() if hasattr(qoutput, 'asnumpy') else qoutput

        fp_output_np = fp_output_np.astype(np.float32)
        qoutput_np = qoutput_np.astype(np.float32)

        output_dict[f'layer_{layer_idx}_fp'] = fp_output_np
        output_dict[f'layer_{layer_idx}_quant'] = qoutput_np

    np.savez(args.output_path, **output_dict)

    # Clean up checkpoint file
    ckpt_file = get_save_file_name('mixed_precision_quant.ckpt')
    if os.path.exists(ckpt_file):
        os.remove(ckpt_file)


if __name__ == "__main__":
    main()
