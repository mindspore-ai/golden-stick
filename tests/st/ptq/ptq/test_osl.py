# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Unit Tests for Outlier Suppression Lite"""
import os
import sys
import argparse
import json
import pytest
import numpy as np

import mindspore as ms
from mindspore import ops as msops
from mindspore import dtype as msdtype
from mindspore import nn, Tensor
from mindspore.dataset import GeneratorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../mindformers")))
from mindformers.parallel_core.inference.tensor_parallel.mappings import scatter_to_model_parallel_region
from mindformers.parallel_core.inference.parallel_state import (default_pgs, get_tensor_model_parallel_group,
                                                                is_initialized)
from mindformers.parallel_core.inference.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
)
from mindformers.parallel_core.transformer_config import TransformerConfig
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQ, PTQConfig, PTQMode, OutliersSuppressionType
from mindspore_gs.ptq.plugins import MFModelHubPlugin

ms.set_context(pynative_synchronize=True)

#############################################################################
# Parallel runner
#   When called directly, the file will run in parallel mode, invoking one testcase;
#   When called by pytest, it will run in single card mode and go through each testcase.
#############################################################################
IS_PARALLEL_RUNNER = False
RANK_ID = 0
def parallel_args():
    """Parse args for parallel runner."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry')
    parser.add_argument('--entry_kwargs', type=json.loads)
    args = parser.parse_args()
    return args

def parallel_runner(args):
    """Init parallel context, and call the entry function."""
    from mindspore import communication as mscomm
    from mindformers.parallel_core.inference.parallel_state import initialize_model_parallel
    mscomm.init()
    initialize_model_parallel(tensor_model_parallel_size=2, order='tp-ep-dp-pp')
    global IS_PARALLEL_RUNNER
    global RANK_ID
    IS_PARALLEL_RUNNER = True
    RANK_ID = mscomm.get_rank()
    globals()[args.entry](**args.entry_kwargs)

def invoke_parallel(entry_func, **entry_kwargs):
    """Start 2 parallel workers, and call the entry function."""
    from tests.st.test_utils import get_available_port
    run_file = os.path.abspath(__file__)
    port = get_available_port()
    os.system(f'kill -9 $(lsof -i:{port} | ' + "awk '{print $2}')")
    return_code = os.system(
        f'msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 '
        f'--master_port={port} --join=True --log_dir=./test_osl_logs '
        f'python {run_file} --entry {entry_func.__name__} --entry_kwargs {repr(json.dumps(entry_kwargs))}'
    )
    if return_code != 0:
        for i in os.listdir('test_osl_logs'):
            if i.endswith('.log'):
                filepath = os.path.join('test_osl_logs', i)
                with open(filepath, 'r', encoding='utf-8') as f:
                    print(f'===================={filepath}====================')
                    print(f.read())
    os.system(f'kill -9 $(lsof -i:{port} | ' + "awk '{print $2}')")
    os.system('rm -rf test_osl_logs')
    assert return_code == 0


#############################################################################
# Utility functions
#############################################################################
class SimpleNet(nn.Cell):
    """
    Network with single GroupedMatmul linear
    """
    class DecoderCell(nn.Cell):
        """decoder cell"""
        def __init__(self, linear, tp_group):
            super().__init__()
            self.linear = linear
            self.tp_group = tp_group
            self.scatter_to_mp_region = isinstance(linear, RowParallelLinear)

        def construct(self, x, *args, **kwargs):
            """linear"""
            if self.scatter_to_mp_region:
                x = scatter_to_model_parallel_region(x, self.tp_group)
            return self.linear(x, *args, **kwargs)

    def __init__(self, linear_type, dtype_str, foo_seq_length=1024):
        super().__init__()

        dtype_map = {
            'float32': msdtype.float32,
            'bfloat16': msdtype.bfloat16,
        }
        dtype = dtype_map.get(dtype_str, None)
        if dtype is None:
            raise ValueError(f'Unsupported dtype: {dtype_str}')

        self.config = TransformerConfig(
            num_attention_heads=1,
            num_layers=1,
            params_dtype=dtype_str,
        )
        self.foo_seq_length = foo_seq_length
        tp_group = get_tensor_model_parallel_group() if is_initialized() else default_pgs
        if linear_type == 'ColumnParallelLinear':
            linear = ColumnParallelLinear(
                foo_seq_length, foo_seq_length,
                compute_dtype=dtype,
                config=self.config,
                bias=False,
                tp_group=tp_group
            )
        elif linear_type == 'RowParallelLinear':
            linear = RowParallelLinear(
                foo_seq_length, foo_seq_length,
                compute_dtype=dtype,
                config=self.config,
                bias=False,
                tp_group=tp_group
            )
        elif linear_type == 'ReplicatedLinear':
            linear = ReplicatedLinear(
                foo_seq_length, foo_seq_length,
                compute_dtype=dtype,
                config=self.config,
                bias=False
            )
        else:
            raise ValueError(f'Unsupported linear type: {linear_type}')

        linear.weight.set_data(msops.rand_like(linear.weight, seed=42))
        self.decoder = SimpleNet.DecoderCell(linear, tp_group)

    def construct(self, x):
        """decoder"""
        return self.decoder(x)

    # pylint: disable=unused-argument
    def generate(self, input_ids, do_sample=False, max_new_tokens=1):
        input_ids = Tensor(input_ids)
        input_ids = ms.ops.pad(input_ids, (0, self.foo_seq_length - input_ids.shape[1]), value=0)
        return self.construct(input_ids.astype(msdtype.bfloat16))

def create_ptq(mode):
    """Returns a PTQ instance with OSL config."""
    cfg = PTQConfig(mode=mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                    act_quant_dtype=msdtype.int8, outliers_suppression=OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE)
    ptq = PTQ(config=cfg)
    # pylint: disable=protected-access
    ptq._config.always_use_fp_input_in_processer = True
    ptq._config.skip_offload_in_processing = True
    ptq._config.algorithm_cache_path = {} # Disable cache for testing
    ptq._config.experimental = True
    ptq._config.fake_quant = True
    ptq.decoder_layer_types.append(SimpleNet.DecoderCell)
    return ptq

def create_dataset(dataset_len):
    """Create a dataset for testing."""
    return GeneratorDataset(
        (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32) for _ in range(dataset_len)),
        column_names=['input_ids'])

def get_save_file_name(save_name):
    """Get the save file name with parallel rank ids."""
    if IS_PARALLEL_RUNNER:
        return f'rank{RANK_ID}_{save_name}'
    return save_name

def quant_net(linear_type, dtype):
    """Quantize: Saves quantized weight to ./osl-quant.ckpt, and returns the original float point output."""
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = 'on'
    os.environ['ENFORCE_EAGER'] = 'true'
    os.environ["RUN_MODE"] = "predict"
    os.environ['MS_ENABLE_LCCL'] = 'off'
    ascend_path = os.environ.get('ASCEND_HOME_PATH', '')
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = '/usr/local/Ascend/latest'

    network = SimpleNet(linear_type, dtype, 1024)
    dataset = create_dataset(10)
    fp_output = [network.generate(i['input_ids']) for i in dataset.create_dict_iterator(output_numpy=True)]

    ms.set_context(mode=ms.PYNATIVE_MODE, jit_config={'jit_level': 'O0', 'infer_boost': 'on'})
    ptq = create_ptq(PTQMode.QUANTIZE)
    network = ptq.apply(network, datasets=dataset)
    network = ptq.convert(network)
    ms.save_checkpoint(network.parameters_dict(), get_save_file_name('osl-quant.ckpt'),
                       choice_func=lambda x: all(i not in x for i in ['key_cache', 'value_cache', 'float_weight']))
    return fp_output

def infer_net(linear_type, dtype):
    """Infer: Load quantized weight from ./osl-quant.ckpt, and returns inference output."""
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = 'on'
    os.environ['MS_INTERNAL_ENABLE_CUSTOM_KERNAL_LIST'] = 'QbmmAllReduceAdd,QbmmAdd'
    os.environ['MS_ENABLE_LCCL'] = 'off'
    os.environ.pop('ENFORCE_EAGER', None)
    ascend_path = os.environ.get('ASCEND_HOME_PATH', '')
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = '/usr/local/Ascend/latest'

    network = SimpleNet(linear_type, dtype, 1024)
    dataset = create_dataset(10)

    ms.set_context(mode=ms.GRAPH_MODE, jit_config={'jit_level': 'O0', 'infer_boost': 'on'})
    ptq = create_ptq(PTQMode.DEPLOY)
    network = ptq.fake_quant(network)
    param_dict = ms.load_checkpoint(get_save_file_name('osl-quant.ckpt'))
    ms.load_param_into_net(network, param_dict)
    qoutput = [network.generate(i['input_ids']) for i in dataset.create_dict_iterator(output_numpy=True)]
    return qoutput

def _test_simple_net(linear_type, dtype):
    """Test procedure: Quantize and evaluate one SimpleNet with one Decoder layer, including one given linear cell."""
    # pylint: disable=protected-access
    MFModelHubPlugin()._load_quant_cells()
    MFModelHubPlugin()._load_algo_modules()
    fpoutput = quant_net(linear_type, dtype)
    qoutput = infer_net(linear_type, dtype)
    os.remove(get_save_file_name('osl-quant.ckpt'))
    for i, (fpout, qout) in enumerate(zip(fpoutput, qoutput)):
        fpout = fpout.astype(msdtype.float32)
        qout = qout.astype(msdtype.float32)
        cos_sim = ms.ops.mean(ms.ops.cosine_similarity(fpout, qout))
        assert cos_sim > 0.99, f'Sample {i} output cos similarity is {cos_sim}, fpout={fpout}, qout={qout}'


#############################################################################
# Testcases
#############################################################################
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
@pytest.mark.parametrize('linear_type', ['ReplicatedLinear'])
@pytest.mark.parametrize('dtype', ['bfloat16'])
def test_parallel_linear(linear_type, dtype):
    """
    Feature: Quantize and evaluate one SimpleNet with one Decoder layer, including one ParallelLinear cell.
        Work on two cards in parallel mode.
    Description: Quantize and evaluate one SimpleNet with PTQ algorithm using OSL outlier suppression.
    Expectation: Cos similarity between original float-point and quantized results is supposed to be greater than 99%.
    """
    invoke_parallel(_test_simple_net, linear_type=linear_type, dtype=dtype)


if __name__ == '__main__':
    uargs = parallel_args()
    parallel_runner(uargs)
