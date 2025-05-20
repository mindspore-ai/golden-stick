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
import argparse
import json
import pytest
import numpy as np
import mindspore as ms
from mindspore import dtype as msdtype
from mindspore import nn, Tensor
from mindspore.dataset import GeneratorDataset
from mindformers.modules.layers import Linear
from mindformers.experimental.infer.core.layers import ColumnParallelLinear, RowParallelLinear
from mindformers.parallel_core.inference.tensor_parallel.mappings import ScatterToModelParallelRegion
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQ, PTQConfig, PTQMode, OutliersSuppressionType

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
                with open(filepath, 'r') as f:
                    print(f'===================={filepath}====================')
                    print(f.read())
    os.system("ps -u | grep 'test_osl' | grep -v grep | awk -F ' ' '{print$2}' | xargs kill -9")
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
        def __init__(self, linear):
            super().__init__()
            self.linear = linear
            if isinstance(linear, RowParallelLinear):
                self.scatter_to_mp_region = ScatterToModelParallelRegion()
            else:
                self.scatter_to_mp_region = None

        def construct(self, x, *args, **kwargs):
            """linear"""
            if self.scatter_to_mp_region is not None:
                x = self.scatter_to_mp_region(x)
            return self.linear(x, *args, **kwargs)

    class ParallelConfig(nn.Cell):
        """ParallelConfig"""
        def __init__(self):
            super().__init__()
            self.use_sequence_parallel = False

    def __init__(self, linear_type, is_expert, foo_seq_length=1024):
        assert not (linear_type == 'Linear' and is_expert), 'expert gmm is not supported for Linear'
        super(SimpleNet, self).__init__()
        self.config = SimpleNet.ParallelConfig()
        self.is_expert = is_expert
        self.foo_seq_length = foo_seq_length
        if linear_type == 'ColumnParallelLinear':
            linear = ColumnParallelLinear(
                foo_seq_length, foo_seq_length,
                config=self.config,
                bias=False,
                param_init_type=msdtype.bfloat16,
                compute_dtype=msdtype.bfloat16,
                is_expert=is_expert,
                expert_num=10 if is_expert else 1
            )
        elif linear_type == 'RowParallelLinear':
            linear = RowParallelLinear(
                foo_seq_length, foo_seq_length,
                config=self.config,
                input_is_parallel=True,
                bias=False,
                param_init_type=msdtype.bfloat16,
                compute_dtype=msdtype.bfloat16,
                is_expert=is_expert,
                expert_num=10 if is_expert else 1
            )
        elif linear_type == 'Linear':
            linear = Linear(
                foo_seq_length, foo_seq_length,
                has_bias=False,
                param_init_type=msdtype.bfloat16,
                compute_dtype=msdtype.bfloat16,
            )

        self.decoder = SimpleNet.DecoderCell(linear)
        self.group_list = Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 2], dtype=msdtype.int64)

    def construct(self, x):
        """decoder"""
        if not self.is_expert:
            return self.decoder(x)
        return self.decoder(x, group_list=self.group_list)

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
    ptq._config.algorithm_cache_path = '' # Disable cache for testing
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

def quant_net(linear_type, is_expert):
    """Quantize: Saves quantized weight to ./osl-quant.ckpt, and returns the original float point output."""
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = 'on'
    os.environ['FORCE_EAGER'] = 'true'
    os.environ["RUN_MODE"] = "predict"
    ascend_path = os.environ.get('ASCEND_HOME_PATH', '')
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = '/usr/local/Ascend/latest'

    network = SimpleNet(linear_type, is_expert, 1024)
    dataset = create_dataset(10)
    fp_output = [network.generate(i['input_ids']) for i in dataset.create_dict_iterator(output_numpy=True)]

    ms.set_context(mode=ms.PYNATIVE_MODE, jit_config={'jit_level': 'O0', 'infer_boost': 'on'})
    ptq = create_ptq(PTQMode.QUANTIZE)
    network = ptq.apply(network, datasets=dataset)
    network = ptq.convert(network)
    ms.save_checkpoint(network.parameters_dict(), get_save_file_name('osl-quant.ckpt'),
                       choice_func=lambda x: all(i not in x for i in ['key_cache', 'value_cache', 'float_weight']))
    return fp_output

def infer_net(linear_type, is_expert):
    """Infer: Load quantized weight from ./osl-quant.ckpt, and returns inference output."""
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = 'on'
    os.environ['MS_INTERNAL_ENABLE_CUSTOM_KERNAL_LIST'] = 'QbmmAllReduceAdd,QbmmAdd'
    os.environ.pop('FORCE_EAGER', None)
    ascend_path = os.environ.get('ASCEND_HOME_PATH', '')
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = '/usr/local/Ascend/latest'

    network = SimpleNet(linear_type, is_expert, 1024)
    dataset = create_dataset(10)

    ms.set_context(mode=ms.GRAPH_MODE, jit_config={'jit_level': 'O0', 'infer_boost': 'on'})
    ptq = create_ptq(PTQMode.DEPLOY)
    network = ptq.apply(network, datasets=dataset)
    network = ptq.convert(network)
    param_dict = ms.load_checkpoint(get_save_file_name('osl-quant.ckpt'))
    ms.load_param_into_net(network, param_dict)
    qoutput = [network.generate(i['input_ids']) for i in dataset.create_dict_iterator(output_numpy=True)]
    return qoutput

def _test_simple_net(linear_type, is_expert):
    """Test procedure: Quantize and evaluate one SimpleNet with one Decoder layer, including one given linear cell."""
    fpoutput = quant_net(linear_type, is_expert)
    qoutput = infer_net(linear_type, is_expert)
    for i, (fpout, qout) in enumerate(zip(fpoutput, qoutput)):
        fpout = fpout.astype(msdtype.float32)
        qout = qout.astype(msdtype.float32)
        cos_sim = ms.ops.mean(ms.ops.cosine_similarity(fpout, qout))
        assert cos_sim > 0.99, f'Sample {i} output cos similarity is {cos_sim}, fpout={fpout}, qout={qout}'
    os.remove(get_save_file_name('osl-quant.ckpt'))


#############################################################################
# Testcases
#############################################################################
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_linear():
    """
    Feature: Quantize and evaluate one SimpleNet with one Decoder layer, including one Linear cell.
    Description: Quantize and evaluate one SimpleNet with PTQ algorithm using OSL outlier suppression.
        is_expert is set to False since Linear cell does not come with expert gmm in real networks.
        Work on one single card.
    Expectation: Cos similarity between original float-point and quantized results is supposed to be greater than 99%.
    """
    _test_simple_net('Linear', False)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
@pytest.mark.parametrize('linear_type', ['RowParallelLinear', 'ColumnParallelLinear'])
@pytest.mark.parametrize('is_expert', [True, False])
def test_parallel_linear(linear_type, is_expert):
    """
    Feature: Quantize and evaluate one SimpleNet with one Decoder layer, including one ParallelLinear cell.
        is_expert might can be enabled for routed experts GMM.
        Work on two cards in parallel mode.
    Description: Quantize and evaluate one SimpleNet with PTQ algorithm using OSL outlier suppression.
    Expectation: Cos similarity between original float-point and quantized results is supposed to be greater than 99%.
    """
    if linear_type == 'RowParallelLinear' and is_expert:
        pytest.skip('RowParallelLinear with is_expert=True is not supported.') # FIXME
        return
    invoke_parallel(_test_simple_net, linear_type=linear_type, is_expert=is_expert)


if __name__ == '__main__':
    uargs = parallel_args()
    parallel_runner(uargs)
