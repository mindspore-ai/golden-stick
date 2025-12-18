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
import os
import sys
import shutil

import numpy as np
from datasets import Dataset
import mindspore as ms
from mindspore import dtype as msdtype

from mindspore_gs.ptq import PTQConfig
from mixed_precision_network import MixedPrecisionMindOneNetwork
from layer_policies_loader import create_layer_policies_for_mindone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../../")))
from tests.st.st0.ptq.networks.custom.linear_info_loader import load_linear_info_from_config
from tests.st.st0.ptq.networks.custom.common_utils import (
    create_test_input, create_linear_ds, convert_to_tensor)
from tests.st.precision_utils import PrecisionChecker

def quant_net(hidden_size, linear_specs):
    """Quantize network: save quantized weights to ./mixed_precision_quant.ckpt, return original float output"""
    np.random.seed(42)
    ms.set_seed(42)

    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = 'on'
    os.environ['MS_ENABLE_LCCL'] = 'off'
    ascend_path = os.environ.get('ASCEND_HOME_PATH', '')
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = '/usr/local/Ascend/latest'

    dataset = create_linear_ds(
        batch_size=2,
        seq_length=hidden_size,
        repeat=10,
        is_parallel=False
    )
    samples = [sample['input_ids'].asnumpy() \
        for sample in dataset.create_dict_iterator()]
    dataset = Dataset.from_dict({"input_ids": samples})
    dataset.set_transform(convert_to_tensor)

    model = MixedPrecisionMindOneNetwork(linear_specs=linear_specs)

    test_input = create_test_input(hidden_size)
    fp_outputs_dict = model.network.get_outputs_dict(test_input)

    ms.set_context(device_target="Ascend",
                   mode=ms.PYNATIVE_MODE,
                   jit_config={"jit_level": "O0", "infer_boost": "on"},
                   deterministic="ON")
    ms.set_seed(42)
    base_config = PTQConfig(
        weight_quant_dtype=msdtype.int8,
        opname_blacklist=['pre_layer']
    )

    layer_policies = create_layer_policies_for_mindone(linear_specs)

    try:
        # pylint: disable=import-outside-toplevel,protected-access
        from mindspore_gs.ptq.plugins import MindOneModelHubPlugin
        MindOneModelHubPlugin()._load_quant_cells()
        MindOneModelHubPlugin()._load_algo_modules()
    except ImportError:
        pass

    model.calibrate(ptq_config=base_config, layers_policy=layer_policies, datasets=dataset)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(curr_dir, 'mo_mixed_precision_quant')
    os.makedirs(save_path, exist_ok=True)
    ms.save_checkpoint(model.network.parameters_dict(),
                       os.path.join(save_path, 'mo_mixed_precision_quant.safetensors'),
                       format='safetensors')
    return fp_outputs_dict


def infer_net(hidden_size, linear_specs):
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

    model = MixedPrecisionMindOneNetwork(linear_specs=linear_specs)

    ms.set_context(device_target="Ascend",
                   mode=ms.PYNATIVE_MODE,
                   jit_config={"jit_level": "O0", "infer_boost": "on"},
                   deterministic="ON")
    base_config = PTQConfig(
        weight_quant_dtype=msdtype.int8,
        opname_blacklist=['pre_layer']
    )
    layer_policies = create_layer_policies_for_mindone(linear_specs)

    try:
        # pylint: disable=import-outside-toplevel,protected-access
        from mindspore_gs.ptq.plugins import MindOneModelHubPlugin
        MindOneModelHubPlugin()._load_quant_cells()
        MindOneModelHubPlugin()._load_algo_modules()
    except ImportError:
        pass

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(curr_dir, 'mo_mixed_precision_quant')
    model.fake_quant(base_config, layer_policies, save_path)

    test_input = create_test_input(hidden_size)
    qoutputs_dict = model.network.get_outputs_dict(test_input)
    return qoutputs_dict


def main():
    """Main function to run mixed precision PTQ test"""
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(curr_dir, 'linear_specs_config.yaml')
    model_config, precision_thd, linear_specs = \
        load_linear_info_from_config(config_path=config_path)

    fp_outputs_dict = quant_net(model_config['input_size'], linear_specs)
    qoutputs_dict = infer_net(model_config['input_size'], linear_specs)

    # Save outputs to npz file
    for layer_idx, linear_spec in enumerate(linear_specs):
        fp_output = fp_outputs_dict[layer_idx]
        qoutput = qoutputs_dict[layer_idx]

        fp_output_np = fp_output.asnumpy() if hasattr(fp_output, 'asnumpy') else fp_output
        qoutput_np = qoutput.asnumpy() if hasattr(qoutput, 'asnumpy') else qoutput

        fp_output_np = fp_output_np.astype(np.float32)
        qoutput_np = qoutput_np.astype(np.float32)

        key = (linear_spec.linear_type,
               linear_spec.compute_dtype,
               linear_spec.quant_policy)
        checker = PrecisionChecker(cos_sim_thd=precision_thd[key]['cos_sim_thd'],
                                   l1_norm_thd=precision_thd[key]['l1_norm_thd'],
                                   kl_dvg_thd=precision_thd[key]['kl_dvg_thd'])
        succeed = checker.check_precision(fp_output_np, qoutput_np)

        layer_info = f" {layer_idx}-{key[0]}-{key[1]}-{key[2]}"
        assert succeed, f"layer {layer_info} check failed!"
        print(f"Precision check for layer {layer_info} passed!")

    # Clean up checkpoint file
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_file = os.path.join(curr_dir, 'mo_mixed_precision_quant')
    if os.path.exists(ckpt_file):
        shutil.rmtree(ckpt_file)


if __name__ == "__main__":
    main()
