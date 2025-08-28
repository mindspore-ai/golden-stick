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
"""test interfaces of ptq."""


from collections import OrderedDict
from typing import Optional
import os
import time
import argparse
from safetensors import safe_open
import pytest

from mindspore import dtype as msdtype
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import (PTQConfig, PTQMode, OutliersSuppressionType,
                              PrecisionRecovery, QuantGranularity, GPTQQuantConfig)
from tests.st.test_utils import get_available_port
from ptq_model_tester import PTQModelTester


class DeepSeekV3Tester(PTQModelTester):
    """PTQModelTester"""
    def create_ptq_config(self, quant_type: str):
        """create_ptq"""
        if quant_type.lower() == 'a8w8':
            cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                            act_quant_dtype=msdtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH,
                            opname_blacklist=['output_layer', 'linear_fc2', 'kv_up_proj'])
            layer_policies = OrderedDict()
        elif quant_type.lower() == 'a8w4':
            cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                            act_quant_dtype=msdtype.int8,
                            outliers_suppression=OutliersSuppressionType.SMOOTH,
                            opname_blacklist=['output_layer', 'kv_up_proj'], weight_clip=False)
            mlp_config = PTQConfig(backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                                   act_quant_dtype=msdtype.int8,
                                   outliers_suppression=OutliersSuppressionType.NONE,
                                   precision_recovery=PrecisionRecovery.NONE,
                                   act_quant_granularity=QuantGranularity.PER_TOKEN,
                                   weight_quant_granularity=QuantGranularity.PER_CHANNEL,
                                   weight_clip=False)
            gptq_config = GPTQQuantConfig(static_groups=True, desc_act=True)
            moe_cfg = PTQConfig(backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.qint4x2,
                                act_quant_dtype=msdtype.int8, act_quant_granularity=QuantGranularity.PER_TOKEN,
                                weight_quant_granularity=QuantGranularity.PER_GROUP, group_size=256,
                                algo_args=gptq_config, precision_recovery=PrecisionRecovery.GPTQ, weight_clip=False)
            layer_policies = OrderedDict({r'.*\.mlp\.linear_fc1.*': mlp_config,
                                          r'.*\.mlp\.linear_fc2.*': mlp_config,
                                          r'.*\.mlp\.shared_experts\.linear_fc1.*': mlp_config,
                                          r'.*\.mlp\.shared_experts\.linear_fc2.*': mlp_config,
                                          r'.*\.mlp\.experts\.linear_fc1.*': moe_cfg,
                                          r'.*\.mlp\.experts\.linear_fc2.*': moe_cfg})
        else:
            raise RuntimeError(f'Input unsupported quant type: {quant_type}.')
        return cfg, layer_policies

    # pylint: disable=unused-argument
    def check_quant_description(self, quant_ckpt_path, quant_type) -> bool:
        "quant_type_description"
        return True

    def get_ds_acc_threshold(self, quant_type) -> Optional[float]:
        score_mapping = {
            "A8W8": 0.41,
        }
        return score_mapping.get(quant_type)

    def _load_file(self, quant_ckpt_path, rank_num):
        """load_file"""
        if not os.path.exists(quant_ckpt_path):
            raise ValueError(f"{quant_ckpt_path} not exists.")

        # load safetensors files
        files = []
        for file in (f"{quant_ckpt_path}/rank_{i}/quant.safetensors" for i in range(rank_num)):
            files.append(safe_open(file, framework="np"))
        files = tuple(files)
        param_keys = files[0].keys()
        return files, param_keys

    def _check_qkv_split(self, files, param_keys):
        """check_qkv_split"""
        layer_prefix = 'model.layers.0.self_attn.'
        layer_names = ['q_a_proj.',
                       'q_b_proj.',
                       'kv_a_proj_with_mqa.']
        param_names = ['weight',
                       'weight_scale',
                       'weight_offset',
                       'input_scale',
                       'input_offset',
                       'smooth_scale',
                       'deq_scale',
                       'quant_bias']
        for layer_name in layer_names:
            for param_name in param_names:
                param_full_name = layer_prefix + layer_name + param_name
                if param_full_name not in param_keys:
                    raise ValueError(f"{param_full_name} not in unify safetensors.")
                if 'q_a' in param_full_name:
                    if param_full_name.endswith("weight"):
                        assert files[0].get_tensor(param_full_name).shape == (1536, 7168), \
                            f"{param_full_name} error, expect (1536, 7168)"
                    if param_full_name.endswith("weight_scale"):
                        assert files[0].get_tensor(param_full_name).shape == (1536,), \
                            f"{param_full_name} error, expect (1536,),"
                    if param_full_name.endswith("weight_offset"):
                        assert files[0].get_tensor(param_full_name).shape == (1536,), \
                            f"{param_full_name} error, expect (1536,),"
                    if param_full_name.endswith("input_scale"):
                        assert files[0].get_tensor(param_full_name).shape == (7168,), \
                            f"{param_full_name} error, expect (7168,)"
                    if param_full_name.endswith("input_offset"):
                        assert files[0].get_tensor(param_full_name).shape == (7168,), \
                            f"{param_full_name} error, expect (7168,)"
                    if param_full_name.endswith("smooth_scale"):
                        assert files[0].get_tensor(param_full_name).shape == (7168,), \
                            f"{param_full_name} error, expect (7168,)"
                    if param_full_name.endswith("deq_scale"):
                        assert files[0].get_tensor(param_full_name).shape == (1536,), \
                            f"{param_full_name} error, expect (1536,),"
                    if param_full_name.endswith("quant_bias"):
                        assert files[0].get_tensor(param_full_name).shape == (1536,), \
                            f"{param_full_name} error, expect (1536,),"
                elif 'q_b' in param_full_name:
                    if param_full_name.endswith("weight"):
                        assert files[0].get_tensor(param_full_name).shape == (6144, 1536), \
                            f"{param_full_name} error, expect (6144, 1536)"
                elif 'kv_a' in param_full_name:
                    if param_full_name.endswith("weight"):
                        assert files[0].get_tensor(param_full_name).shape == (576, 7168), \
                            f"{param_full_name} error, expect (576, 7168)"
                else:
                    raise ValueError(f"{param_full_name} is not expected.")

    def _check_ffn_split(self, files, param_keys):
        """check_ffn_split"""
        layer_prefix = 'model.layers.0.mlp.'
        layer_names = ['gate_proj.',
                       'down_proj.',
                       'up_proj.']
        param_names = ['weight',
                       'weight_scale',
                       'weight_offset']
        for layer_name in layer_names:
            for param_name in param_names:
                param_full_name = layer_prefix + layer_name + param_name
                if param_full_name not in param_keys:
                    raise ValueError(f"{param_full_name} not in unify safetensors.")
                if 'gate' in param_full_name:
                    if param_full_name.endswith("weight"):
                        assert files[0].get_tensor(param_full_name).shape == (4608, 7168), \
                            f"{param_full_name} error, expect (4608, 7168)"
                    if param_full_name.endswith("weight_scale"):
                        assert files[0].get_tensor(param_full_name).shape == (4608,), \
                            f"{param_full_name} error, expect (4608,)"
                    if param_full_name.endswith("weight_offset"):
                        assert files[0].get_tensor(param_full_name).shape == (4608,), \
                            f"{param_full_name} error, expect (4608,)"
                elif 'down' in param_full_name:
                    if param_full_name.endswith("weight"):
                        assert files[0].get_tensor(param_full_name).shape == (7168, 4608), \
                            f"{param_full_name} error, expect (7168, 4608)"
                elif 'up' in param_full_name:
                    if param_full_name.endswith("weight"):
                        assert files[0].get_tensor(param_full_name).shape == (4608, 7168), \
                            f"{param_full_name} error, expect (4608, 7168)"
                else:
                    raise ValueError(f"{param_full_name} is not expected.")

    def _check_moe_split(self, files, param_keys):
        """check_moe_split"""
        layer_prefix = 'model.layers.3.mlp.experts.'
        layer_names = ['gate_proj.',
                       'down_proj.',
                       'up_proj.']
        param_names = ['weight',
                       'weight_scale',
                       'weight_offset']
        for layer_name in layer_names:
            for param_name in param_names:
                experts_dict = [k for k in param_keys if layer_prefix in k \
                                and layer_name in k and k.endswith(param_name)]
                assert len(experts_dict) == 256, \
                    f"The number of {layer_prefix}x.{layer_name}{param_name} should be 256, \
                    but got {len(experts_dict)}"

                param_full_name = layer_prefix + '0.' +  layer_name + param_name
                if param_full_name not in param_keys:
                    raise ValueError(f"{param_full_name} not in unify safetensors.")

                if 'gate' in param_full_name:
                    if param_full_name.endswith("weight"):
                        assert files[0].get_tensor(param_full_name).shape == (512, 7168), \
                            f"{param_full_name} error, expect (512, 7168)"
                    if param_full_name.endswith("weight_scale"):
                        assert files[0].get_tensor(param_full_name).shape == (512, 28), \
                            f"{param_full_name} error, expect (512, 28)"
                    if param_full_name.endswith("weight_offset"):
                        assert files[0].get_tensor(param_full_name).shape == (512, 28), \
                            f"{param_full_name} error, expect (512, 28)"
                elif 'down' in param_full_name:
                    if param_full_name.endswith("weight"):
                        assert files[0].get_tensor(param_full_name).shape == (7168, 512), \
                            f"{param_full_name} error, expect (7168, 512)"
                elif 'up' in param_full_name:
                    if param_full_name.endswith("weight"):
                        assert files[0].get_tensor(param_full_name).shape == (512, 7168), \
                            f"{param_full_name} error, expect (512, 7168)"
                else:
                    raise ValueError(f"{param_full_name} is not expected.")

    def check_safetensor_split(self, quant_ckpt_path, rank_num):
        """check_safetensor_split"""
        files, param_keys = self._load_file(quant_ckpt_path, rank_num)
        print("checking qkv split...")
        self._check_qkv_split(files, param_keys)
        print("checking moe split...")
        self._check_moe_split(files, param_keys)
        print("checking ffn split...")
        self._check_ffn_split(files, param_keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quant_algo', '-a', type=str, required=True)
    uargs = parser.parse_args()
    input_quant_algo = uargs.quant_algo

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    calibrate_config_path = os.path.join(cur_dir, "calibrate_deepseek3_671b.yaml")
    infer_config_path = os.path.join(cur_dir, "predict_deepseek3_671b.yaml")
    q_ckpt_path = os.path.join(cur_dir, f"dsv3-quant-4p-{input_quant_algo}")
    dataset_path = os.path.join(cur_dir, '/nfs/dataset/workspace/mindspore_dataset/ceval/dev')
    tester = DeepSeekV3Tester()
    tester.quant_model(calibrate_config_path, q_ckpt_path, input_quant_algo, dataset_path)
    if uargs.quant_algo.lower() == 'a8w4':
        tester.check_safetensor_split(q_ckpt_path, 4)

def ptq_predict_2stage_4p_run(quant_algo):
    """
    Feature: test dynamic quant adjust parameter in two stages with two cards.
    Description: apply ptq on deepseek-v3/r1 and check accuracy.
    Expectation: accuracy is good.
    """
    os.environ['quant_algo'] = f"{quant_algo}"
    os.environ['HCCL_CONNECT_TIMEOUT'] = "1800"
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_dsv3_accuracy.py")
    port = get_available_port()
    os.system(f"kill -9 $(lsof -i:{port} | " + "awk '{print $2}')")
    time.sleep(1.0)
    return_code = os.system(
        f"msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 "
        f"--master_port={port} --join=True --log_dir=./test_ptq_{quant_algo}_predict_dsv3_4p_logs "
        f"python {run_file} -a {quant_algo}"
    )
    os.system("ps -u | grep 'test_dsv3_accuracy' | grep -v grep | awk -F ' ' '{print$2}' | xargs kill -9")
    os.system(f"kill -9 $(lsof -i:{port} | " + "awk '{print $2}')")
    time.sleep(1.0)
    assert return_code == 0


@pytest.mark.level2
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_ptq_dsv3_a8w8_accuracy():
    """
    Feature: test omni quant adjust parameter in two stages with two cards.
    Description: apply A8W8 on deepseek-v3/r1 and check score.
    Expectation: score is good.
    """
    ptq_predict_2stage_4p_run("A8W8")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_ptq_dsv3_a8w4_accuracy():
    """
    Feature: test omni quant adjust parameter in two stages with two cards.
    Description: apply A8W4 on deepseek-v3/r1 and check score.
    Expectation: score is good.
    """
    ptq_predict_2stage_4p_run("A8W4")
