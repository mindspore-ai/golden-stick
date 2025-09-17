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
import shutil
from safetensors import safe_open
import pytest

from mindspore import dtype as msdtype
from mindspore.communication import get_rank
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import (PTQConfig, PTQMode, OutliersSuppressionType,
                              PrecisionRecovery, QuantGranularity, GPTQQuantConfig)
from tests.st.test_utils import get_available_port
from ptq_model_tester import PTQModelTester


class DeepSeekV3Tester(PTQModelTester):
    """PTQModelTester"""
    def create_ptq_config(self):
        """create_ptq"""
        smoothquant_cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND,
                                    weight_quant_dtype=msdtype.int8, act_quant_dtype=msdtype.int8,
                                    outliers_suppression=OutliersSuppressionType.SMOOTH,
                                    opname_blacklist=['output_layer', 'linear_fc2', 'kv_up_proj'])
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
                                      r'.*\.mlp\.experts\.linear_fc2.*': moe_cfg,
                                      'not match': smoothquant_cfg})
        return cfg, layer_policies

    # pylint: disable=unused-argument
    def check_quant_description(self, quant_ckpt_path) -> bool:
        "quant_type_description"
        return True

    # pylint: disable=unused-argument
    def get_ds_acc_threshold(self) -> Optional[float]:
        return 0.41

    def _load_file(self, quant_ckpt_path):
        """load_file"""
        if not os.path.exists(quant_ckpt_path):
            raise ValueError(f"{quant_ckpt_path} not exists.")

        # load safetensors files
        rank_id = get_rank()
        filename = f"{quant_ckpt_path}/rank_{rank_id}/quant.safetensors"
        file = safe_open(filename, framework="np")
        param_keys = file.keys()
        return file, param_keys

    def _check_qkv_split(self, file, param_keys):
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
                        assert file.get_tensor(param_full_name).shape == (1536, 7168), \
                            f"{param_full_name} error, expect (1536, 7168)"
                    if param_full_name.endswith("weight_scale"):
                        assert file.get_tensor(param_full_name).shape == (1536,), \
                            f"{param_full_name} error, expect (1536,),"
                    if param_full_name.endswith("weight_offset"):
                        assert file.get_tensor(param_full_name).shape == (1536,), \
                            f"{param_full_name} error, expect (1536,),"
                    if param_full_name.endswith("input_scale"):
                        assert file.get_tensor(param_full_name).shape == (7168,), \
                            f"{param_full_name} error, expect (7168,)"
                    if param_full_name.endswith("input_offset"):
                        assert file.get_tensor(param_full_name).shape == (7168,), \
                            f"{param_full_name} error, expect (7168,)"
                    if param_full_name.endswith("smooth_scale"):
                        assert file.get_tensor(param_full_name).shape == (7168,), \
                            f"{param_full_name} error, expect (7168,)"
                    if param_full_name.endswith("deq_scale"):
                        assert file.get_tensor(param_full_name).shape == (1536,), \
                            f"{param_full_name} error, expect (1536,),"
                    if param_full_name.endswith("quant_bias"):
                        assert file.get_tensor(param_full_name).shape == (1536,), \
                            f"{param_full_name} error, expect (1536,),"
                elif 'q_b' in param_full_name:
                    if param_full_name.endswith("weight"):
                        assert file.get_tensor(param_full_name).shape == (6144, 1536), \
                            f"{param_full_name} error, expect (6144, 1536)"
                elif 'kv_a' in param_full_name:
                    if param_full_name.endswith("weight"):
                        assert file.get_tensor(param_full_name).shape == (576, 7168), \
                            f"{param_full_name} error, expect (576, 7168)"
                else:
                    raise ValueError(f"{param_full_name} is not expected.")

    def _check_ffn_split(self, file, param_keys):
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
                        assert file.get_tensor(param_full_name).shape == (4608, 7168), \
                            f"{param_full_name} error, expect (4608, 7168)"
                    if param_full_name.endswith("weight_scale"):
                        assert file.get_tensor(param_full_name).shape == (4608,), \
                            f"{param_full_name} error, expect (4608,)"
                    if param_full_name.endswith("weight_offset"):
                        assert file.get_tensor(param_full_name).shape == (4608,), \
                            f"{param_full_name} error, expect (4608,)"
                elif 'down' in param_full_name:
                    if param_full_name.endswith("weight"):
                        assert file.get_tensor(param_full_name).shape == (7168, 4608), \
                            f"{param_full_name} error, expect (7168, 4608)"
                elif 'up' in param_full_name:
                    if param_full_name.endswith("weight"):
                        assert file.get_tensor(param_full_name).shape == (4608, 7168), \
                            f"{param_full_name} error, expect (4608, 7168)"
                else:
                    raise ValueError(f"{param_full_name} is not expected.")

    def _check_moe_split(self, file, param_keys):
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
                        assert file.get_tensor(param_full_name).shape == (256, 7168), \
                            f"{param_full_name} error, expect (256, 7168)"
                    if param_full_name.endswith("weight_scale"):
                        assert file.get_tensor(param_full_name).shape == (512, 28), \
                            f"{param_full_name} error, expect (512, 28)"
                    if param_full_name.endswith("weight_offset"):
                        assert file.get_tensor(param_full_name).shape == (512, 28), \
                            f"{param_full_name} error, expect (512, 28)"
                elif 'down' in param_full_name:
                    if param_full_name.endswith("weight"):
                        assert file.get_tensor(param_full_name).shape == (3584, 512), \
                            f"{param_full_name} error, expect (3584, 512)"
                elif 'up' in param_full_name:
                    if param_full_name.endswith("weight"):
                        assert file.get_tensor(param_full_name).shape == (256, 7168), \
                            f"{param_full_name} error, expect (256, 7168)"
                else:
                    raise ValueError(f"{param_full_name} is not expected.")

    def check_safetensor_split(self, quant_ckpt_path):
        """check_safetensor_split"""
        file, param_keys = self._load_file(quant_ckpt_path)
        print("checking qkv split...")
        self._check_qkv_split(file, param_keys)
        print("checking moe split...")
        self._check_moe_split(file, param_keys)
        print("checking ffn split...")
        self._check_ffn_split(file, param_keys)


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    calibrate_config_path = os.path.join(cur_dir, "calibrate_deepseek3_671b.yaml")
    infer_config_path = os.path.join(cur_dir, "predict_deepseek3_671b.yaml")
    q_ckpt_path = os.path.join(cur_dir, f"dsv3-quant")
    dataset_path = os.path.join(cur_dir, '/nfs/dataset/workspace/mindspore_dataset/ceval/dev')
    tester = DeepSeekV3Tester()
    tester.quant_model(calibrate_config_path, q_ckpt_path, dataset_path, fake_quant=False)
    tester.check_safetensor_split(q_ckpt_path)
    try:
        print(f"mv: {q_ckpt_path} to: '/home/workspace/mindspore_dataset/weight'", flush=True)
        shutil.move(q_ckpt_path, "/home/workspace/mindspore_dataset/weight/dsv3-a8w4-quant")
    except (OSError, FileNotFoundError):
        pass


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_ptq_dsv3_mix_accuracy():
    """
    Feature: test omni quant adjust parameter in two stages with two cards.
    Description: apply mix-quant on deepseek-v3/r1 and check score.
    Expectation: score is good.
    """
    os.environ['HCCL_CONNECT_TIMEOUT'] = "1800"
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_dsv3_accuracy.py")
    port = get_available_port()
    os.system(f"kill -9 $(lsof -i:{port} | " + "awk '{print $2}')")
    time.sleep(1.0)
    return_code = os.system(
        f"msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 "
        f"--master_port={port} --join=True --log_dir=./test_ptq_predict_dsv3_8p_logs "
        f"python {run_file}"
    )
    time.sleep(1.0)
    assert return_code == 0
