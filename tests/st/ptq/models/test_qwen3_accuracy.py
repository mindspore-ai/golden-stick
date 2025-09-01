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
import json
import pytest
from mindspore import dtype as msdtype
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq.utils import QuantType

os.environ['GSLOG'] = "1"

from mindspore_gs.common import logger
from mindspore_gs.ptq import (PTQConfig, PTQMode,
                              OutliersSuppressionType,
                              GPTQQuantConfig,
                              PrecisionRecovery,
                              QuantGranularity)
from tests.st.test_utils import get_available_port
from ptq_model_tester import PTQModelTester


class QWen3Tester(PTQModelTester):
    """QWen3Tester"""
    def create_ptq_config(self, quant_type: str):
        """create_ptq"""
        if quant_type.lower() == 'a8w8':
            cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                            act_quant_dtype=msdtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH,
                            opname_blacklist=['output_layer', 'linear_fc2'])
            layer_policies = OrderedDict()
        elif quant_type.lower() == 'a8w4':
            gptq_config = GPTQQuantConfig(static_groups=True, desc_act=True)
            cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.qint4x2,
                            act_quant_dtype=msdtype.int8, act_quant_granularity=QuantGranularity.PER_TOKEN,
                            weight_quant_granularity=QuantGranularity.PER_GROUP, group_size=64,
                            algo_args=gptq_config, precision_recovery=PrecisionRecovery.GPTQ, weight_clip=True,
                            opname_blacklist=['output_layer', 'linear_fc2'])
            a8w8_cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                                 act_quant_dtype=msdtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH,
                                 opname_blacklist=['output_layer', 'linear_fc2'])
            layer_policies = OrderedDict({r'.*\.linear_proj\.*': a8w8_cfg,
                                          r'.*\.linear_fc1\.*': a8w8_cfg})
        elif quant_type.lower() == 'mix':
            a8w8_dynamic_cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND,
                                         weight_quant_dtype=msdtype.int8, act_quant_dtype=msdtype.int8,
                                         act_quant_granularity=QuantGranularity.PER_TOKEN,
                                         opname_blacklist=['output_layer'])
            a8w8_cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND,
                                 weight_quant_dtype=msdtype.int8, act_quant_dtype=msdtype.int8,
                                 act_quant_granularity=QuantGranularity.PER_TENSOR,
                                 outliers_suppression=OutliersSuppressionType.SMOOTH,
                                 opname_blacklist=['output_layer', 'linear_fc2'])
            a8w4_dynamic_cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND,
                                         weight_quant_dtype=msdtype.qint4x2, act_quant_dtype=msdtype.int8,
                                         weight_quant_granularity=QuantGranularity.PER_GROUP,
                                         group_size=64, algo_args=GPTQQuantConfig(static_groups=True, desc_act=True),
                                         act_quant_granularity=QuantGranularity.PER_TOKEN,
                                         precision_recovery=PrecisionRecovery.GPTQ, weight_clip=True,
                                         opname_blacklist=['output_layer'])
            cfg = a8w8_dynamic_cfg
            layer_policies = OrderedDict({r'.*\.1[0-4]\..*': a8w8_cfg,
                                          r'.*\.1[5-9]\..*': a8w8_cfg,
                                          'not_match': a8w4_dynamic_cfg
                                         })
        else:
            raise RuntimeError(f'Input unsupported quant type: {quant_type}.')
        return cfg, layer_policies

    def check_quant_description(self, quant_ckpt_path, quant_type) -> bool:
        "quant_type_description"
        if not os.path.exists(quant_ckpt_path):
            logger.error(f"{quant_ckpt_path} dose not exist.")
            return False
        desc_json_path = ""
        for file_name in os.listdir(quant_ckpt_path):
            if file_name.endswith(".json") and "quantization_description" in file_name:
                desc_json_path = os.path.join(quant_ckpt_path, file_name)
        if desc_json_path is None:
            logger.error("No quant description json file.")
            return False
        with open(desc_json_path, "r") as fp:
            desc_map = json.load(fp)

        def check(name, expect):
            cur = desc_map.get(name)
            ret = cur == expect
            if not ret:
                logger.error(f"quant info of {name} should be {expect}, but got: {cur}.")
            return ret

        if quant_type.lower() == "a8w8":
            check_map = {
                'model.decoder.layers.0.self_attention.linear_q.weight': QuantType.W8A8.value,
                'model.decoder.layers.0.self_attention.linear_k.weight': QuantType.W8A8.value,
                'model.decoder.layers.0.self_attention.linear_v.weight': QuantType.W8A8.value,
                'model.decoder.layers.1.self_attention.linear_proj.smooth_scale': QuantType.W8A8.value,
                'model.decoder.layers.2.self_attention.linear_q.weight_scale': QuantType.W8A8.value,
                'model.decoder.layers.2.self_attention.linear_k.weight_scale': QuantType.W8A8.value,
                'model.decoder.layers.2.self_attention.linear_v.weight_scale': QuantType.W8A8.value,
                'model.decoder.layers.3.self_attention.linear_proj.weight_offset': QuantType.W8A8.value,
                'model.decoder.layers.4.self_attention.linear_proj.input_scale': QuantType.W8A8.value,
                'model.decoder.layers.5.self_attention.linear_proj.input_offset': QuantType.W8A8.value,
                'model.decoder.layers.7.mlp.gating.weight': QuantType.W8A8.value,
                'model.decoder.layers.7.mlp.hidden.weight': QuantType.W8A8.value,
                'model.decoder.layers.9.mlp.gating.weight_scale': QuantType.W8A8.value,
                'model.decoder.layers.9.mlp.hidden.weight_scale': QuantType.W8A8.value,
                'model.decoder.layers.10.mlp.gating.weight_offset': QuantType.W8A8.value,
                'model.decoder.layers.10.mlp.hidden.weight_offset': QuantType.W8A8.value,
                'model.decoder.layers.11.mlp.gating.input_scale': QuantType.W8A8.value,
                'model.decoder.layers.11.mlp.hidden.input_scale': QuantType.W8A8.value,
                'model.decoder.layers.12.mlp.gating.input_offset': QuantType.W8A8.value,
                'model.decoder.layers.12.mlp.hidden.input_offset': QuantType.W8A8.value,
                'model.decoder.layers.13.mlp.linear_fc2.weight': QuantType.FLOAT.value,
            }
            for name, value in check_map.items():
                if not check(name, value):
                    return False
            logger.info(f"{quant_type} description test success.")
            return True
        if quant_type.lower() in ["a8w4"]:
            check_map = {
                'model.decoder.layers.0.self_attention.linear_q.weight': QuantType.W4A8_DYNAMIC.value,
                'model.decoder.layers.0.self_attention.linear_k.weight': QuantType.W4A8_DYNAMIC.value,
                'model.decoder.layers.0.self_attention.linear_v.weight': QuantType.W4A8_DYNAMIC.value,
                'model.decoder.layers.1.self_attention.linear_q.weight_scale': QuantType.W4A8_DYNAMIC.value,
                'model.decoder.layers.1.self_attention.linear_k.weight_scale': QuantType.W4A8_DYNAMIC.value,
                'model.decoder.layers.1.self_attention.linear_v.weight_scale': QuantType.W4A8_DYNAMIC.value,
                'model.decoder.layers.2.self_attention.linear_q.weight_offset': QuantType.W4A8_DYNAMIC.value,
                'model.decoder.layers.2.self_attention.linear_k.weight_offset': QuantType.W4A8_DYNAMIC.value,
                'model.decoder.layers.2.self_attention.linear_v.weight_offset': QuantType.W4A8_DYNAMIC.value,
                'model.decoder.layers.3.self_attention.linear_proj.weight': QuantType.W8A8.value,
                'model.decoder.layers.3.self_attention.linear_proj.smooth_scale': QuantType.W8A8.value,
                'model.decoder.layers.4.self_attention.linear_proj.weight_scale': QuantType.W8A8.value,
                'model.decoder.layers.5.self_attention.linear_proj.weight_offset': QuantType.W8A8.value,
                'model.decoder.layers.6.self_attention.linear_proj.input_scale': QuantType.W8A8.value,
                'model.decoder.layers.7.self_attention.linear_proj.input_offset': QuantType.W8A8.value,
                'model.decoder.layers.8.mlp.gating.weight': QuantType.W8A8.value,
                'model.decoder.layers.8.mlp.hidden.weight': QuantType.W8A8.value,
                'model.decoder.layers.9.mlp.gating.smooth_scale': QuantType.W8A8.value,
                'model.decoder.layers.9.mlp.hidden.smooth_scale': QuantType.W8A8.value,
                'model.decoder.layers.10.mlp.gating.weight_scale': QuantType.W8A8.value,
                'model.decoder.layers.10.mlp.hidden.weight_scale': QuantType.W8A8.value,
                'model.decoder.layers.11.mlp.gating.weight_offset': QuantType.W8A8.value,
                'model.decoder.layers.11.mlp.hidden.weight_offset': QuantType.W8A8.value,
                'model.decoder.layers.12.mlp.gating.input_scale': QuantType.W8A8.value,
                'model.decoder.layers.12.mlp.hidden.input_scale': QuantType.W8A8.value,
                'model.decoder.layers.13.mlp.gating.input_offset': QuantType.W8A8.value,
                'model.decoder.layers.13.mlp.hidden.input_offset': QuantType.W8A8.value,
                'model.decoder.layers.14.mlp.linear_fc2.weight': QuantType.FLOAT.value,
            }
            for name, value in check_map.items():
                if not check(name, value):
                    return False
            logger.info(f"{quant_type} description test success.")
            return True
        if quant_type.lower() in ["mix"]:
            check_map = {
                'model.decoder.layers.0.self_attention.linear_q.weight': QuantType.W8A8_DYNAMIC.value,
                'model.decoder.layers.0.self_attention.linear_k.weight': QuantType.W8A8_DYNAMIC.value,
                'model.decoder.layers.0.self_attention.linear_v.weight': QuantType.W8A8_DYNAMIC.value,
                'model.decoder.layers.1.self_attention.linear_q.weight_scale': QuantType.W8A8_DYNAMIC.value,
                'model.decoder.layers.1.self_attention.linear_k.weight_scale': QuantType.W8A8_DYNAMIC.value,
                'model.decoder.layers.1.self_attention.linear_v.weight_scale': QuantType.W8A8_DYNAMIC.value,
                'model.decoder.layers.2.self_attention.linear_q.weight_offset': QuantType.W8A8_DYNAMIC.value,
                'model.decoder.layers.2.self_attention.linear_k.weight_offset': QuantType.W8A8_DYNAMIC.value,
                'model.decoder.layers.2.self_attention.linear_v.weight_offset': QuantType.W8A8_DYNAMIC.value,
                'model.decoder.layers.3.self_attention.linear_proj.weight': QuantType.W8A8_DYNAMIC.value,
                'model.decoder.layers.4.self_attention.linear_proj.weight_scale': QuantType.W8A8_DYNAMIC.value,
                'model.decoder.layers.5.self_attention.linear_proj.weight_offset': QuantType.W8A8_DYNAMIC.value,
                'model.decoder.layers.6.mlp.gating.weight': QuantType.W8A8_DYNAMIC.value,
                'model.decoder.layers.6.mlp.hidden.weight': QuantType.W8A8_DYNAMIC.value,
                'model.decoder.layers.7.mlp.gating.weight_scale': QuantType.W8A8_DYNAMIC.value,
                'model.decoder.layers.7.mlp.hidden.weight_scale': QuantType.W8A8_DYNAMIC.value,
                'model.decoder.layers.8.mlp.gating.weight_offset': QuantType.W8A8_DYNAMIC.value,
                'model.decoder.layers.8.mlp.hidden.weight_offset': QuantType.W8A8_DYNAMIC.value,
                'model.decoder.layers.9.mlp.linear_fc2.weight': QuantType.W8A8_DYNAMIC.value,
                'model.decoder.layers.20.mlp.linear_fc2.weight_scale': QuantType.W8A8_DYNAMIC.value,
                'model.decoder.layers.21.mlp.linear_fc2.weight_offset': QuantType.W8A8_DYNAMIC.value,

                'model.decoder.layers.10.self_attention.linear_q.weight': QuantType.W8A8.value,
                'model.decoder.layers.10.self_attention.linear_k.weight': QuantType.W8A8.value,
                'model.decoder.layers.10.self_attention.linear_v.weight': QuantType.W8A8.value,
                'model.decoder.layers.11.self_attention.linear_q.weight_scale': QuantType.W8A8.value,
                'model.decoder.layers.11.self_attention.linear_k.weight_scale': QuantType.W8A8.value,
                'model.decoder.layers.11.self_attention.linear_v.weight_scale': QuantType.W8A8.value,
                'model.decoder.layers.12.self_attention.linear_q.weight_offset': QuantType.W8A8.value,
                'model.decoder.layers.12.self_attention.linear_k.weight_offset': QuantType.W8A8.value,
                'model.decoder.layers.12.self_attention.linear_v.weight_offset': QuantType.W8A8.value,
                'model.decoder.layers.13.self_attention.linear_q.input_scale': QuantType.W8A8.value,
                'model.decoder.layers.13.self_attention.linear_k.input_scale': QuantType.W8A8.value,
                'model.decoder.layers.13.self_attention.linear_v.input_scale': QuantType.W8A8.value,
                'model.decoder.layers.14.self_attention.linear_q.input_offset': QuantType.W8A8.value,
                'model.decoder.layers.14.self_attention.linear_k.input_offset': QuantType.W8A8.value,
                'model.decoder.layers.14.self_attention.linear_v.input_offset': QuantType.W8A8.value,
                'model.decoder.layers.15.self_attention.linear_q.smooth_scale': QuantType.W8A8.value,
                'model.decoder.layers.15.self_attention.linear_k.smooth_scale': QuantType.W8A8.value,
                'model.decoder.layers.15.self_attention.linear_v.smooth_scale': QuantType.W8A8.value,
                'model.decoder.layers.16.self_attention.linear_proj.weight': QuantType.W8A8.value,
                'model.decoder.layers.16.self_attention.linear_proj.weight_scale': QuantType.W8A8.value,
                'model.decoder.layers.16.self_attention.linear_proj.weight_offset': QuantType.W8A8.value,
                'model.decoder.layers.16.self_attention.linear_proj.input_scale': QuantType.W8A8.value,
                'model.decoder.layers.16.self_attention.linear_proj.input_offset': QuantType.W8A8.value,
                'model.decoder.layers.16.self_attention.linear_proj.smooth_scale': QuantType.W8A8.value,
                'model.decoder.layers.17.mlp.gating.weight': QuantType.W8A8.value,
                'model.decoder.layers.17.mlp.hidden.weight': QuantType.W8A8.value,
                'model.decoder.layers.17.mlp.gating.weight_scale': QuantType.W8A8.value,
                'model.decoder.layers.17.mlp.hidden.weight_scale': QuantType.W8A8.value,
                'model.decoder.layers.17.mlp.gating.weight_offset': QuantType.W8A8.value,
                'model.decoder.layers.17.mlp.hidden.weight_offset': QuantType.W8A8.value,
                'model.decoder.layers.17.mlp.gating.input_scale': QuantType.W8A8.value,
                'model.decoder.layers.17.mlp.hidden.input_scale': QuantType.W8A8.value,
                'model.decoder.layers.17.mlp.gating.input_offset': QuantType.W8A8.value,
                'model.decoder.layers.17.mlp.hidden.input_offset': QuantType.W8A8.value,
                'model.decoder.layers.17.mlp.gating.smooth_scale': QuantType.W8A8.value,
                'model.decoder.layers.17.mlp.hidden.smooth_scale': QuantType.W8A8.value,
                'model.decoder.layers.18.mlp.linear_fc2.weight': QuantType.FLOAT.value,
            }
            for name, value in check_map.items():
                if not check(name, value):
                    return False
            logger.info(f"{quant_type} description test success.")
            return True
        raise RuntimeError(f'Input unsupported quant type: {quant_type}.')

    def get_ds_acc_threshold(self, quant_type) -> Optional[float]:
        score_mapping = {
            "A8W8": 0.41,
            "A8W4": 0.29,
            "mix": 0.34
        }
        return score_mapping.get(quant_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quant_algo', '-a', type=str, required=True)
    uargs = parser.parse_args()
    input_quant_algo = uargs.quant_algo

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    calibrate_config_path = os.path.join(cur_dir, "calibrate_qwen3.yaml")
    infer_config_path = os.path.join(cur_dir, "predict_qwen3.yaml")
    q_ckpt_path = os.path.join(cur_dir, f"qwen3-quant-2p-{input_quant_algo}")
    dataset_path = os.path.join(cur_dir, '/nfs/dataset/workspace/mindspore_dataset/ceval/dev')
    tester = QWen3Tester()
    tester.test_accuracy(calibrate_config_path, infer_config_path, q_ckpt_path, input_quant_algo, dataset_path)


def ptq_predict_2stage_2p_run(quant_algo):
    """
    Feature: test dynamic quant adjust parameter in two stages with two cards.
    Description: apply ptq on qwen3 and check accuracy.
    Expectation: accuracy is good.
    """
    os.environ['quant_algo'] = f"{quant_algo}"
    os.environ['HCCL_CONNECT_TIMEOUT'] = "1800"
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_qwen3_accuracy.py")
    port = get_available_port()
    os.system(f"kill -9 $(lsof -i:{port} | " + "awk '{print $2}')")
    time.sleep(1.0)
    return_code = os.system(
        f"msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 "
        f"--master_port={port} --join=True --log_dir=./test_ptq_{quant_algo}_predict_qwen3_2p_logs "
        f"python {run_file} -a {quant_algo}"
    )
    os.system("ps -u | grep 'test_qwen3_accuracy' | grep -v grep | awk -F ' ' '{print$2}' | xargs kill -9")
    os.system(f"kill -9 $(lsof -i:{port} | " + "awk '{print $2}')")
    time.sleep(1.0)
    assert return_code == 0


@pytest.mark.level2
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_ptq_qwen3_a8w8_accuracy():
    """
    Feature: test omni quant adjust parameter in two stages with two cards.
    Description: apply A8W8 on qwen3 and check score.
    Expectation: score is good.
    """
    ptq_predict_2stage_2p_run("A8W8")


@pytest.mark.level2
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_ptq_qwen3_a8w4_accuracy():
    """
    Feature: test omni quant adjust parameter in two stages with two cards.
    Description: apply A8W4 on llama2 and check score.
    Expectation: score is good.
    """
    ptq_predict_2stage_2p_run("A8W4")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_ptq_qwen3_mix_accuracy():
    """
    Feature: test omni quant adjust parameter in two stages with two cards.
    Description: apply mix quant policy on llama2 and check score.
    Expectation: score is good.
    """
    ptq_predict_2stage_2p_run("mix")
