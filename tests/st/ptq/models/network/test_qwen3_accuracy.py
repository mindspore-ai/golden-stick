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
import json
import pytest
from mindspore import dtype as msdtype
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq.utils import QuantType

os.environ['GSLOG'] = "1"

from mindspore_gs.common import logger
from mindspore_gs.ptq import (PTQConfig, PTQMode,
                              OutliersSuppressionType,
                              QuantGranularity)
from tests.st.test_utils import get_available_port
from ptq_model_tester import PTQModelTester


class QWen3Tester(PTQModelTester):
    """QWen3Tester"""
    def create_ptq_config(self):
        """create_ptq"""
        a8w8_dynamic_cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND,
                                     weight_quant_dtype=msdtype.int8, act_quant_dtype=msdtype.int8,
                                     act_quant_granularity=QuantGranularity.PER_TOKEN,
                                     opname_blacklist=['output_layer'])
        smoothquant_cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND,
                                    weight_quant_dtype=msdtype.int8, act_quant_dtype=msdtype.int8,
                                    act_quant_granularity=QuantGranularity.PER_TENSOR,
                                    outliers_suppression=OutliersSuppressionType.SMOOTH,
                                    opname_blacklist=['output_layer', 'linear_fc2'])
        osl_cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND,
                            weight_quant_dtype=msdtype.int8, act_quant_dtype=msdtype.int8,
                            act_quant_granularity=QuantGranularity.PER_TENSOR,
                            outliers_suppression=OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE,
                            opname_blacklist=['output_layer', 'linear_fc2'])
        cfg = a8w8_dynamic_cfg
        layer_policies = OrderedDict({r'.*\.[0-9]\.self_attention.*': osl_cfg,
                                      r'.*\.1[0-9]\.self_attention.*': smoothquant_cfg,
                                     })
        return cfg, layer_policies

    def check_quant_description(self, quant_ckpt_path) -> bool:
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

        check_map = {
            'model.decoder.layers.0.mlp.gating.weight': QuantType.W8A8_DYNAMIC.value,
            'model.decoder.layers.1.mlp.hidden.weight': QuantType.W8A8_DYNAMIC.value,
            'model.decoder.layers.2.mlp.gating.weight_scale': QuantType.W8A8_DYNAMIC.value,
            'model.decoder.layers.3.mlp.hidden.weight_scale': QuantType.W8A8_DYNAMIC.value,
            'model.decoder.layers.4.mlp.gating.weight_offset': QuantType.W8A8_DYNAMIC.value,
            'model.decoder.layers.5.mlp.hidden.weight_offset': QuantType.W8A8_DYNAMIC.value,
            'model.decoder.layers.6.mlp.linear_fc2.weight': QuantType.W8A8_DYNAMIC.value,
            'model.decoder.layers.7.mlp.linear_fc2.weight_scale': QuantType.W8A8_DYNAMIC.value,
            'model.decoder.layers.8.mlp.linear_fc2.weight_offset': QuantType.W8A8_DYNAMIC.value,
            'model.decoder.layers.9.mlp.gating.weight': QuantType.W8A8_DYNAMIC.value,
            'model.decoder.layers.10.mlp.hidden.weight': QuantType.W8A8_DYNAMIC.value,
            'model.decoder.layers.11.mlp.gating.weight_scale': QuantType.W8A8_DYNAMIC.value,
            'model.decoder.layers.12.mlp.hidden.weight_scale': QuantType.W8A8_DYNAMIC.value,
            'model.decoder.layers.20.mlp.gating.weight_offset': QuantType.W8A8_DYNAMIC.value,
            'model.decoder.layers.21.mlp.hidden.weight_offset': QuantType.W8A8_DYNAMIC.value,
            'model.decoder.layers.22.mlp.linear_fc2.weight': QuantType.W8A8_DYNAMIC.value,
            'model.decoder.layers.23.mlp.linear_fc2.weight_scale': QuantType.W8A8_DYNAMIC.value,
            'model.decoder.layers.24.mlp.linear_fc2.weight_offset': QuantType.W8A8_DYNAMIC.value,

            'model.decoder.layers.0.self_attention.linear_q.weight': QuantType.W8A8.value,
            'model.decoder.layers.1.self_attention.linear_k.weight': QuantType.W8A8.value,
            'model.decoder.layers.2.self_attention.linear_v.weight': QuantType.W8A8.value,
            'model.decoder.layers.3.self_attention.linear_q.weight_scale': QuantType.W8A8.value,
            'model.decoder.layers.4.self_attention.linear_k.weight_scale': QuantType.W8A8.value,
            'model.decoder.layers.5.self_attention.linear_v.weight_scale': QuantType.W8A8.value,
            'model.decoder.layers.6.self_attention.linear_q.weight_offset': QuantType.W8A8.value,
            'model.decoder.layers.7.self_attention.linear_k.weight_offset': QuantType.W8A8.value,
            'model.decoder.layers.8.self_attention.linear_v.weight_offset': QuantType.W8A8.value,
            'model.decoder.layers.9.self_attention.linear_q.input_scale': QuantType.W8A8.value,
            'model.decoder.layers.10.self_attention.linear_k.input_scale': QuantType.W8A8.value,
            'model.decoder.layers.11.self_attention.linear_v.input_scale': QuantType.W8A8.value,
            'model.decoder.layers.12.self_attention.linear_q.input_offset': QuantType.W8A8.value,
            'model.decoder.layers.13.self_attention.linear_k.input_offset': QuantType.W8A8.value,
            'model.decoder.layers.14.self_attention.linear_v.input_offset': QuantType.W8A8.value,
            'model.decoder.layers.15.self_attention.linear_q.smooth_scale': QuantType.W8A8.value,
            'model.decoder.layers.16.self_attention.linear_k.smooth_scale': QuantType.W8A8.value,
            'model.decoder.layers.17.self_attention.linear_v.smooth_scale': QuantType.W8A8.value,
            'model.decoder.layers.18.self_attention.linear_proj.weight': QuantType.W8A8.value,
            'model.decoder.layers.19.self_attention.linear_proj.weight_scale': QuantType.W8A8.value,
            'model.decoder.layers.19.self_attention.linear_proj.weight_offset': QuantType.W8A8.value,
            'model.decoder.layers.19.self_attention.linear_proj.input_scale': QuantType.W8A8.value,
            'model.decoder.layers.19.self_attention.linear_proj.input_offset': QuantType.W8A8.value,
            'model.decoder.layers.19.self_attention.linear_proj.smooth_scale': QuantType.W8A8.value,
        }
        for name, value in check_map.items():
            if not check(name, value):
                return False
        logger.info("quant description test success.")
        return True

    def get_ds_acc_threshold(self) -> Optional[float]:
        return 0.42


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    calibrate_config_path = os.path.join(cur_dir, "calibrate_qwen3.yaml")
    infer_config_path = os.path.join(cur_dir, "predict_qwen3.yaml")
    q_ckpt_path = os.path.join(cur_dir, f"qwen3-quant")
    log_path = f"./test_ptq_predict_qwen3_2p_logs"
    dataset_path = os.path.join(cur_dir, '/nfs/dataset/workspace/mindspore_dataset/ceval/dev')
    tester = QWen3Tester()
    result = tester.dataset_accuracy(calibrate_config_path, infer_config_path, q_ckpt_path, dataset_path)
    if not result:
        tester.print_log(log_path)
    tester.tear_down(q_ckpt_path, log_path)
    assert result, 'qwen3 accuracy test failed.'


def ptq_predict_2stage_2p_run():
    """
    Feature: test dynamic quant adjust parameter in two stages with two cards.
    Description: apply ptq on qwen3 and check accuracy.
    Expectation: accuracy is good.
    """
    os.environ['HCCL_CONNECT_TIMEOUT'] = "1800"
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_qwen3_accuracy.py")
    port = get_available_port()
    os.system(f"kill -9 $(lsof -i:{port} | " + "awk '{print $2}')")
    time.sleep(1.0)
    return_code = os.system(
        f"msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 "
        f"--master_port={port} --join=True --log_dir=./test_ptq_predict_qwen3_2p_logs "
        f"python {run_file}"
    )
    time.sleep(1.0)
    assert return_code == 0


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_ptq_qwen3_mix_accuracy():
    """
    Feature: test omni quant adjust parameter in two stages with two cards.
    Description: apply mix quant policy on llama2 and check score.
    Expectation: score is good.
    """
    ptq_predict_2stage_2p_run()
