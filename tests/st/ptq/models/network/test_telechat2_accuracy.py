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
from mindspore_gs.common import logger
from mindspore_gs.ptq import (PTQConfig, PTQMode,
                              OutliersSuppressionType)
from tests.st.test_utils import get_available_port
from ptq_model_tester import PTQModelTester


class Telechat2Tester(PTQModelTester):
    """Telechat2Tester"""
    def create_ptq_config(self):
        """create_ptq"""
        cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                        act_quant_dtype=msdtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH,
                        opname_blacklist=['output_layer', 'linear_fc2'])
        layer_policies = OrderedDict()
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
            'model.decoder.layers.8.mlp.gating.smooth_scale': QuantType.W8A8.value,
            'model.decoder.layers.8.mlp.hidden.smooth_scale': QuantType.W8A8.value,
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
        logger.info("quant description test success.")
        return True

    def get_ds_acc_threshold(self) -> Optional[float]:
        return 0.8


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    calibrate_config_path = os.path.join(cur_dir, "calibrate_telechat2.yaml")
    infer_config_path = os.path.join(cur_dir, "predict_telechat2.yaml")
    q_ckpt_path = os.path.join(cur_dir, f"telechat2-quant")
    log_path = f"./test_ptq_predict_telechat2_4p_logs"
    dataset_path = os.path.join(cur_dir, '/nfs/dataset/workspace/mindspore_dataset/ceval/dev')
    tester = Telechat2Tester()
    result = tester.dataset_accuracy(calibrate_config_path, infer_config_path, q_ckpt_path, dataset_path)
    if not result:
        tester.print_log(log_path)
    tester.tear_down(q_ckpt_path, log_path)
    assert result, 'telechat2 accuracy test failed.'


def ptq_predict_2stage_4p_run():
    """
    Feature: test dynamic quant adjust parameter in two stages with four cards.
    Description: apply ptq on telechat2 and check accuracy.
    Expectation: accuracy is good.
    """
    os.environ['HCCL_CONNECT_TIMEOUT'] = "1800"
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_telechat2_accuracy.py")
    port = get_available_port()
    os.system(f"kill -9 $(lsof -i:{port} | " + "awk '{print $2}')")
    time.sleep(1.0)
    return_code = os.system(
        f"msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 "
        f"--master_port={port} --join=True --log_dir=./test_ptq_predict_telechat2_4p_logs "
        f"python {run_file}"
    )
    time.sleep(1.0)
    assert return_code == 0


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_ptq_telechat2_a8w8_accuracy():
    """
    Feature: test omni quant adjust parameter in two stages with four cards.
    Description: apply A8W8 on telechat2 and check score.
    Expectation: score is good.
    """
    ptq_predict_2stage_4p_run()
