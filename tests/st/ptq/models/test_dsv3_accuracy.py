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
import pytest
from mindspore import dtype as msdtype
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import (PTQConfig, PTQMode,
                              OutliersSuppressionType)
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
    tester.test_accuracy(calibrate_config_path, infer_config_path, q_ckpt_path, input_quant_algo, dataset_path)


def ptq_predict_2stage_4p_run(quant_algo):
    """
    Feature: test dynamic quant adjust parameter in two stages with two cards.
    Description: apply ptq on deepseek-v3/r1 and check accuracy.
    Expectation: accuracy is good.
    """
    os.environ['quant_algo'] = f"{quant_algo}"
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptq_dsv3_runner.py")
    port = get_available_port()
    os.system(f"kill -9 $(lsof -i:{port} | " + "awk '{print $2}')")
    time.sleep(1.0)
    return_code = os.system(
        f"msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 "
        f"--master_port={port} --join=True --log_dir=./test_ptq_{quant_algo}_predict_dsv3_4p_logs "
        f"python {run_file} -a {quant_algo}"
    )
    os.system("ps -u | grep 'ptq_dsv3_runner' | grep -v grep | awk -F ' ' '{print$2}' | xargs kill -9")
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
