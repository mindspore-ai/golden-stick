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


import argparse
import json
import os
from collections import OrderedDict

from mindspore import dtype as msdtype
from mindspore_gs.common import BackendTarget, logger
from mindspore_gs.ptq import (PTQConfig, PTQMode,
                              OutliersSuppressionType)
from mindspore_gs.ptq.utils import QuantType
from ptq_model_tester import PTQModelTester


class Telechat2Tester(PTQModelTester):
    """Telechat2Tester"""
    def create_ptq_config(self):
        """create_ptq"""
        cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                        act_quant_dtype=msdtype.int8,
                        outliers_suppression=OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE,
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
            'transformer.h.0.self_attention.query.weight': QuantType.W8A8.value,
            'transformer.h.0.self_attention.key.weight': QuantType.W8A8.value,
            'transformer.h.0.self_attention.value.weight': QuantType.W8A8.value,
            'transformer.h.1.self_attention.dense.smooth_scale': QuantType.W8A8.value,
            'transformer.h.1.self_attention.query.weight_scale': QuantType.W8A8.value,
            'transformer.h.1.self_attention.key.weight_scale': QuantType.W8A8.value,
            'transformer.h.2.self_attention.value.weight_scale': QuantType.W8A8.value,
            'transformer.h.2.self_attention.dense.weight_offset': QuantType.W8A8.value,
            'transformer.h.2.self_attention.dense.input_scale': QuantType.W8A8.value,
            'transformer.h.3.self_attention.dense.input_offset': QuantType.W8A8.value,
            'transformer.h.3.mlp.gate_proj.weight': QuantType.W8A8.value,
            'transformer.h.3.mlp.up_proj.weight': QuantType.W8A8.value,
            'transformer.h.4.mlp.gate_proj.smooth_scale': QuantType.W8A8.value,
            'transformer.h.4.mlp.up_proj.smooth_scale': QuantType.W8A8.value,
            'transformer.h.4.mlp.gate_proj.weight_scale': QuantType.W8A8.value,
            'transformer.h.5.mlp.up_proj.weight_scale': QuantType.W8A8.value,
            'transformer.h.5.mlp.gate_proj.weight_offset': QuantType.W8A8.value,
            'transformer.h.5.mlp.up_proj.weight_offset': QuantType.W8A8.value,
            'transformer.h.6.mlp.gate_proj.input_scale': QuantType.W8A8.value,
            'transformer.h.7.mlp.up_proj.input_scale': QuantType.W8A8.value,
            'transformer.h.8.mlp.gate_proj.input_offset': QuantType.W8A8.value,
            'transformer.h.9.mlp.up_proj.input_offset': QuantType.W8A8.value,
            'transformer.h.9.mlp.down_proj.weight': QuantType.FLOAT.value,
        }
        for name, value in check_map.items():
            if not check(name, value):
                return False
        logger.info("quant description test success.")
        return True

    def get_golden(self) -> tuple[str, str]:
        return "介绍北京故宫", "介绍北京故宫湿疣沃沃担联网担湿疣沃 Passage"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Telechat2 accuracy")
    parser.add_argument("--log_path", type=str, required=True)
    args = parser.parse_args()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    calibrate_config_path = os.path.join(cur_dir, "calibrate_telechat2.yaml")
    infer_config_path = os.path.join(cur_dir, "predict_telechat2.yaml")
    q_ckpt_path = os.path.join(cur_dir, f"telechat2-quant")
    dataset_path = os.path.join(cur_dir, '/home/workspace/mindspore_dataset/ceval/dev')
    tester = Telechat2Tester()
    result = tester.golden_accuracy(calibrate_config_path, infer_config_path, q_ckpt_path, dataset_path)
    tester.tear_down(q_ckpt_path, args.log_path)
    assert result, 'telechat2 accuracy test failed.'
