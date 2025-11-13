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


from typing import Optional
import argparse
import json
import os
from collections import OrderedDict

os.environ['GSLOG'] = "1"

from mindspore import dtype as msdtype
from mindspore_gs.common import BackendTarget, logger
from mindspore_gs.ptq import (PTQConfig, PTQMode,
                              OutliersSuppressionType,
                              QuantGranularity)
from mindspore_gs.ptq.utils import QuantType
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
        with open(desc_json_path, "r", encoding="utf-8") as fp:
            desc_map = json.load(fp)

        def check(name, expect):
            cur = desc_map.get(name)
            ret = cur == expect
            if not ret:
                logger.error(f"quant info of {name} should be {expect}, but got: {cur}.")
            return ret

        check_map = {
            'model.layers.0.mlp.gate_proj.weight': QuantType.W8A8_DYNAMIC.value,
            'model.layers.1.mlp.up_proj.weight': QuantType.W8A8_DYNAMIC.value,
            'model.layers.2.mlp.gate_proj.weight_scale': QuantType.W8A8_DYNAMIC.value,
            'model.layers.3.mlp.up_proj.weight_scale': QuantType.W8A8_DYNAMIC.value,
            'model.layers.4.mlp.gate_proj.weight_offset': QuantType.W8A8_DYNAMIC.value,
            'model.layers.5.mlp.up_proj.weight_offset': QuantType.W8A8_DYNAMIC.value,
            'model.layers.6.mlp.down_proj.weight': QuantType.W8A8_DYNAMIC.value,
            'model.layers.7.mlp.down_proj.weight_scale': QuantType.W8A8_DYNAMIC.value,
            'model.layers.8.mlp.down_proj.weight_offset': QuantType.W8A8_DYNAMIC.value,
            'model.layers.9.mlp.gate_proj.weight': QuantType.W8A8_DYNAMIC.value,
            'model.layers.10.mlp.up_proj.weight': QuantType.W8A8_DYNAMIC.value,
            'model.layers.11.mlp.gate_proj.weight_scale': QuantType.W8A8_DYNAMIC.value,
            'model.layers.12.mlp.up_proj.weight_scale': QuantType.W8A8_DYNAMIC.value,
            'model.layers.20.mlp.gate_proj.weight_offset': QuantType.W8A8_DYNAMIC.value,
            'model.layers.21.mlp.up_proj.weight_offset': QuantType.W8A8_DYNAMIC.value,
            'model.layers.22.mlp.down_proj.weight': QuantType.W8A8_DYNAMIC.value,
            'model.layers.23.mlp.down_proj.weight_scale': QuantType.W8A8_DYNAMIC.value,
            'model.layers.24.mlp.down_proj.weight_offset': QuantType.W8A8_DYNAMIC.value,

            'model.layers.0.self_attn.q_proj.weight': QuantType.W8A8.value,
            'model.layers.1.self_attn.k_proj.weight': QuantType.W8A8.value,
            'model.layers.2.self_attn.v_proj.weight': QuantType.W8A8.value,
            'model.layers.3.self_attn.q_proj.weight_scale': QuantType.W8A8.value,
            'model.layers.4.self_attn.k_proj.weight_scale': QuantType.W8A8.value,
            'model.layers.5.self_attn.v_proj.weight_scale': QuantType.W8A8.value,
            'model.layers.6.self_attn.q_proj.weight_offset': QuantType.W8A8.value,
            'model.layers.7.self_attn.k_proj.weight_offset': QuantType.W8A8.value,
            'model.layers.8.self_attn.v_proj.weight_offset': QuantType.W8A8.value,
            'model.layers.9.self_attn.q_proj.input_scale': QuantType.W8A8.value,
            'model.layers.10.self_attn.k_proj.input_scale': QuantType.W8A8.value,
            'model.layers.11.self_attn.v_proj.input_scale': QuantType.W8A8.value,
            'model.layers.12.self_attn.q_proj.input_offset': QuantType.W8A8.value,
            'model.layers.13.self_attn.k_proj.input_offset': QuantType.W8A8.value,
            'model.layers.14.self_attn.v_proj.input_offset': QuantType.W8A8.value,
            'model.layers.15.self_attn.q_proj.smooth_scale': QuantType.W8A8.value,
            'model.layers.16.self_attn.k_proj.smooth_scale': QuantType.W8A8.value,
            'model.layers.17.self_attn.v_proj.smooth_scale': QuantType.W8A8.value,
            'model.layers.18.self_attn.o_proj.weight': QuantType.W8A8.value,
            'model.layers.19.self_attn.o_proj.weight_scale': QuantType.W8A8.value,
            'model.layers.19.self_attn.o_proj.weight_offset': QuantType.W8A8.value,
            'model.layers.19.self_attn.o_proj.input_scale': QuantType.W8A8.value,
            'model.layers.19.self_attn.o_proj.input_offset': QuantType.W8A8.value,
            'model.layers.19.self_attn.o_proj.smooth_scale': QuantType.W8A8.value,
        }
        for name, value in check_map.items():
            if not check(name, value):
                return False
        logger.info("quant description test success.")
        any_output_layer = any("output_layer." in k for k in desc_map.keys())
        if any_output_layer:
            logger.error("output_layer.* should be mapped to lm_head.* in description.")
            return False

        index_json_path = os.path.join(quant_ckpt_path, "model.safetensors.index.json")
        if not os.path.exists(index_json_path):
            logger.error("No safetensors index json file.")
            return False
        with open(index_json_path, "r", encoding="utf-8") as fp:
            index_data = json.load(fp)
        weight_map = index_data.get("weight_map", {})
        any_output_layer_idx = any("output_layer." in k for k in weight_map.keys())
        if any_output_layer_idx:
            logger.error("output_layer.* found in safetensors index.")
            return False
        return True

    def get_ds_acc_threshold(self) -> Optional[float]:
        return 0.295


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test QWen3 accuracy")
    parser.add_argument("--log_path", type=str, required=True)
    args = parser.parse_args()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    calibrate_config_path = os.path.join(cur_dir, "calibrate_qwen3.yaml")
    infer_config_path = os.path.join(cur_dir, "predict_qwen3.yaml")
    q_ckpt_path = os.path.join(cur_dir, "qwen3-quant")
    dataset_path = os.path.join(cur_dir, '/home/workspace/mindspore_dataset/ceval/dev')
    tester = QWen3Tester()
    result = tester.dataset_accuracy(calibrate_config_path, infer_config_path, q_ckpt_path, dataset_path)
    tester.tear_down(q_ckpt_path, args.log_path)
    assert result, 'qwen3 accuracy test failed.'
