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

os.environ['GSLOG'] = "1"

from mindspore import dtype as msdtype
from mindspore_gs.common import BackendTarget, logger
from mindspore_gs.ptq import (PTQConfig, PTQMode,
                              OutliersSuppressionType,
                              QuantGranularity,
                              GPTQQuantConfig,
                              PrecisionRecovery)
from mindspore_gs.ptq.utils import QuantType
from ptq_model_tester import PTQModelTester


class QWen3MoETester(PTQModelTester):
    """QWen3MoETester"""
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
        gptq_config = GPTQQuantConfig(static_groups=True, desc_act=True)
        a8w4_dynamic_cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND,
                                     weight_quant_dtype=msdtype.qint4x2, act_quant_dtype=msdtype.int8,
                                     weight_quant_granularity=QuantGranularity.PER_GROUP,
                                     group_size=64, algo_args=gptq_config,
                                     act_quant_granularity=QuantGranularity.PER_TOKEN,
                                     precision_recovery=PrecisionRecovery.GPTQ, weight_clip=True,
                                     opname_blacklist=['output_layer'])
        layer_policies = OrderedDict({r".*\.[0,1]\.self_attention.*": osl_cfg,
                                      r".*\.[2,3]\.self_attention.*": smoothquant_cfg,
                                      'not match': a8w4_dynamic_cfg,
                                     })
        return a8w8_dynamic_cfg, layer_policies

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
            'model.layers.4.self_attn.q_proj.weight': QuantType.W8A8_DYNAMIC.value,
            'model.layers.4.self_attn.k_proj.weight': QuantType.W8A8_DYNAMIC.value,
            'model.layers.4.self_attn.v_proj.weight': QuantType.W8A8_DYNAMIC.value,
            'model.layers.5.self_attn.q_proj.weight_scale': QuantType.W8A8_DYNAMIC.value,
            'model.layers.5.self_attn.k_proj.weight_scale': QuantType.W8A8_DYNAMIC.value,
            'model.layers.5.self_attn.v_proj.weight_scale': QuantType.W8A8_DYNAMIC.value,
            'model.layers.4.self_attn.q_proj.weight_offset': QuantType.W8A8_DYNAMIC.value,
            'model.layers.4.self_attn.k_proj.weight_offset': QuantType.W8A8_DYNAMIC.value,
            'model.layers.4.self_attn.v_proj.weight_offset': QuantType.W8A8_DYNAMIC.value,
            'model.layers.5.self_attn.o_proj.weight': QuantType.W8A8_DYNAMIC.value,
            'model.layers.5.self_attn.o_proj.weight_scale': QuantType.W8A8_DYNAMIC.value,
            'model.layers.5.self_attn.o_proj.weight_offset': QuantType.W8A8_DYNAMIC.value,
            'model.layers.0.mlp.experts.0.gate_proj.weight': QuantType.W8A8_DYNAMIC.value,
            'model.layers.1.mlp.experts.10.up_proj.weight': QuantType.W8A8_DYNAMIC.value,
            'model.layers.2.mlp.experts.20.gate_proj.weight_scale': QuantType.W8A8_DYNAMIC.value,
            'model.layers.3.mlp.experts.30.up_proj.weight_scale': QuantType.W8A8_DYNAMIC.value,
            'model.layers.4.mlp.experts.40.gate_proj.weight_offset': QuantType.W8A8_DYNAMIC.value,
            'model.layers.5.mlp.experts.50.up_proj.weight_offset': QuantType.W8A8_DYNAMIC.value,
            'model.layers.0.mlp.experts.60.down_proj.weight': QuantType.W8A8_DYNAMIC.value,
            'model.layers.1.mlp.experts.70.down_proj.weight_scale': QuantType.W8A8_DYNAMIC.value,
            'model.layers.2.mlp.experts.80.down_proj.weight_offset': QuantType.W8A8_DYNAMIC.value,

            'model.layers.0.self_attn.q_proj.weight': QuantType.W8A8.value,
            'model.layers.1.self_attn.k_proj.weight': QuantType.W8A8.value,
            'model.layers.2.self_attn.v_proj.weight': QuantType.W8A8.value,
            'model.layers.3.self_attn.q_proj.weight_scale': QuantType.W8A8.value,
            'model.layers.0.self_attn.k_proj.weight_scale': QuantType.W8A8.value,
            'model.layers.1.self_attn.v_proj.weight_scale': QuantType.W8A8.value,
            'model.layers.2.self_attn.q_proj.weight_offset': QuantType.W8A8.value,
            'model.layers.3.self_attn.k_proj.weight_offset': QuantType.W8A8.value,
            'model.layers.0.self_attn.v_proj.weight_offset': QuantType.W8A8.value,
            'model.layers.1.self_attn.q_proj.input_scale': QuantType.W8A8.value,
            'model.layers.2.self_attn.k_proj.input_scale': QuantType.W8A8.value,
            'model.layers.3.self_attn.v_proj.input_scale': QuantType.W8A8.value,
            'model.layers.0.self_attn.q_proj.input_offset': QuantType.W8A8.value,
            'model.layers.1.self_attn.k_proj.input_offset': QuantType.W8A8.value,
            'model.layers.2.self_attn.v_proj.input_offset': QuantType.W8A8.value,
            'model.layers.3.self_attn.q_proj.smooth_scale': QuantType.W8A8.value,
            'model.layers.0.self_attn.k_proj.smooth_scale': QuantType.W8A8.value,
            'model.layers.1.self_attn.v_proj.smooth_scale': QuantType.W8A8.value,
            'model.layers.2.self_attn.o_proj.weight': QuantType.W8A8.value,
            'model.layers.3.self_attn.o_proj.weight_scale': QuantType.W8A8.value,
            'model.layers.0.self_attn.o_proj.weight_offset': QuantType.W8A8.value,
            'model.layers.1.self_attn.o_proj.input_scale': QuantType.W8A8.value,
            'model.layers.2.self_attn.o_proj.input_offset': QuantType.W8A8.value,
            'model.layers.3.self_attn.o_proj.smooth_scale': QuantType.W8A8.value,
        }
        for name, value in check_map.items():
            if not check(name, value):
                return False
        logger.info("quant description test success.")
        return True

    def get_golden(self) -> tuple[str, str]:
        return "介绍北京故宫", "介绍北京故宫(passport(passport(passport护身 tjejer"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, required=True)
    args = parser.parse_args()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    calibrate_config_path = os.path.join(cur_dir, "calibrate_qwen3_moe.yaml")
    infer_config_path = os.path.join(cur_dir, "predict_qwen3_moe.yaml")
    q_ckpt_path = os.path.join(cur_dir, f"qwen3-moe-quant")
    dataset_path = os.path.join(cur_dir, '/nfs/dataset/workspace/mindspore_dataset/ceval/dev')
    tester = QWen3MoETester()
    result = tester.golden_accuracy(calibrate_config_path, infer_config_path, q_ckpt_path, dataset_path)
    tester.tear_down(q_ckpt_path, args.log_path)
    assert result, 'qwen3 moe accuracy test failed.'
