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
import sys
import time
import json
import shutil
from pathlib import Path
from safetensors import safe_open
import pytest

import mindspore as ms
from mindspore import dtype as msdtype
from mindspore.communication import get_rank, get_group_size
from mindspore.nn.utils import no_init_parameters

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../mindformers")))

from mindformers import (AutoModel, MindFormerConfig,
                         build_context, build_parallel_config)
from transformers import AutoTokenizer

from mindspore_gs.common import BackendTarget, logger
from mindspore_gs.ptq import (PTQConfig, PTQMode, OutliersSuppressionType,
                              PrecisionRecovery, QuantGranularity, GPTQQuantConfig)
from mindspore_gs.ptq.utils import QuantType
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
            'model.layers.0.self_attn.q_a_proj.weight': QuantType.W8A8.value,
            'model.layers.1.self_attn.q_b_proj.weight_scale': QuantType.W8A8.value,
            'model.layers.2.self_attn.kv_a_proj_with_mqa.weight_offset': QuantType.W8A8.value,
            'model.layers.0.self_attn.o_proj.smooth_scale': QuantType.W8A8.value,
            'model.layers.1.self_attn.kv_b_proj.weight': QuantType.FLOAT.value,
            'model.layers.0.mlp.gate_proj.weight': QuantType.W8A8_DYNAMIC.value,
            'model.layers.1.mlp.up_proj.weight_scale': QuantType.W8A8_DYNAMIC.value,
            'model.layers.2.mlp.down_proj.weight_offset': QuantType.W8A8_DYNAMIC.value,
            'model.layers.3.mlp.experts.0.gate_proj.weight': QuantType.W4A8_DYNAMIC.value,
            'model.layers.3.mlp.experts.4.up_proj.weight_scale': QuantType.W4A8_DYNAMIC.value,
            'model.layers.3.mlp.experts.118.up_proj.weight_offset': QuantType.W4A8_DYNAMIC.value,
            'model.layers.3.mlp.shared_experts.gate_proj.weight': QuantType.W8A8_DYNAMIC.value,
            'model.layers.3.mlp.shared_experts.up_proj.weight_scale': QuantType.W8A8_DYNAMIC.value,
            'model.layers.3.mlp.shared_experts.down_proj.weight_offset': QuantType.W8A8_DYNAMIC.value,
        }
        for name, value in check_map.items():
            if not check(name, value):
                return False
        logger.info("quant description test success.")
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

    def _copy_original_json(self, original_path, save_path):
        src_path = Path(original_path)
        for json_file in src_path.glob('*.json'):
            if json_file.name.endswith('.index.json'):
                continue
            shutil.copy2(json_file, os.path.join(save_path, json_file.name))

    def _modify_description_file(self, quant_ckpt_path, unify_quant_ckpt_path):
        """_modify_description_file"""
        file_path = os.path.join(quant_ckpt_path, "quantization_description.json")
        save_path = os.path.join(unify_quant_ckpt_path, "quantization_description.json")
        if not os.path.exists(file_path):
            raise ValueError(f"Not found quantization_description.json in {quant_ckpt_path}, "
                             "please check the quantization process.")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            data["group_size"] = 256
            with open(save_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
        except Exception as e:
            raise RuntimeError("Found error when Modify description file."
                               f"The details of error are {e}") from e

    def unify_safetensors(self, float_ckpt_path, quant_ckpt_path,
                          unify_quant_ckpt_path):
        """unify_safetensors"""
        run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "utils/unify_safetensors.py")
        return_code = os.system(
            f"python {run_file} --input_dir={quant_ckpt_path} "
            f"--output_dir={unify_quant_ckpt_path} "
            f"--output_file_prefix=a8w4 "
            f"--rank_num=4 "
            f"--quant_type=a8w4")
        time.sleep(1.0)
        assert return_code == 0

        # copy origin *.json file to unify quant ckpt path
        self._copy_original_json(float_ckpt_path, unify_quant_ckpt_path)
        # add group_size to quantization_description.json
        self._modify_description_file(quant_ckpt_path, unify_quant_ckpt_path)

    # pylint: disable=arguments-differ
    def forward_model(self, config_path_, ckpt_path_, question):
        """forward model"""
        os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
        os.environ['MS_INTERNAL_ENABLE_CUSTOM_KERNAL_LIST'] = "QbmmAllReduceAdd,QbmmAdd"
        os.environ['MS_ALLOC_CONF'] = "enable_vmm:True"
        os.environ.pop('ENFORCE_EAGER', None)
        ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
        if not ascend_path:
            os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"

        set_load_checkpoint = [
            "sed",
            "-i",
            f's#"load_checkpoint: .*"#"load_checkpoint: {ckpt_path_}"#g',
            config_path_
        ]
        set_pretrained_model_dir = [
            "sed",
            "-i",
            f's#"pretrained_model_dir: .*"#"pretrained_model_dir: {ckpt_path_}"#g',
            config_path_
        ]
        return_code = os.system(" ".join(set_load_checkpoint))
        assert return_code == 0, "Set load_checkpoint failed."
        return_code = os.system(" ".join(set_pretrained_model_dir))
        assert return_code == 0, "Set pretrained_model_dir failed."

        config = MindFormerConfig(config_path_)
        build_context(config)
        build_parallel_config(config)
        with no_init_parameters():
            network = AutoModel.from_config(config_path_)
        if config.load_checkpoint:
            network.load_weights(config.load_checkpoint)

        os.environ['MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST'] = "PagedAttention"
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_dir,
                                                  trust_remote_code=True)
        input_ids = tokenizer.encode(question, add_special_tokens=True)
        outputs = network.generate(input_ids, max_new_tokens=20)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_golden(self) -> tuple[str, str]:
        return "介绍下北京故宫", "介绍下北京故宫博物院ODాలు"

    # pylint: disable=arguments-differ
    def golden_accuracy(self, infer_config_path_, unify_quant_ckpt_path_):
        """golden_accuracy"""
        question, answer = self.get_golden()
        result = question is not None and answer is not None, \
                 "Please implement get_golden before invoke golden_accuracy."

        result = self.check_quant_description(unify_quant_ckpt_path_)
        if result:
            pred = self.forward_model(infer_config_path_, unify_quant_ckpt_path_, question)
            result = pred.startswith(answer)
            print("="*50, flush=True)
            print(f"{question} predict: {pred}, answer: {answer}", "success" if result else "failed", flush=True)
        try:
            group_size = get_group_size()
        except RuntimeError:
            group_size = 0
        if group_size > 0:
            ms.mint.distributed.barrier()
        return result

def test_quant_deepseek():
    """
    Feature: test mixture quant adjust parameter in two stages with two cards.
    Description: apply mix-quant on deepseek-v3/r1 and check score.
    Expectation: the quantization process is successful.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    calibrate_config_path = os.path.join(cur_dir, "calibrate_deepseek3_671b.yaml")
    q_ckpt_path = os.path.join(cur_dir, "dsv3-quant")
    dataset_path = os.path.join(cur_dir, '/home/workspace/mindspore_dataset/ceval/dev')
    tester = DeepSeekV3Tester()
    tester.quant_model(calibrate_config_path, q_ckpt_path,
                       dataset_path, fake_quant=False)
    tester.check_safetensor_split(q_ckpt_path)


def test_unify_safetensor():
    """
    Feature: test unify safetensors for quantized tp split safetensors.
    Description: unify safetensors from tp split safetensors.
    Expectation: unify successfully.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    float_ckpt_path = "/home/workspace/mindspore_dataset/weight/DeepSeek-R1-bf16"
    q_ckpt_path = os.path.join(cur_dir, "dsv3-quant")
    unify_q_ckpt_path = os.path.join(cur_dir, "dsv3-quant-unify")
    tester = DeepSeekV3Tester()
    tester.unify_safetensors(float_ckpt_path, q_ckpt_path,
                             unify_q_ckpt_path)


def test_eval_deepseek():
    """
    Feature: test evaluation of deepseek v3/r1 a8w4 quantization.
    Description: evaluate the quant model output.
    Expectation: score or output id is good.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    infer_config_path = os.path.join(cur_dir, "predict_deepseek3_671b.yaml")
    unify_q_ckpt_path = os.path.join(cur_dir, "dsv3-quant-unify")
    tester = DeepSeekV3Tester()
    tester.golden_accuracy(infer_config_path, unify_q_ckpt_path)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_ptq_dsv3_mix_accuracy():
    """
    Feature: test mixture quant adjust parameter in two stages with two cards.
    Description: apply mix-quant on deepseek-v3/r1 and check score.
    Expectation: score is good.
    """
    os.environ['HCCL_CONNECT_TIMEOUT'] = "1800"
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_dsv3_accuracy.py")
    port = get_available_port()
    os.system(f"kill -9 $(lsof -i:{port} | " + "awk '{print $2}')")
    time.sleep(1.0)
    # Step1: quant deepseek
    return_code = os.system(
        f"msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 "
        f"--master_port={port} --join=True --log_dir=./test_ptq_quant_dsv3_4p_logs "
        f"pytest -sv {run_file}::test_quant_deepseek"
    )
    time.sleep(1.0)
    assert return_code == 0
    # Step2: unify safetensors
    return_code = os.system(
        f"pytest -sv {run_file}::test_unify_safetensor"
    )
    time.sleep(1.0)
    assert return_code == 0
    # step3: eval deepseek
    return_code = os.system(
        f"msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 "
        f"--master_port={port} --join=True --log_dir=./test_ptq_predict_dsv3_4p_logs "
        f"pytest -sv {run_file}::test_eval_deepseek"
    )
    time.sleep(1.0)
    assert return_code == 0
