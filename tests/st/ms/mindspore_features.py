# Copyright 2024 Huawei Technologies Co., Ltd
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
"""test some features of mindspore."""


import argparse
import os

from mindspore import context
from mindspore import dtype as mstype

from mindformers.models.llama.llama_tokenizer import LlamaTokenizer

from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQConfig, PTQMode, OutliersSuppressionType
from mindspore_gs.ptq.smooth_quant.smooth_quant import SmoothQuant
from mindspore_gs.ptq.ptq import PTQ
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper, MFParallelLlama2Helper


def add_rmsnorm_quant_fusion_test(approach):
    """test_add_rmsnorm_quant_fusion"""
    cur_dir, _ = os.path.split(os.path.abspath(__file__))
    tokenizer_path = os.path.join(cur_dir, "../../data/llama2-tokenizer.model")
    if approach == "smooth-quant":
        w8a8_config_path = "../../data/test_llama2/predict_llama2_13b_fp16_910b_1p_common_config.yaml"
        helper = MFLlama2Helper(w8a8_config_path)
    elif approach == "ptq":
        w8a8_config_path = "../../data/test_llama2/predict_parallelLlama2_13b_1p.yaml"
        helper = MFParallelLlama2Helper(w8a8_config_path)

    device_id = int(os.environ.get('DEVICE_ID', '0'))
    helper.mf_config.context.device_id = device_id
    network = helper.create_network()
    save_graphs_path = "./add_rmsnorm_quant_fusion_" + approach
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                        jit_config={"jit_level": "O0", "infer_boost": "on"}, save_graphs=True,
                        save_graphs_path=save_graphs_path)
    cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, weight_quant_dtype=mstype.int8,
                    act_quant_dtype=mstype.int8, kvcache_quant_dtype=None,
                    outliers_suppression=OutliersSuppressionType.SMOOTH,
                    opname_blacklist=["w2", "lm_head"])
    if approach == "smooth-quant":
        ptq = SmoothQuant(config=cfg)
    else:
        ptq = PTQ(config=cfg)
    network = ptq.apply(network)
    network = ptq.convert(network)

    tokenizer = LlamaTokenizer(vocab_file=tokenizer_path)
    seq_len = 100
    input_ = "Hello"
    input_ids = tokenizer(input_)['input_ids']
    network.generate(input_ids, do_sample=False, max_length=seq_len, top_p=1, top_k=3)
    res_ok = False
    all_files = os.listdir(save_graphs_path)
    for ir_file in all_files:
        if not ir_file.startswith("trace_code_graph"):
            continue
        full_ir_path = os.path.join(save_graphs_path, ir_file)
        cmd = f"grep 'PrimFunc_AddRmsNormQuantV2(' {full_ir_path}" + " | wc -l"
        with os.popen(cmd, "r") as f:
            result = f.read().strip().strip('\n')
            assert result == "1"
        cmd = f"grep 'AddRmsNorm(' {full_ir_path}" + " | wc -l"
        with os.popen(cmd, "r") as f:
            result = f.read().strip().strip('\n')
            assert result == "1"
        cmd = f"grep 'PrimFunc_RmsNorm(' {full_ir_path}" + " | wc -l"
        with os.popen(cmd, "r") as f:
            result = f.read().strip().strip('\n')
            assert result == "1"
        res_ok = True
        break
    return res_ok


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', '-a', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    uargs = get_args()
    assert add_rmsnorm_quant_fusion_test(uargs.approach)
