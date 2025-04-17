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
"""unify_safetensors"""

import argparse
import json
import os
import mindspore as ms
import numpy as np

q_lora_rank = 1536
kv_lora_rank = 512
qk_rope_head_dim = 64

def trans_int8_to_int4(ckpt_path):
    '''trans_int8_to_int4'''
    for folder_name in os.listdir(ckpt_path):
        if folder_name.endswith(".safetensors"):
            folder_path = os.path.join(ckpt_path, folder_name)
            ori_param = ms.load_checkpoint(folder_path, format="safetensors")
            for name in sorted(ori_param.keys()):
                value = ori_param[name]
                if value.dtype == ms.dtype.int8:
                    print(f"convert {name} to int4.")
                    ori_param[name] = ms.Parameter(ms.Tensor(value.asnumpy(), ms.qint4x2), name=name,
                                                   requires_grad=False)
            ms.save_checkpoint(ori_param, folder_path, format="safetensors")
            print(f"save {folder_path} int4 safetensors success.")


def split_1_dim_w_gate_hidden(value, rank_num):
    '''split_1_dim_w_gate_hidden'''
    if len(value.shape) != 1:
        raise ValueError(f"value.shape:{value.shape} 's dim != 1.")
    if value.shape[0]%rank_num != 0:
        raise ValueError(f"value.shape:{value.shape}[0] % {rank_num} != 0.")
    per_rank_len = int(value.shape[0] / rank_num)
    reshape_array = np.reshape(value, (rank_num, per_rank_len))
    half_len = int(per_rank_len / 2)
    w1 = reshape_array[:, :half_len]
    w3 = reshape_array[:, half_len:]
    return w1.flatten(), w3.flatten()


def split_2_dim_w_gate_hidden(value, rank_num):
    '''split_2_dim_w_gate_hidden'''
    if len(value.shape) != 2:
        raise ValueError(f"value.shape:{value.shape} 's dim != 2.")
    if value.shape[0]%rank_num != 0:
        raise ValueError(f"value.shape:{value.shape}[0] % {rank_num} != 0.")
    per_rank_len = int(value.shape[0] / rank_num)
    reshape_array = np.reshape(value, (rank_num, per_rank_len, value.shape[1]))
    half_len = int(per_rank_len / 2)
    w1 = reshape_array[:, :half_len, :]
    w3 = reshape_array[:, half_len:, :]
    return np.reshape(w1, (-1, value.shape[1])), np.reshape(w3, (-1, value.shape[1]))


def split_1_dim_qkv(value):
    '''split_1_dim_qkv'''
    if len(value.shape) != 1:
        raise ValueError(f"value.shape:{value.shape} 's dim != 1.")
    if value.shape[0] != q_lora_rank + kv_lora_rank + qk_rope_head_dim:
        raise ValueError(f"value.shape[0]:{value.shape[0]} != {q_lora_rank + kv_lora_rank + qk_rope_head_dim}.")
    q2l = value[:q_lora_rank]
    kv2l = value[q_lora_rank:]
    return q2l, kv2l


def split_2_dim_qkv(value):
    '''split_2_dim_qkv'''
    global kv_lora_rank
    global q_lora_rank
    global qk_rope_head_dim
    if len(value.shape) != 2:
        raise ValueError(f"value.shape:{value.shape} 's dim != 2.")
    if value.shape[0] != q_lora_rank + kv_lora_rank + qk_rope_head_dim:
        raise ValueError(f"value.shape[0]:{value.shape[0]} != {q_lora_rank + kv_lora_rank + qk_rope_head_dim}.")
    q2l = value[:q_lora_rank, :]
    kv2l = value[q_lora_rank:, :]
    return q2l, kv2l


def split_routed_expert_2_dim_w_gate_hidden(value, rank_num):
    '''split_routed_expert_2_dim_w_gate_hidden'''
    if len(value.shape) != 2:
        raise ValueError(f"value.shape:{value.shape} 's dim != 2.")
    if value.shape[1]%rank_num != 0:
        raise ValueError(f"value.shape:{value.shape}[1] % {rank_num} != 0.")
    per_rank_len = int(value.shape[1] / rank_num)
    reshape_array = np.reshape(value, (value.shape[0], rank_num, per_rank_len))
    half_len = int(per_rank_len / 2)
    w1 = reshape_array[:, :, :half_len]
    w3 = reshape_array[:, :, half_len:]
    return np.reshape(w1, (value.shape[0], -1)), np.reshape(w3, (value.shape[0], -1))


def split_3_dim_w_gate_hidden(value, rank_num):
    '''split_3_dim_w_gate_hidden'''
    if len(value.shape) != 3:
        raise ValueError(f"value.shape:{value.shape} 's dim != 3.")
    if value.shape[2]%rank_num != 0:
        raise ValueError(f"value.shape:{value.shape}[2] % {rank_num} != 0.")
    per_rank_len = int(value.shape[2] / rank_num)
    reshape_array = np.reshape(value, (value.shape[0], value.shape[1], rank_num, per_rank_len))
    half_len = int(per_rank_len / 2)
    w1 = reshape_array[:, :, :, :half_len]
    w3 = reshape_array[:, :, :, half_len:]
    return np.reshape(w1, (value.shape[0], value.shape[1], -1)), np.reshape(w3, (value.shape[0], value.shape[1], -1))


def split_w_gate_hidden_gmm_bias(value, rank_num):
    '''split_3_dim_w_gate_hidden'''
    if len(value.shape) > 2:
        raise ValueError(f"value.shape:{value.shape} 's dim > 2.")
    if value.shape[1]%rank_num != 0:
        raise ValueError(f"value.shape:{value.shape}[1] % {rank_num} != 0.")
    per_rank_len = int(value.shape[1] / rank_num)
    reshape_array = np.reshape(value, (value.shape[0], rank_num, per_rank_len))
    half_len = int(per_rank_len / 2)
    w1 = reshape_array[:, :, :half_len]
    w3 = reshape_array[:, :, half_len:]
    return np.reshape(w1, (value.shape[0], -1)), np.reshape(w3, (value.shape[0], -1))


def process_feed_forward_w_gate_hidden(name, ori_param, rank_num):
    '''process_feed_forward_w_gate_hidden'''
    if "matmul" in name:
        param_dtype = ori_param[name].dtype
        if param_dtype == ms.bfloat16:
            value = ori_param[name].astype(ms.float32).asnumpy()
        else:
            value = ori_param[name].asnumpy()
        value = ori_param[name].asnumpy()
        w1, w3 = split_1_dim_w_gate_hidden(value, rank_num)
        w1_name = name.replace("w_gate_hidden", "w1")
        w3_name = name.replace("w_gate_hidden", "w3")
        ori_param.pop(name)
        ori_param[w1_name] = ms.Parameter(ms.Tensor(w1, param_dtype), name=w1_name,
                                          requires_grad=False)
        ori_param[w3_name] = ms.Parameter(ms.Tensor(w3, param_dtype), name=w3_name,
                                          requires_grad=False)
        return
    if "weight" in name:
        param_dtype = ori_param[name].dtype
        if param_dtype == ms.bfloat16:
            raise ValueError(f"param_dtype:{param_dtype} == ms.bfloat16.")
        value = ori_param[name].asnumpy()
        w1, w3 = split_2_dim_w_gate_hidden(value, rank_num)
        w1_name = name.replace("w_gate_hidden", "w1")
        w3_name = name.replace("w_gate_hidden", "w3")
        ori_param.pop(name)
        ori_param[w1_name] = ms.Parameter(ms.Tensor(w1, param_dtype), name=w1_name,
                                          requires_grad=False)
        ori_param[w3_name] = ms.Parameter(ms.Tensor(w3, param_dtype), name=w3_name,
                                          requires_grad=False)
        return


def process_qkv_split(name, ori_param):
    '''process_qkv_split'''
    global kv_lora_rank
    global q_lora_rank
    global qk_rope_head_dim

    if "matmul" in name:
        param_dtype = ori_param[name].dtype
        if param_dtype == ms.bfloat16:
            raise ValueError(f"param_dtype:{param_dtype} == ms.bfloat16.")
        value = ori_param[name].asnumpy()
        q2l, kv2l = split_1_dim_qkv(value)
        q2l_name = name.replace("qkv2l", "q2l_proj")
        kv2l_name = name.replace("qkv2l", "kv2l")
        ori_param.pop(name)
        ori_param[q2l_name] = ms.Parameter(ms.Tensor(q2l, param_dtype), name=q2l_name,
                                           requires_grad=False)
        ori_param[kv2l_name] = ms.Parameter(ms.Tensor(kv2l, param_dtype), name=kv2l_name,
                                            requires_grad=False)
        return
    if "weight" in name:
        param_dtype = ori_param[name].dtype
        if param_dtype == ms.bfloat16:
            raise ValueError(f"param_dtype:{param_dtype} == ms.bfloat16.")
        value = ori_param[name].asnumpy()
        q2l, kv2l = split_2_dim_qkv(value)
        q2l_name = name.replace("qkv2l", "q2l_proj")
        kv2l_name = name.replace("qkv2l", "kv2l")
        ori_param.pop(name)
        ori_param[q2l_name] = ms.Parameter(ms.Tensor(q2l, param_dtype), name=q2l_name,
                                           requires_grad=False)
        ori_param[kv2l_name] = ms.Parameter(ms.Tensor(kv2l, param_dtype), name=kv2l_name,
                                            requires_grad=False)
        return
    if "quant_op" in name:
        param_dtype = ori_param[name].dtype
        if param_dtype == ms.bfloat16:
            value = ori_param[name].astype(ms.float32).asnumpy()
        else:
            value = ori_param[name].asnumpy()
        q2l_name = name.replace("qkv2l", "q2l_proj")
        kv2l_name = name.replace("qkv2l", "kv2l")
        ori_param.pop(name)
        ori_param[q2l_name] = ms.Parameter(ms.Tensor(value, param_dtype), name=q2l_name,
                                           requires_grad=False)
        ori_param[kv2l_name] = ms.Parameter(ms.Tensor(value, param_dtype), name=kv2l_name,
                                            requires_grad=False)


def process_routed_experts_w_gate_hidden(name, ori_param, rank_num):
    '''process_routed_experts_w_gate_hidden'''
    if "matmul" in name:
        param_dtype = ori_param[name].dtype
        if param_dtype == ms.bfloat16:
            value = ori_param[name].astype(ms.float32).asnumpy()
        else:
            value = ori_param[name].asnumpy()
        w1, w3 = split_routed_expert_2_dim_w_gate_hidden(value, rank_num)
        w1_name = name.replace("w_gate_hidden", "w1")
        w3_name = name.replace("w_gate_hidden", "w3")
        ori_param.pop(name)
        ori_param[w1_name] = ms.Parameter(ms.Tensor(w1, param_dtype), name=w1_name,
                                          requires_grad=False)
        ori_param[w3_name] = ms.Parameter(ms.Tensor(w3, param_dtype), name=w3_name,
                                          requires_grad=False)
        return
    if "weight" in name:
        param_dtype = ori_param[name].dtype
        if param_dtype == ms.bfloat16:
            raise ValueError(f"param_dtype:{param_dtype} == ms.bfloat16.")
        value = ori_param[name].asnumpy()
        w1, w3 = split_3_dim_w_gate_hidden(value, rank_num)
        w1_name = name.replace("w_gate_hidden", "w1")
        w3_name = name.replace("w_gate_hidden", "w3")
        ori_param.pop(name)
        ori_param[w1_name] = ms.Parameter(ms.Tensor(w1, param_dtype), name=w1_name,
                                          requires_grad=False)
        ori_param[w3_name] = ms.Parameter(ms.Tensor(w3, param_dtype), name=w3_name,
                                          requires_grad=False)
        return


def process_routed_experts_w_gate_hidden_pergroup_quant(name, ori_param, rank_num):
    '''process_routed_experts_w_gate_hidden_pergroup'''
    param_dtype = ori_param[name].dtype
    if param_dtype == ms.bfloat16:
        value = ori_param[name].astype(ms.float32).asnumpy()
    else:
        value = ori_param[name].asnumpy()

    if "gmm_bias" in name:
        w1, w3 = split_w_gate_hidden_gmm_bias(value, rank_num)
    else:
        w1, w3 = split_3_dim_w_gate_hidden(value, rank_num)
    w1_name = name.replace("w_gate_hidden", "w1")
    w3_name = name.replace("w_gate_hidden", "w3")
    ori_param.pop(name)
    ori_param[w1_name] = ms.Parameter(ms.Tensor(w1, param_dtype), name=w1_name,
                                      requires_grad=False)
    ori_param[w3_name] = ms.Parameter(ms.Tensor(w3, param_dtype), name=w3_name,
                                      requires_grad=False)


def process_name_map(ckpt_path, folder_name, ffn_split, qkv_split):
    """process_name_map"""
    param_json_path = os.path.join(ckpt_path, folder_name)
    with open(param_json_path, "r") as fp:
        origin_map = json.load(fp)
    weight_map = origin_map['weight_map']
    keys = list(weight_map.keys())
    for key in keys:
        value = weight_map[key]
        if ffn_split and "w_gate_hidden" in key:
            w1_key = key.replace("w_gate_hidden", "w1")
            w3_key = key.replace("w_gate_hidden", "w3")
            weight_map.pop(key)
            weight_map[w1_key] = value
            weight_map[w3_key] = value
        if qkv_split and "qkv2l" in key:
            q2l_key = key.replace("qkv2l", "q2l_proj")
            kv2l_key = key.replace("qkv2l", "kv2l")
            weight_map.pop(key)
            weight_map[q2l_key] = value
            weight_map[kv2l_key] = value
    origin_map["weight_map"] = weight_map
    with open(param_json_path, "w") as fp:
        json.dump(weight_map, fp, indent=2)


def split_for_smooth_quant(ckpt_path, rank_num, ffn_split, qkv_split):
    '''trans_int8_to_int4'''
    for folder_name in os.listdir(ckpt_path):
        if folder_name.endswith(".safetensors"):
            folder_path = os.path.join(ckpt_path, folder_name)
            ori_param = ms.load_checkpoint(folder_path, format="safetensors")
            for name in sorted(ori_param.keys()):
                if ffn_split and ("feed_forward.w_gate_hidden" in name or "shared_experts.w_gate_hidden" in name):
                    print(f"ffn split for {name}")
                    process_feed_forward_w_gate_hidden(name, ori_param, rank_num)
                if  ffn_split and ("routed_experts.ffn.w_gate_hidden" in name):
                    print(f"ffn split for {name}")
                    process_routed_experts_w_gate_hidden(name, ori_param, rank_num)
                if qkv_split and ("attention.qkv2l" in name):
                    print(f"qkv split for {name}")
                    process_qkv_split(name, ori_param)
            ms.save_checkpoint(ori_param, folder_path, format="safetensors")
            print(f"save {folder_path} int4 safetensors success.")
        if folder_name.endswith('name_map.json'):
            process_name_map(ckpt_path, folder_name, ffn_split, qkv_split)


def split_for_a8w4(ckpt_path, rank_num, ffn_split, qkv_split):
    '''split_for_a8w4'''
    for folder_name in os.listdir(ckpt_path):
        if folder_name.endswith(".safetensors"):
            folder_path = os.path.join(ckpt_path, folder_name)
            ori_param = ms.load_checkpoint(folder_path, format="safetensors")
            for name in sorted(ori_param.keys()):
                if ffn_split and ("feed_forward.w_gate_hidden" in name or "shared_experts.w_gate_hidden" in name):
                    print(f"ffn split for {name}")
                    process_feed_forward_w_gate_hidden(name, ori_param, rank_num)
                if  ffn_split and ("routed_experts.ffn.w_gate_hidden" in name):
                    print(f"ffn split for {name}")
                    process_routed_experts_w_gate_hidden_pergroup_quant(name, ori_param, rank_num)
                if qkv_split and ("attention.qkv2l" in name):
                    print(f"qkv split for {name}")
                    process_qkv_split(name, ori_param)
            ms.save_checkpoint(ori_param, folder_path, format="safetensors")
            print(f"save a8w4 {folder_path} {folder_name} safetensors success.")
        if folder_name.endswith('name_map.json'):
            process_name_map(ckpt_path, folder_name, ffn_split, qkv_split)
            print(f"save {folder_path} {folder_name} success.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', required=True, type=str)
    parser.add_argument('--src_strategy_file', required=True, type=str)
    parser.add_argument('--dst_dir', required=True, type=str)
    parser.add_argument('--int4_trans', required=False, type=bool)
    parser.add_argument('--approach', required=False, default="none", type=str)
    parser.add_argument('--ffn_split', required=False, default=False, type=bool)
    parser.add_argument('--qkv_split', required=False, default=False, type=bool)
    parser.add_argument('--rank_num', required=False, default=16, type=int)
    args = parser.parse_args()
    print(f"args:---{args}")
    ms.unified_safetensors(args.src_dir, args.src_strategy_file, args.dst_dir)
    print("unified_safetensors success.")
    if args.int4_trans:
        print("start trans_int8_to_int4")
        trans_int8_to_int4(args.dst_dir)
        print("trans_int8_to_int4 success.")
    if args.ffn_split or args.qkv_split:
        print(f"start split for {args.approach} approach. ffn_split:{args.ffn_split}, qkv_split:{args.qkv_split}")
        if args.approach == "smoothquant":
            split_for_smooth_quant(args.dst_dir, args.rank_num, args.ffn_split, args.qkv_split)
        elif args.approach == "osl":
            split_for_smooth_quant(args.dst_dir, args.rank_num, args.ffn_split, args.qkv_split)
        elif args.approach == "a8w4":
            split_for_a8w4(args.dst_dir, args.rank_num, args.ffn_split, args.qkv_split)
        else:
            print(f"not support split for {args.approach} approach. ffn_split:{args.ffn_split}," \
                  f" qkv_split:{args.qkv_split}")
