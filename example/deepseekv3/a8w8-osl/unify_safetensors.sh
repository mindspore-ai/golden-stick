#!/bin/bash
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

src_dir=$1
dst_dir=$2
rank_num=${3:-16}
base_path=$(cd "$(dirname $0)"; pwd)
src_strategy_file=${base_path}/ckpt_strategy.ckpt
quant_infer_path=${base_path}/../unify_safetensors.py
python ${quant_infer_path} --src_dir ${src_dir} --src_strategy_file ${src_strategy_file} --dst_dir ${dst_dir} --ffn_split True --rank_num ${rank_num} --approach "osl"