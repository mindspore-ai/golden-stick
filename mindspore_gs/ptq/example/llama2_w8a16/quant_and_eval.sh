#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
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

cur_dir=$(cd "$(dirname $0)"; pwd)

if [[ $# -ne 2 && $# -ne 3 && $# -ne 4 ]]; then
  echo "Please set preprocessed wikitext-2 dataset path like: bash quant_and_eval.sh /path/to/wiki.test.tokens /path/to/tokenizer.model."
  exit 1
fi

ds_path=$1
vocab_path=$2
device_id=${3-0}
config_file=${4-"${cur_dir}/llama2_70b_2l_1p.yaml"}

python llama2_w8a16_generate_task.py --config_path $config_file --device_id $device_id --dataset_path $ds_path --tokenizer_path=$vocab_path
