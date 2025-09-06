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

export GSLOG=1
export MS_ENABLE_INTERNAL_KERNELS=on
export ENFORCE_EAGER=true

ds_path=$1
worker_num=${2:-16}
output_dir=${3:-quantized_model}
base_path=$(cd "$(dirname $0)"; pwd)
yaml_path=${base_path}/calibrate_deepseek3_671b.yaml
calibrate_path=${base_path}/calibrate.py
unify_path=${base_path}/unify_safetensors.py


msrun \
    --worker_num=${worker_num} \
    --local_worker_num=${worker_num} \
    --master_port=8188 \
    --cluster_time_out=300 \
    --join=True \
    --log_dir=log_calibrate \
    python $calibrate_path \
        --config_path $yaml_path \
        --output_dir $output_dir \
        --ds_path $ds_path 2>&1 | tee calibrate.log

python $unify_path \
    --input_dir $output_dir \
    --output_dir ${output_dir}_unified \
    --output_file_prefix quantized_model \
    --rank_num $worker_num 2>&1 | tee unify.log
