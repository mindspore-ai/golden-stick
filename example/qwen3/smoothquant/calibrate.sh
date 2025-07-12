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
export FORCE_EAGER=True

mf_path=$1
ceval_path=$2
quant_type=${3:-smoothquant}
worker_num=${4:-2}
yaml=${5:-../predict_qwen3_8b_instruct_calibrate.yaml}

export PYTHONPATH=${mf_path}:${PYTHONPATH}

msrun --worker_num=${worker_num} \
      --local_worker_num=${worker_num} \
      --master_port=8188 \
      --cluster_time_out=300 \
      --join=False \
      --log_dir=calibrate_log \
      python ../calibrate.py \
               --config ${yaml} \
               --approach ${quant_type} \
               --dataset_type ceval \
               --dataset_path ${ceval_path} > log_calibrate 2>&1 &
