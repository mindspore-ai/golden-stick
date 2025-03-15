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
export MS_ENABLE_LCCL=off
export HCCL_OP_EXPANSION_MODE=AIV
export MS_DEV_RUNTIME_CONF="parallel_dispatch_kernel:True"
export HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT=TRUE
export MS_ALLOC_CONF="enable_vmm:False"
export MS_PARALLEL_DISPATCH_NUM=4 #2
export MS_ENABLE_SYNC_COPY_INPUT=1

mf_path=$1
dataset_path=$2
worker_num=${3:-16}
export DEVICE_NUM_PER_NODE=${worker_num}
base_path=$(cd "$(dirname $0)"; pwd)
yaml=${4:-${base_path}/predict_deepseek_r1_671b_a16w8.yaml}
eval_ceval_path=${base_path}/../eval_ceval.py

export PYTHONPATH=${mf_path}:${PYTHONPATH}

msrun --worker_num=${worker_num} \
      --local_worker_num=${worker_num} \
      --master_port=8188 \
      --cluster_time_out=300 \
      --join=False \
      --log_dir=eval_ceval_a16w8_log \
      python ${eval_ceval_path} \
             --config ${yaml} \
             --dataset_path ${dataset_path} \
             --approach a16w8 > log_eval_ceval_a16w8 2>&1 &
