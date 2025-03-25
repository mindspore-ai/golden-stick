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
worker_num=${2:-8}
base_path=$(cd "$(dirname $0)"; pwd)
strategy_ckpt_save_dir=${base_path}
export DEVICE_NUM_PER_NODE=${worker_num}
yaml=${base_path}/predict_deepseek_r1_671b_qinfer.yaml
gen_strategy_path=${base_path}/../gen_strategy.py

export PYTHONPATH=${mf_path}:${PYTHONPATH}

msrun --worker_num=${worker_num} \
      --local_worker_num=${worker_num} \
      --master_port=8188 \
      --cluster_time_out=300 \
      --join=False \
      --log_dir=gen_strategy_log \
      python ${gen_strategy_path} \
             --config ${yaml} \
             --strategy_ckpt_save_dir ${strategy_ckpt_save_dir} \
             --approach gptq-pergroup > log_gen_strategy_gptq_pergroup 2>&1 &
