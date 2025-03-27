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
export MS_ALLOC_CONF="enable_vmm:True"
export MS_PARALLEL_DISPATCH_NUM=4 #2
export MS_ENABLE_SYNC_COPY_INPUT=1

mf_path=$1
quant_type=$2
worker_num=${3:-16}
yaml=${4:-${mf_path}/research/deepseek3/deepseek_r1_671b/predict_deepseek_r1_671b.yaml}
base_path=$(cd "$(dirname $0)"; pwd)
ceval_path=${base_path}/../../tests/data/ceval-dataset/dev/

export PYTHONPATH=${mf_path}:${mf_path}/research/deepseek3:${PYTHONPATH}

export MS_JIT="0"
export FORCE_EAGER="true"
msrun --worker_num=${worker_num} \
      --local_worker_num=${worker_num} \
      --master_port=8188 \
      --cluster_time_out=300 \
      --join=False \
      --log_dir=calibrate_${quant_type}_log \
      python calibrate.py \
            --config ${yaml} \
            --approach=${quant_type} \
            -t ceval \
            -s ${ceval_path} > log_calibrate_${quant_type} 2>&1 &
