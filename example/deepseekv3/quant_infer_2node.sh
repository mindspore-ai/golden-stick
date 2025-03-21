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
node_id=$3
cur_ip=$4
master_ip=$5
yaml=${6:-${mf_path}/research/deepseek3/deepseek_r1_671b/predict_deepseek_r1_671b.yaml}

export HCCL_IF_IP=${cur_ip}
export PYTHONPATH=${mf_path}:${mf_path}/research/deepseek3:${PYTHONPATH}

bash ${mf_path}/scripts/msrun_launcher.sh "quant_infer.py --config ${yaml} --approach ${quant_type}" 16 8 ${master_ip} 8432 ${node_id} quant_infer_log False 300