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
export DEVICE_NUM_PER_NODE=8
export MS_SUBMODULE_LOG_v="{PARSER:2,ANALYZER:2}"
export HCCL_OP_EXPANSION_MODE="AI_CPU"

# 310P 新增
export MS_ENABLE_INTERNAL_BOOST=off
export MS_ENABLE_TRACE_MEMORY=off

# 关闭 ShapeReshapeFusion
export DISABLE_SHAPE_RESAHPE=on
export MS_ALLOC_CONF=enable_vmm:true
export MS_DEV_RUNTIME_CONF="only_local_comm:true"
export MS_NODE_TIMEOUT=20000

mf_path=$1
dataset_path=$2
worker_num=$3
master_ip=$4
rank_node=$5
export DEVICE_NUM_PER_NODE=${worker_num}
base_path=$(cd "$(dirname $0)"; pwd)
yaml=${base_path}/predict_deepseek_r1_671b_qinfer.yaml
eval_ceval_path=${base_path}/../eval_ceval.py

export PYTHONPATH=${mf_path}:${PYTHONPATH}

msrun --worker_num=24 \
      --local_worker_num=${worker_num} \
      --master_addr=${master_ip} \
      --rank_node=${rank_node} \
      --master_port=8188 \
      --cluster_time_out=300 \
      --join=False \
      --log_dir=eval_ceval_300iduo_log \
      python ${eval_ceval_path} \
             --config ${yaml} \
             --dataset_path ${dataset_path} \
             --approach dsquant > log_eval_ceval_300iduo 2>&1 &
