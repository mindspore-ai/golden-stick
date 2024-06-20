#!/bin/bash
# Copyright 2024 Huawei Technologies Co., Ltd
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


config_file=$1
run_mf_path=$2
question=$3
device_id=$4
pnum=$5

did_str=${device_id}
for ((i=1; i<${pnum}; i++)); do
  did_str="${did_str},$((i+device_id))"
done

export GRAPH_OP_RUN=1
export ASCEND_RT_VISIBLE_DEVICES=${did_str}
msrun --worker_num=${pnum} --local_worker_num=${pnum} --master_port=8123 --log_dir=msrun_log \
      --join=True --cluster_time_out=300 python ${run_mf_path} --config ${config_file} --run_mode predict \
      --predict_data ${question} > log_msrun 2>&1 &
