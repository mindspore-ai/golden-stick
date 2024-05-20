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

BASEPATH=$(cd "$(dirname $0)"; pwd)

fp_ckpt_dir=$1
fp_strategy_file=$2
device_id=$3
pnum=$(4-4)
tokenizer_file=""

did_str=${device_id}
for ((i=1; i<${pnum}; i++)); do
  did_str="${did_str},$((i+device_id))"
done

export GRAPH_OP_RUN=1
export ASCEND_RT_VISIBLE_DEVICES=${did_str}

echo "----------------- start quantize-ing ckpt..."
python llama2_w8a16_save_ckpt.py -c "${BASEPATH}/predict_llama2_57b_quant.yaml" -k ${fp_ckpt_dir} -s ${fp_strategy_file} -n llama2_70b

echo "----------------- start split quantized ckpt..."

rm -rf split_msrun_log
mkdir split_msrun_log
rm -rf quant_ckpt
mkdir quant_ckpt
for ((i=0; i<${pnum}; i++)); do
  mkdir "quant_ckpt/rank_${i}"
done

msrun --worker_num=${pnum} --local_worker_num=${pnum} --master_port=8190 --log_dir=msrun_log --join=True --cluster_time_out=300 \
      python llama2_split_ckpt.py -c "${BASEPATH}/predict_llama2_57b_deploy.yaml" -k "llama2-w8a16.ckpt" -p ${pnum} -n llama2_70b -t ${tokenizer_file}

echo "----------------- quantized ckpt saved in ${BASEPATH}/quant_ckpt/"
