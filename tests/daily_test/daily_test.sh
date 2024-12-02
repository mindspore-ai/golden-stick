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

echo "Require a new conda env(will reinstall some package), an Ascend env, and make sure following config is good and load_checkpoint field is settled in yaml."
# config
MF_PKG_LINK="https://repo.mindspore.cn/mindspore/mindformers/version/202411/20241106/dev_20241106220046_3bf27eb9fe9c95cb364ddee91152d0723b3e29ca_newest/any/mindformers-1.3.0-py3-none-any.whl"
MS_PKG_LINK="https://repo.mindspore.cn/mindspore/mindspore/version/202411/20241126/master_20241126101834_a95514d8ad1bfa5c2cfb72a7142df7d19a638dbd_newest/unified/aarch64/mindspore-2.4.0-cp310-cp310-linux_aarch64.whl"
ds_type="boolq"
dataset="./gs/tests/data/boolq-dataset/dev.jsonl"
eval_script="eval_boolq.py"
export ASCEND_RT_VISIBLE_DEVICES=6,7
export GSLOG=1
sleep_time=10

prepare_env()
{
  echo "create test workspace."
  mkdir -p ws || exit 1
  echo "enter test workspace."
  cd ws || exit 1
  rm -rf gs *.whl *log* *yaml output graph *json kernel_meta *py
  cp ../*.yaml ./
  echo "download mf pkg ${MF_PKG_LINK}."
  wget --no-check-certificate $MF_PKG_LINK || exit 1
  echo "download ms pkg ${MS_PKG_LINK}."
  wget --no-check-certificate $MS_PKG_LINK || exit 1
  echo "clone gs repo."
  git clone https://gitee.com/mindspore/golden-stick.git gs || exit 1
  cd gs || exit 1
  echo "build gs."
  bash build.sh || exit 1
  mv ./output/mindspore*whl ../ || exit 1
  echo "cp quant ckpt script"
  cp ./example/ptq/quant_ckpt.py ../daily_quant_ckpt.py || exit 1
  echo "cp eval script"
  cp "./example/ptq/${eval_script}" ../daily_eval.py || exit 1
  cd .. || exit 1
  echo "uninstall pkgs"
  pip uninstall mindspore mindformers mindspore-gs -y || exit 1
  mf_pkg=$(find ./ -name "mindformers-*.whl")
  echo "install mf ${mf_pkg}"
  pip install ./${mf_pkg} || exit 1
  ms_pkg=$(find ./ -name "mindspore-*.whl")
  echo "install ms ${ms_pkg}"
  pip install ./${ms_pkg} || exit 1
  gs_pkg=$(find ./ -name "mindspore_gs-*.whl")
  echo "install gs ${gs_pkg}"
  pip install ./${gs_pkg} || exit 1
  cd ..
}

eval()
{
  echo "enter test workspace."
  cd ws || exit 1
  echo "${1}"
  msrun --worker_num=2 --local_worker_num=2 --master_port=33333 --log_dir="${2}_eval_log" --join=True --cluster_time_out=300 python daily_eval.py -c "${3}" -s ${dataset} -n 2000 > "eval_${2}_log" 2>&1 &
  sleep ${sleep_time}
  pid=$(ps -u | grep msrun | grep "daily_eval.py" | grep -v grep | awk -F ' ' '{print$2}')
  echo "waiting pid ${pid}"
  tail --pid ${pid} -f "${2}_eval_log/worker_0.log"
  sleep ${sleep_time}
  cd ..
}

quant()
{
  echo "enter test workspace."
  cd ws || exit 1
  echo "${1}"
  msrun --worker_num=2 --local_worker_num=2 --master_port=33334 --log_dir="${2}_quant_log" --join=True --cluster_time_out=300 python daily_quant_ckpt.py -c "${3}" -q ptq -w int8 -a int8 -k int8 -o smooth -b w2 lm_head -t ${ds_type} -s ${dataset} > "quant_${2}_log" 2>&1 &
  sleep ${sleep_time}
  pid=$(ps -u | grep msrun | grep "daily_quant_ckpt.py" | grep -v grep | awk -F ' ' '{print$2}')
  echo "waiting pid ${pid}"
  tail --pid ${pid} -f "${2}_quant_log/worker_0.log"
  sleep ${sleep_time}
  cd ..
}

prepare_env
############################ fp16 ############################
# fp16 acc
eval "eval fp16 llama2-13b" "fp16" "./predict_llama2_13b_fp16.yaml"

############################ fp16->a8w8c8 ############################
# quant ckpt a8w8c8
quant "quant llama2-13b-fp16 to a8w8c8" "fp16-a8w8c8" "./predict_llama2_13b_fp16.yaml"
# a8w8c8 acc
eval "eval a8w8c8 llama2-13b-fp16" "fp16-a8w8c8" "./predict_llama2_13b_fp16_a8w8c8.yaml"

############################ fp16->a16w8c8 ############################
# quant ckpt a16w8c8
quant "quant llama2-13b-fp16 to a16w8c8" "fp16-a16w8c8" "./predict_llama2_13b_fp16.yaml"
# a8w8c8 acc
eval "eval a16w8c8 llama2-13b-fp16" "fp16-a16w8c8" "./predict_llama2_13b_fp16_a16w8c8.yaml"


############################ bf16 ############################
# bf16 acc
eval "eval bf16 llama2-13b" "bf16" "./predict_llama2_13b_bf16.yaml"

############################ bf16->a8w8c8 ############################
# quant ckpt a8w8c8
quant "quant llama2-13b-bf16 to a8w8c8" "bf16-a8w8c8" "./predict_llama2_13b_bf16.yaml"
# a8w8c8 acc
eval "eval a8w8c8 llama2-13b-bf16" "bf16-a8w8c8" "./predict_llama2_13b_bf16_a8w8c8.yaml"

############################ bf16->a16w8c8 ############################
# quant ckpt a16w8c8
quant "quant llama2-13b-bf16 to a16w8c8" "bf16-a16w8c8" "./predict_llama2_13b_bf16.yaml"
# a8w8c8 acc
eval "eval a16w8c8 llama2-13b-bf16" "bf16-a16w8c8" "./predict_llama2_13b_bf16_a16w8c8.yaml"

echo "fp16 llama2-13b ${ds_type} result:"
tail -n 3 fp16_eval_log/worker_0.log
echo "a8w8c8 llama2-13b-fp16 ${ds_type} result:"
tail -n 3 fp16-a8w8c8_eval_log/worker_0.log
echo "a16w8c8 llama2-13b-fp16 ${ds_type} result:"
tail -n 3 fp16-a16w8c8_eval_log/worker_0.log

echo "bf16 llama2-13b ${ds_type} result:"
tail -n 3 bf16_eval_log/worker_0.log
echo "a8w8c8 llama2-13b-bf16 ${ds_type} result:"
tail -n 3 bf16-a8w8c8_eval_log/worker_0.log
echo "a16w8c8 llama2-13b-bf16 ${ds_type} result:"
tail -n 3 bf16-a16w8c8_eval_log/worker_0.log
