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

# shellcheck disable=SC2140

BASEPATH=$(cd "$(dirname $0)"; pwd)

echo "Please prepare Ascend env, prepare a new conda env(will reinstall ms, mf, gs package in env)."
echo "Make sure vocab_file is settled in all yaml."
echo "Make sure load_checkpoint is settled in predict_llama2_13b_qckpt.yaml"
echo "Make sure following config is good for you."
# config
MS_PKG_LINK="https://repo.mindspore.cn/mindspore/mindspore/version/202504/20250418/r2.6_20250418170109_2cc2ee094b82c66da5a725509616d97be9693aff_newest/unified/aarch64/mindspore-2.6.0-cp310-cp310-linux_aarch64.whl"
MF_PKG_LINK="https://repo.mindspore.cn/mindspore/mindformers/version/202504/20250419/r1.5.0_20250419031508_49931b7b27e53de7ed75d299a11d7d4858f68856_newest/any/mindformers-1.5.0-py3-none-any.whl"
ds_type="boolq"
dataset="${BASEPATH}/ws/gs/tests/data/boolq-dataset/dev.jsonl"
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
  git clone -b r1.1.0 https://gitee.com/mindspore/golden-stick.git gs || exit 1
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

sed_dtype()
{
  f=$1
  dtype=$2
  sed -i s/"compute_dtype: \".*float16\""/"compute_dtype: \"${dtype}\""/g ${f}
  sed -i s/"layernorm_compute_type: \".*float16\""/"layernorm_compute_type: \"${dtype}\""/g ${f}
  sed -i s/"rotary_dtype: \".*float16\""/"rotary_dtype: \"${dtype}\""/g ${f}
  sed -i s/"param_init_type: \".*float16\""/"param_init_type: \"${dtype}\""/g ${f}
}

sed_qconfig()
{
  f=$1
  wqg=${8:-'per_channel'}
  group_size=${9:-"0"}
  black_opname=${10:-"[\'lm_head\', \'w2\']"}
  kqg=${11:-'per_channel'}
  use_fp=${12:-'True'}
  sed -i s/"activation_dtype: .*"/"activation_dtype: \"${2}\""/g ${f}
  sed -i s/"weight_dtype: .*"/"weight_dtype: \"${3}\""/g ${f}
  sed -i s/"kvcache_dtype: .*"/"kvcache_dtype: \"${4}\""/g ${f}
  sed -i s/"outliers_suppression: .*"/"outliers_suppression: \"${5}\""/g ${f}
  sed -i s/"precision_recovery: .*"/"precision_recovery: \"${6}\""/g ${f}
  sed -i s/"load_checkpoint: .*"/"load_checkpoint: \'${7}\'"/g ${f}
  sed -i s/"weight_quant_granularity: .*"/"weight_quant_granularity: \'${wqg}\'"/g ${f}
  sed -i s/"group_size: .*"/"group_size: ${group_size}"/g ${f}
  sed -i s/"modules_to_not_convert: .*"/"modules_to_not_convert: ${black_opname}"/g ${f}
  sed -i s/"kvcache_quant_granularity: .*"/"kvcache_quant_granularity: \'${kqg}\'"/g ${f}
  sed -i s/"use_flash_attention: .*"/"use_flash_attention: \'${use_fp}\'"/g ${f}
}

eval()
{
  unset FORCE_EAGER
  unset MS_JIT
  echo "enter test workspace."
  cd ws || exit 1
  echo "${1}, save yaml to ${2}_eval_log/"
  mkdir -p "${2}_eval_log"
  cp "${3}" "${2}_eval_log/"
  echo "msrun --worker_num=2 --local_worker_num=2 --master_port=33333 --log_dir=${2}_eval_log --join=True --cluster_time_out=300 python daily_eval.py -c ${3} -s ${dataset} -n 2000 > eval_${2}_log 2>&1 &" > "${2}_eval_log/cmd.sh"
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
  export FORCE_EAGER=true
  export MS_JIT=0
  echo "enter test workspace."
  cd ws || exit 1
  echo "${1}, save yaml to ${2}_quant_log/"
  mkdir -p "${2}_quant_log"
  cp "${3}" "${2}_quant_log/"
  echo "msrun --worker_num=2 --local_worker_num=2 --master_port=33334 --log_dir=${2}_quant_log --join=True --cluster_time_out=300 python daily_quant_ckpt.py -c ${3} -q ptq -a $4 -w $5 -k $6 -o $7 -b w2 lm_head -t ${ds_type} -s ${dataset} > quant_${2}_log 2>&1 &" > "${2}_quant_log/cmd.sh"
  msrun --worker_num=2 --local_worker_num=2 --master_port=33334 --log_dir="${2}_quant_log" --join=True --cluster_time_out=300 python daily_quant_ckpt.py -c "${3}" -q ptq -a $4 -w $5 -k $6 -o $7 -b w2 lm_head -t ${ds_type} -s ${dataset} > "quant_${2}_log" 2>&1 &
  sleep ${sleep_time}
  pid=$(ps -u | grep msrun | grep "daily_quant_ckpt.py" | grep -v grep | awk -F ' ' '{print$2}')
  echo "waiting pid ${pid}"
  tail --pid ${pid} -f "${2}_quant_log/worker_0.log"
  sleep ${sleep_time}
  cd ..
}

quant_awq()
{
  export FORCE_EAGER=true
  export MS_JIT=0
  echo "enter test workspace."
  cd ws || exit 1
  echo "${1}, save yaml to ${2}_quant_log/"
  mkdir -p "${2}_quant_log"
  cp "${3}" "${2}_quant_log/"
  echo "msrun --worker_num=2 --local_worker_num=2 --master_port=33334 --log_dir=${2}_quant_log --join=True --cluster_time_out=300 python daily_quant_ckpt.py -c ${3} -q ptq -a none -w int4 -k none -o awq -wg ${4} -g ${5} -b lm_head -t ${ds_type} -s ${dataset} > quant_${2}_log 2>&1 &" > "${2}_quant_log/cmd.sh"
  msrun --worker_num=2 --local_worker_num=2 --master_port=33334 --log_dir="${2}_quant_log" --join=True --cluster_time_out=300 python daily_quant_ckpt.py -c "${3}" -q ptq -a none -w int4 -k none -o awq -wg ${4} -g ${5} -b lm_head -t ${ds_type} -s ${dataset} > "quant_${2}_log" 2>&1 &
  sleep ${sleep_time}
  pid=$(ps -u | grep msrun | grep "daily_quant_ckpt.py" | grep -v grep | awk -F ' ' '{print$2}')
  echo "waiting pid ${pid}"
  tail --pid ${pid} -f "${2}_quant_log/worker_0.log"
  sleep ${sleep_time}
  cd ..
}

quant_gptq()
{
  export FORCE_EAGER=true
  export MS_JIT=0
  echo "enter test workspace."
  cd ws || exit 1
  echo "${1}, save yaml to ${2}_quant_log/"
  mkdir -p "${2}_quant_log"
  cp "${3}" "${2}_quant_log/"
  echo "msrun --worker_num=2 --local_worker_num=2 --master_port=33334 --log_dir=${2}_quant_log --join=True --cluster_time_out=300 python daily_quant_ckpt.py -c ${3} -q ptq -a none -w int4 -k none -p gptq -wg ${4} -g ${5} -b lm_head -t ${ds_type} -s ${dataset} > quant_${2}_log 2>&1 &" > "${2}_quant_log/cmd.sh"
  msrun --worker_num=2 --local_worker_num=2 --master_port=33334 --log_dir="${2}_quant_log" --join=True --cluster_time_out=300 python daily_quant_ckpt.py -c "${3}" -q ptq -a none -w int4 -k none -p gptq -wg ${4} -g ${5} -b lm_head -t ${ds_type} -s ${dataset} > "quant_${2}_log" 2>&1 &
  sleep ${sleep_time}
  pid=$(ps -u | grep msrun | grep "daily_quant_ckpt.py" | grep -v grep | awk -F ' ' '{print$2}')
  echo "waiting pid ${pid}"
  tail --pid ${pid} -f "${2}_quant_log/worker_0.log"
  sleep ${sleep_time}
  cd ..
}

prepare_env
############################ fp16 ############################
sed_dtype "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "float16"
sed_dtype "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "float16"
# fp16 acc
eval "eval fp16 llama2-13b" "fp16" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml"

############################ fp16->a8w8 ############################
# quant ckpt a8w8
quant "quant llama2-13b-fp16 to a8w8" "fp16-a8w8" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "int8" "int8" "none" "smooth"
# a8w8 acc
sed_qconfig "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "int8" "int8" "none" "smooth" "none" "\.\/output\/llama2_13b_ptq_smooth_a8w8_ckpt\/"
eval "eval a8w8 llama2-13b-fp16" "fp16-a8w8" "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml"

############################ fp16->a16w8 ############################
# quant ckpt a16w8
quant "quant llama2-13b-fp16 to a16w8" "fp16-a16w8" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "none" "int8" "none" "none"
# a16w8 acc
sed_qconfig "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "none" "int8" "none" "None" "none" "\.\/output\/llama2_13b_ptq_no_smooth_a16w8_ckpt\/"
eval "eval a16w8 llama2-13b-fp16" "fp16-a16w8" "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml"

############################ fp16->a8w8c8 ############################
# quant ckpt a8w8c8
quant "quant llama2-13b-fp16 to a8w8c8" "fp16-a8w8c8" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "int8" "int8" "int8" "smooth"
# a8w8c8 acc
sed_qconfig "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "int8" "int8" "int8" "smooth" "none" "\.\/output\/llama2_13b_ptq_smooth_a8w8c8_ckpt\/"
eval "eval a8w8c8 llama2-13b-fp16" "fp16-a8w8c8" "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml"

############################ fp16->a16w8c8 ############################
# quant ckpt a16w8c8
quant "quant llama2-13b-fp16 to a16w8c8" "fp16-a16w8c8" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "none" "int8" "int8" "none"
# a8w8c8 acc
sed_qconfig "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "none" "int8" "int8" "None" "none" "\.\/output\/llama2_13b_ptq_no_smooth_a16w8c8_ckpt\/"
eval "eval a16w8c8 llama2-13b-fp16" "fp16-a16w8c8" "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml"

############################ fp16->awq-pergroup-a16w4 ############################
# quant ckpt awq
quant_awq "quant llama2-13b-fp16 to awq-pergroup" "fp16-awq-pergroup" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "per_group" "128"
# awq acc
sed_qconfig "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "none" "int4" "none" "awq" "none" "\.\/output\/llama2_13b_ptq_awq_a16w4_ckpt\/" "per_group" "128" "[\'lm_head\']"
eval "eval awq-pergroup llama2-13b-fp16" "fp16-awq-pergroup" "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml"

############################ fp16->awq-perchannel-a16w4 ############################
# quant ckpt awq
quant_awq "quant llama2-13b-fp16 to awq-perchannel" "fp16-awq-perchannel" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "per_channel" "0"
# awq acc
sed_qconfig "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "none" "int4" "none" "awq" "none" "\.\/output\/llama2_13b_ptq_awq_a16w4_ckpt\/" "per_channel" "0" "[\'lm_head\']"
eval "eval awq-perchannel llama2-13b-fp16" "fp16-awq-perchannel" "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml"

############################ fp16->a16w16c8-pertoken ############################
# a16w16c8 pertoken
ckpt_path=$(grep -oP "load_checkpoint:\s*'\K[^']+" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" | sed 's/[&/\]/\\&/g')
echo ${ckpt_path}
sed_qconfig "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "none" "none" "int8" "None" "none" "${ckpt_path}" "per_channel" "0" "[\'lm_head\', \'w2\']" "per_token"
eval "eval a16w16c8-pertoken llama2-13b-fp16" "fp16-a16w16c8-pertoken" "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml"

############################ fp16->gptq-pergroup-a16w4 ############################
# quant ckpt gptq
quant_gptq "quant llama2-13b-fp16 to gptq-pergroup" "fp16-gptq-pergroup" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "per_group" "128"
# gptq acc
sed_qconfig "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "none" "int4" "none" "None" "gptq" "\.\/output\/llama2_13b_ptq_no_smooth_gptq_a16w4_ckpt\/" "per_group" "128" "[\'lm_head\']"
eval "eval gptq-pergroup llama2-13b-fp16" "fp16-gptq-pergroup" "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml"

############################ fp16->gptq-perchannel-a16w4 ############################
# quant ckpt gptq
quant_gptq "quant llama2-13b-fp16 to gptq-perchannel" "fp16-gptq-perchannel" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "per_channel" "0"
# gptq acc
sed_qconfig "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "none" "int4" "none" "None" "gptq" "\.\/output\/llama2_13b_ptq_no_smooth_gptq_a16w4_ckpt\/" "per_channel" "0" "[\'lm_head\']"
eval "eval gptq-perchannel llama2-13b-fp16" "fp16-gptq-perchannel" "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml"


############################ bf16 ############################
sed_dtype "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "bfloat16"
sed_dtype "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "bfloat16"
# bf16 acc
eval "eval bf16 llama2-13b" "bf16" "./predict_llama2_13b_qckpt.yaml"

############################ bf16->a8w8 ############################
# quant ckpt a8w8
quant "quant llama2-13b-bf16 to a8w8" "bf16-a8w8" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "int8" "int8" "none" "smooth"
# a8w8 acc
sed_qconfig "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "int8" "int8" "none" "smooth" "none" "\.\/output\/llama2_13b_ptq_smooth_a8w8_ckpt\/"
eval "eval a8w8 llama2-13b-bf16" "bf16-a8w8" "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml"

############################ bf16->a16w8 ############################
# quant ckpt a16w8
quant "quant llama2-13b-bf16 to a16w8" "bf16-a16w8" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "none" "int8" "none" "none"
# a8w8 acc
sed_qconfig "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "none" "int8" "none" "None" "none" "\.\/output\/llama2_13b_ptq_no_smooth_a16w8_ckpt\/"
eval "eval a16w8 llama2-13b-bf16" "bf16-a16w8" "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml"

conda_path=$(pip show mindspore | grep 'Location' | awk -F ' ' '{print$2}')
echo "mindspore commit:"
cat ${conda_path}/mindspore/.commit_id
echo "mindspore_gs commit:"
cat ${conda_path}/mindspore_gs/.commit_id
echo "mindformers commit:"
head -n 3 ${conda_path}/mindformers/.commit_id

echo_result()
{
  name=$1
  path=$2
  if [ -f "${path}" ]; then
    echo "----------------- ${name} ${ds_type} result -----------------"
    grep "total acc" ${path}
  fi
}

echo_result "fp16 llama2-13b" "${BASEPATH}/ws/fp16_eval_log/worker_0.log"
echo_result "fp16->a8w8 llama2-13b" "${BASEPATH}/ws/fp16-a8w8_eval_log/worker_0.log"
echo_result "fp16->a16w8 llama2-13b" "${BASEPATH}/ws/fp16-a16w8_eval_log/worker_0.log"
echo_result "fp16->a8w8c8 llama2-13b" "${BASEPATH}/ws/fp16-a8w8c8_eval_log/worker_0.log"
echo_result "fp16->a16w8c8 llama2-13b" "${BASEPATH}/ws/fp16-a16w8c8_eval_log/worker_0.log"
echo_result "fp16->a16w4-awq-pergroup llama2-13b" "${BASEPATH}/ws/fp16-awq-pergroup_eval_log/worker_0.log"
echo_result "fp16->a16w4-awq-perchannel llama2-13b" "${BASEPATH}/ws/fp16-awq-perchannel_eval_log/worker_0.log"
echo_result "fp16->a16w16c8-pertoken llama2-13b" "${BASEPATH}/ws/fp16-a16w16c8-pertoken_eval_log/worker_0.log"
echo_result "fp16->a16w4-gptq-pergroup llama2-13b" "${BASEPATH}/ws/fp16-gptq-pergroup_eval_log/worker_0.log"
echo_result "fp16->a16w4-gptq-perchannel llama2-13b" "${BASEPATH}/ws/fp16-gptq-perchannel_eval_log/worker_0.log"

echo_result "bf16 llama2-13b" "${BASEPATH}/ws/bf16_eval_log/worker_0.log"
echo_result "bf16->a8w8 llama2-13b" "${BASEPATH}/ws/bf16-a8w8_eval_log/worker_0.log"
echo_result "bf16->a16w8 llama2-13b" "${BASEPATH}/ws/bf16-a16w8_eval_log/worker_0.log"
