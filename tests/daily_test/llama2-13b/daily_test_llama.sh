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
ds_type="boolq"
dataset="${BASEPATH}/ws/gs/tests/data/boolq-dataset/dev.jsonl"
eval_script="eval_boolq.py"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GSLOG=1
sleep_time=60

checkpoint_path=${1}
vocab_file=${2}
MS_PKG_LINK=${3:-"https://repo.mindspore.cn/mindspore/mindspore/version/202507/20250708/r2.7.rc1_20250708024507_497949a8c21f1bdfb8b1f77f51314ef65a24125d/unified/aarch64/mindspore-2.7.0rc1-cp310-cp310-linux_aarch64.whl"}
MF_PKG_LINK=${4-"https://repo.mindspore.cn/mindspore/mindformers/version/202507/20250708/r1.6.0_20250708031508_2d79b904bd970702f358070337ff0375fcfa3e8c_newest/any/mindformers-1.6.0-py3-none-any.whl"}

prepare_env()
{
  echo "create test workspace."
  mkdir -p ws || exit 1
  echo "enter test workspace."
  cd ws || exit 1
  rm -rf gs *log* *yaml output graph *json kernel_meta *py
  cp ../*.yaml ./
  echo "download mf pkg ${MF_PKG_LINK}."
  wget --no-check-certificate -c $MF_PKG_LINK || exit 1
  echo "download ms pkg ${MS_PKG_LINK}."
  wget --no-check-certificate -c $MS_PKG_LINK || exit 1
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
}

sed_mode()
{
  f=$1
  sed -i s/"  mode: .*"/"  mode: ${2}"/g ${f}
}

sed_ckpt()
{
  f=$1
  sed -i s#"load_checkpoint: .*"#"load_checkpoint: \"${2}\""#g ${f}
}

sed_vocab_file()
{
  f=$1
  sed -i s#"vocab_file: .*"#"vocab_file: \"${2}\""#g ${f}
}

# Function to find two available NPU cards
find_available_devices()
{
  local available_devices=()
  for device_id in {0..7}; do
    # Check if the device is in use by checking if msrun is running on it
    output=$(npu-smi info -t proc-mem -i $device_id 2>&1)

    if [[ "$output" == *"No process in device."* ]]; then
      # shellcheck disable=SC2206
      available_devices+=($device_id)
    fi
  done
  echo "${available_devices[@]}"
}

set_devices()
{
  echo "Searching for available NPU devices..."
  # shellcheck disable=SC2207
  available_devices=($(find_available_devices))
  while [ ${#available_devices[@]} -lt 2 ]; do
    echo "Not enough available NPU devices. Waiting for 10 seconds..."
    sleep 10
    # shellcheck disable=SC2207
    available_devices=($(find_available_devices))
  done

  # Set the available devices
  export ASCEND_RT_VISIBLE_DEVICES="${available_devices[0]},${available_devices[1]}"
  echo "Using NPU devices: ${ASCEND_RT_VISIBLE_DEVICES}"
}

get_port()
{
  port=$((RANDOM % (10000 - 1000) + 1000))
  echo "Using port ${port}"
}

eval_nocheck()
{
  unset FORCE_EAGER
  unset MS_JIT
  echo "enter test workspace."
  cd ws || exit 1

  set_devices
  get_port

  echo "${1}, save yaml to ${2}_eval_log/"
  mkdir -p "${2}_eval_log"
  cp "${3}" "${2}_eval_log/"

  echo "msrun --worker_num=2 --local_worker_num=2 --master_port=${port} --log_dir=${2}_eval_log --join=True --cluster_time_out=300 python daily_eval.py -c ${3} -s ${dataset} -n 2000 > eval_${2}_log 2>&1 &" > "${2}_eval_log/cmd.sh"
  msrun --worker_num=2 --local_worker_num=2 --master_port=${port} --log_dir="${2}_eval_log" --join=True --cluster_time_out=300 python daily_eval.py -c "${3}" -s ${dataset} -n 2000 > "eval_${2}_log" 2>&1 &
  sleep ${sleep_time}
  cd ..
}

eval()
{
  unset FORCE_EAGER
  echo "enter test workspace."
  cd ws || exit 1

  set_devices
  get_port

  echo "${1}, save yaml to ${2}_eval_log/"
  mkdir -p "${2}_eval_log"
  cp "${3}" "${2}_eval_log/"

  timeout=3600  # 1小时的秒数
  start_time=$(date +%s)  # 获取当前时间的秒数

  while ! grep -q "Save checkpoint cost time is" "${2}_quant_log/worker_0.log"; do
    current_time=$(date +%s)  # 获取当前时间的秒数
    elapsed_time=$((current_time - start_time))  # 计算已过去的时间

    if [ $elapsed_time -ge $timeout ]; then
        echo "${2} quant process has been running for more than 2 hours. Continuing with the next steps..."
        break
    fi

    echo "${2} is in quant process. Waiting for 10 seconds..."
    sleep 10
  done

  echo "msrun --worker_num=2 --local_worker_num=2 --master_port=${port} --log_dir=${2}_eval_log --join=True --cluster_time_out=300 python daily_eval.py -c ${3} -s ${dataset} -n 2000 > eval_${2}_log 2>&1 &" > "${2}_eval_log/cmd.sh"
  msrun --worker_num=2 --local_worker_num=2 --master_port=${port} --log_dir="${2}_eval_log" --join=True --cluster_time_out=300 python daily_eval.py -c "${3}" -s ${dataset} -n 2000 > "eval_${2}_log" 2>&1 &
  sleep ${sleep_time}
  cd ..
}

eval_pynative()
{
  export FORCE_EAGER=true
  export MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST=PageAttention
  echo "enter test workspace."
  cd ws || exit 1
  echo "${1}, save yaml to ${2}_pynative_eval_log/"
  mkdir -p "${2}_pynative_eval_log"
  cp "${3}" "${2}_pynative_eval_log/"
  echo "msrun --worker_num=2 --local_worker_num=2 --master_port=${port} --log_dir=${2}_pynative_eval_log --join=True --cluster_time_out=300 python daily_eval.py -c ${3} -s ${dataset} -n 2000 > pynative_eval_${2}_log 2>&1 &" > "${2}_pynative_eval_log/cmd.sh"
  msrun --worker_num=2 --local_worker_num=2 --master_port=${port} --log_dir="${2}_pynative_eval_log" --join=True --cluster_time_out=300 python daily_eval.py -c "${3}" -s ${dataset} -n 2000 > "pynative_eval_${2}_log" 2>&1 &
  sleep ${sleep_time}
  pid=$(ps -u | grep msrun | grep "daily_eval.py" | grep -v grep | awk -F ' ' '{print$2}')
  echo "waiting pid ${pid}"
  tail --pid ${pid} -f "${2}_pynative_eval_log/worker_0.log"
  sleep ${sleep_time}
  cd ..
}

quant()
{
  export FORCE_EAGER=true
  echo "enter test workspace."
  cd ws || exit 1

  set_devices
  get_port

  echo "${1}, save yaml to ${2}_quant_log/"
  mkdir -p "${2}_quant_log"
  cp "${3}" "${2}_quant_log/"
  echo "msrun --worker_num=2 --local_worker_num=2 --master_port=${port} --log_dir=${2}_quant_log --join=True --cluster_time_out=300 python daily_quant_ckpt.py -c ${3} -q ptq -a $4 -w $5 -k $6 -o $7 -b w2 lm_head -t ${ds_type} -s ${dataset} > quant_${2}_log 2>&1 &" > "${2}_quant_log/cmd.sh"
  msrun --worker_num=2 --local_worker_num=2 --master_port=${port} --log_dir="${2}_quant_log" --join=True --cluster_time_out=300 python daily_quant_ckpt.py -c "${3}" -q ptq -a $4 -w $5 -k $6 -o $7 -b w2 lm_head -t ${ds_type} -s ${dataset} > "quant_${2}_log" 2>&1 &
  sleep ${sleep_time}
  cd ..
}

quant_awq()
{
  export FORCE_EAGER=true
  echo "enter test workspace."
  cd ws || exit 1

  set_devices
  get_port

  echo "${1}, save yaml to ${2}_quant_log/"
  mkdir -p "${2}_quant_log"
  cp "${3}" "${2}_quant_log/"
  echo "msrun --worker_num=2 --local_worker_num=2 --master_port=${port} --log_dir=${2}_quant_log --join=True --cluster_time_out=300 python daily_quant_ckpt.py -c ${3} -q ptq -a none -w int4 -k none -o awq -wg ${4} -g ${5} -b lm_head -t ${ds_type} -s ${dataset} > quant_${2}_log 2>&1 &" > "${2}_quant_log/cmd.sh"
  msrun --worker_num=2 --local_worker_num=2 --master_port=${port} --log_dir="${2}_quant_log" --join=True --cluster_time_out=300 python daily_quant_ckpt.py -c "${3}" -q ptq -a none -w int4 -k none -o awq -wg ${4} -g ${5} -b lm_head -t ${ds_type} -s ${dataset} > "quant_${2}_log" 2>&1 &
  sleep ${sleep_time}
  cd ..
}

quant_gptq()
{
  export FORCE_EAGER=true
  echo "enter test workspace."
  cd ws || exit 1

  set_devices
  get_port

  echo "${1}, save yaml to ${2}_quant_log/"
  mkdir -p "${2}_quant_log"
  cp "${3}" "${2}_quant_log/"
  echo "msrun --worker_num=2 --local_worker_num=2 --master_port=${port} --log_dir=${2}_quant_log --join=True --cluster_time_out=300 python daily_quant_ckpt.py -c ${3} -q ptq -a none -w int4 -k none -p gptq -wg ${4} -g ${5} -b lm_head -t calibrate -s "${BASEPATH}/ws/gs/tests/data/calibrate-dataset/calibrate.jsonl" > quant_${2}_log 2>&1 &" > "${2}_quant_log/cmd.sh"
  msrun --worker_num=2 --local_worker_num=2 --master_port=${port} --log_dir="${2}_quant_log" --join=True --cluster_time_out=300 python daily_quant_ckpt.py -c "${3}" -q ptq -a none -w int4 -k none -p gptq -wg ${4} -g ${5} -b lm_head -t calibrate -s "${BASEPATH}/ws/gs/tests/data/calibrate-dataset/calibrate.jsonl" > "quant_${2}_log" 2>&1 &
  sleep ${sleep_time}
  cd ..
}

prepare_env
sed_ckpt "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" ${checkpoint_path}
sed_vocab_file "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" ${vocab_file}
sed_vocab_file "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" ${vocab_file}

algo_quant_list()
{
  sed_mode "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "1"

  # quant ckpt a8w8
  quant "quant llama2-13b-fp16 to a8w8" "fp16-a8w8" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "int8" "int8" "none" "smooth"

  # quant ckpt a16w8
  quant "quant llama2-13b-fp16 to a16w8" "fp16-a16w8" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "none" "int8" "none" "none"

  # quant ckpt a8w8c8
  quant "quant llama2-13b-fp16 to a8w8c8" "fp16-a8w8c8" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "int8" "int8" "int8" "smooth"

  # quant ckpt a16w8c8
  quant "quant llama2-13b-fp16 to a16w8c8" "fp16-a16w8c8" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "none" "int8" "int8" "none"

  # quant ckpt awq-pergroup
  quant_awq "quant llama2-13b-fp16 to awq-pergroup" "fp16-awq-pergroup" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "per_group" "128"

  # quant ckpt awq-perchannel
  quant_awq "quant llama2-13b-fp16 to awq-perchannel" "fp16-awq-perchannel" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "per_channel" "0"

  # quant ckpt gptq-pergroup
  quant_gptq "quant llama2-13b-fp16 to gptq-pergroup" "fp16-gptq-pergroup" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "per_group" "128"

  # quant ckpt gptq-perchannel
  quant_gptq "quant llama2-13b-fp16 to gptq-perchannel" "fp16-gptq-perchannel" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "per_channel" "0"
}

algo_eval_list()
{
  # a8w8 acc
  sed_qconfig "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "int8" "int8" "none" "smooth" "none" "\.\/output\/llama2_13b_ptq_float16_smooth_a8_per_tensor_w8_per_channel_ckpt\/"
  eval "eval a8w8 llama2-13b-fp16" "fp16-a8w8" "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml"

  # a16w8 acc
  sed_qconfig "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "none" "int8" "none" "None" "none" "\.\/output\/llama2_13b_ptq_float16_no_smooth_a16_w8_per_channel_ckpt\/"
  eval "eval a16w8 llama2-13b-fp16" "fp16-a16w8" "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml"

  # a8w8c8 acc
  sed_qconfig "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "int8" "int8" "int8" "smooth" "none" "\.\/output\/llama2_13b_ptq_float16_smooth_a8_per_tensor_w8_per_channel_c8_per_channel_ckpt\/"
  eval "eval a8w8c8 llama2-13b-fp16" "fp16-a8w8c8" "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml"

  # a8w8c8 acc
  sed_qconfig "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "none" "int8" "int8" "None" "none" "\.\/output\/llama2_13b_ptq_float16_no_smooth_a16_w8_per_channel_c8_per_channel_ckpt\/"
  eval "eval a16w8c8 llama2-13b-fp16" "fp16-a16w8c8" "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml"

  # awq-pergroup acc
  sed_qconfig "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "none" "int4" "none" "awq" "none" "\.\/output\/llama2_13b_ptq_float16_awq_a16_w4_per_group_ckpt\/" "per_group" "128" "[\'lm_head\']"
  eval "eval awq-pergroup llama2-13b-fp16" "fp16-awq-pergroup" "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml"

  # awq-perchannel acc
  sed_qconfig "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "none" "int4" "none" "awq" "none" "\.\/output\/llama2_13b_ptq_float16_awq_a16_w4_per_channel_ckpt\/" "per_channel" "0" "[\'lm_head\']"
  eval "eval awq-perchannel llama2-13b-fp16" "fp16-awq-perchannel" "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml"

  # a16w16c8 pertoken
  ckpt_path=$(grep -oP "load_checkpoint:\s*'\K[^']+" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" | sed 's/[&/\]/\\&/g')
  echo ${ckpt_path}
  sed_qconfig "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "none" "none" "int8" "None" "none" "${ckpt_path}" "per_channel" "0" "[\'lm_head\', \'w2\']" "per_token"
  eval_nocheck "eval a16w16c8-pertoken llama2-13b-fp16" "fp16-a16w16c8-pertoken" "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml"

  # gptq-pergroup acc
  sed_qconfig "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "none" "int4" "none" "None" "gptq" "\.\/output\/llama2_13b_ptq_float16_no_smooth_gptq_a16_w4_per_group_ckpt\/" "per_group" "128" "[\'lm_head\']" "per_channel"
  eval "eval gptq-pergroup llama2-13b-fp16" "fp16-gptq-pergroup" "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml"

  # gptq-perchannel acc
  sed_qconfig "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "none" "int4" "none" "None" "gptq" "\.\/output\/llama2_13b_ptq_float16_no_smooth_gptq_a16_w4_per_channel_ckpt\/" "per_channel" "0" "[\'lm_head\']" "per_channel"
  eval "eval gptq-perchannel llama2-13b-fp16" "fp16-gptq-perchannel" "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml"
}

############################ fp16-float ############################
sed_dtype "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "float16"
sed_dtype "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "float16"
# fp16 acc
sed_mode "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "0"
eval_nocheck "eval fp16 llama2-13b" "fp16" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml"
# fp16 quant
algo_quant_list
# fp16 eval
algo_eval_list


############################ bf16 ############################
sed_dtype "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "bfloat16"
sed_dtype "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "bfloat16"
# bf16 acc
sed_mode "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "0"
eval_nocheck "eval bf16 llama2-13b" "bf16" "./predict_llama2_13b_qckpt.yaml"
sed_mode "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "1"

############################ bf16-quant-eval ############################
# quant ckpt a8w8
quant "quant llama2-13b-bf16 to a8w8" "bf16-a8w8" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "int8" "int8" "none" "smooth"
# quant ckpt a16w8
quant "quant llama2-13b-bf16 to a16w8" "bf16-a16w8" "${BASEPATH}/ws/predict_llama2_13b_qckpt.yaml" "none" "int8" "none" "none"

# a8w8 acc
sed_qconfig "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "int8" "int8" "none" "smooth" "none" "\.\/output\/llama2_13b_ptq_bfloat16_smooth_a8_per_tensor_w8_per_channel_ckpt\/"
eval "eval a8w8 llama2-13b-bf16" "bf16-a8w8" "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml"
# a16w8 acc
sed_qconfig "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml" "none" "int8" "none" "None" "none" "\.\/output\/llama2_13b_ptq_bfloat16_no_smooth_a16_w8_per_channel_ckpt\/"
eval "eval a16w8 llama2-13b-bf16" "bf16-a16w8" "${BASEPATH}/ws/predict_llama2_13b_qinfer.yaml"


echo "waiting all process done..."
# shellcheck disable=SC2207
available_devices=($(find_available_devices))
while [ ${#available_devices[@]} -lt 8 ]; do
  echo "Still have process running, Waiting for 60 seconds..."
  sleep 60
  # shellcheck disable=SC2207
  available_devices=($(find_available_devices))
done

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
echo_result "fp16 pynative llama2-13b" "${BASEPATH}/ws/fp16_pynative_eval_log/worker_0.log"
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
