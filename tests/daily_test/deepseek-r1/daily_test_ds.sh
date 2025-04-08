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
echo "Make sure load_checkpoint is settled in predict_deepseek_r1_671b_qckpt.yaml"
echo "Make sure following config is good for you."
# config
MS_PKG_LINK="https://repo.mindspore.cn/mindspore/mindspore/version/202504/20250408/br_infer_deepseek_os_20250408004507_7e391e0536245cd8b314fe60adbb2a7206c38fd2_newest/unified/aarch64/mindspore-2.6.0-cp311-cp311-linux_aarch64.whl"

export GSLOG=1
export MS_ENABLE_LCCL=off
export HCCL_OP_EXPANSION_MODE=AIV
export MS_DEV_RUNTIME_CONF="parallel_dispatch_kernel:True"
export HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT=TRUE
export MS_ALLOC_CONF="enable_vmm:True"
export MS_PARALLEL_DISPATCH_NUM=4 #2
export MS_ENABLE_SYNC_COPY_INPUT=1

ds_type="ceval"
dataset="${BASEPATH}/ws/gs/tests/data/ceval-dataset/dev"
bfp16_model_path=${1}
atb_a8w8_model_path=${2}
vocab_file=${3}
tokenizer_file=${4}
sleep_time=10

prepare_env()
{
  echo "uninstall pkgs"
  pip uninstall mindspore mindformers mindspore-gs -y || exit 1

  echo "create test workspace."
  mkdir -p ws || exit 1

  echo "enter test workspace."
  cd ws || exit 1
  rm -rf gs *log* *yaml output graph *json kernel_meta *py
  cp ../*.yaml ./

  echo "download ms pkg ${MS_PKG_LINK}."
  wget --no-check-certificate -c $MS_PKG_LINK || exit 1

  echo "clone mf repo and install mf."
  git clone -b br_infer_deepseek_os https://gitee.com/mindspore/mindformers.git mf || exit 1
  cd mf || exit 1
  echo "build mf."
  bash build.sh || exit 1
  cd .. || exit 1

  echo "clone gs repo."
  git clone https://gitee.com/mindspore/golden-stick.git gs || exit 1
  cd gs || exit 1
  echo "build gs."
  bash build.sh || exit 1
  mv ./output/mindspore*whl ../ || exit 1

  echo "cp quant ckpt script"
  cp ./example/deepseekv3/calibrate.py ../daily_quant_ckpt.py || exit 1
  cp ./example/deepseekv3/deepseekv3_infer_parallelism.py ../ || exit 1
  cp ./example/deepseekv3/model_parallelism.py ../ || exit 1
  echo "cp eval script"
  cp ./example/deepseekv3/ds_utils.py ../ || exit 1
  cp ./example/deepseekv3/eval_ceval.py ../daily_eval.py || exit 1
  cd .. || exit 1

  ms_pkg=$(find ./ -name "mindspore-*.whl")
  echo "install ms ${ms_pkg}"
  pip install ./${ms_pkg} || exit 1

  gs_pkg=$(find ./ -name "mindspore_gs-*.whl")
  echo "install gs ${gs_pkg}"
  pip install ./${gs_pkg} || exit 1
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
  cp "${4}" "${2}_quant_log/"

  echo "msrun --worker_num=${3} --local_worker_num=${3} --master_port=33334 --log_dir="${2}_quant_log" --join=True --cluster_time_out=300 python daily_quant_ckpt.py --config ${4} --approach ${5} -t ${ds_type} -s ${dataset}" > "${2}_quant_log/cmd.sh"
  msrun --worker_num=${3} --local_worker_num=${3} --master_port=33334 --log_dir="${2}_quant_log" --join=True --cluster_time_out=300 python daily_quant_ckpt.py --config "${4}" --approach ${5} -t ${ds_type} -s ${dataset} > "quant_${2}_log" 2>&1 &
  sleep ${sleep_time}
  pid=$(ps -u | grep msrun | grep "daily_quant_ckpt.py" | grep -v grep | awk -F ' ' '{print$2}')
  echo "waiting pid ${pid}"
  tail --pid ${pid} -f "${2}_quant_log/worker_0.log"
  sleep ${sleep_time}
  cd ..
}

eval()
{
  unset FORCE_EAGER
  unset MS_JIT
  echo "enter test workspace."
  cd ws || exit 1
  echo "${1}, save yaml to ${2}_eval_log/"
  mkdir -p "${2}_eval_log"
  echo "msrun --worker_num=${3} --local_worker_num=${3} --master_port=33333 --log_dir=${2}_eval_log --join=True --cluster_time_out=300 python daily_eval.py -c ${4} -s ${dataset} -n 2000 > eval_${2}_log 2>&1 &" > "${2}_eval_log/cmd.sh"
  msrun --worker_num=${3} --local_worker_num=${3} --master_port=33333 --log_dir="${2}_eval_log" --join=True --cluster_time_out=300 python daily_eval.py --config "${4}" --dataset_path ${dataset} --approach ${5} > "eval_${2}_log" 2>&1 &
  sleep ${sleep_time}
  pid=$(ps -u | grep msrun | grep "daily_eval.py" | grep -v grep | awk -F ' ' '{print$2}')
  echo "waiting pid ${pid}"
  tail --pid ${pid} -f "${2}_eval_log/worker_0.log"
  sleep ${sleep_time}
  cd ..
}

sed_qconfig()
{
  f=$1
  sed -i s#"load_checkpoint: .*"#"load_checkpoint: \"${2}\""#g ${f}
  sed -i s#"vocab_file: .*"#"vocab_file: \"${3}\""#g ${f}
  sed -i s#"tokenizer_file: .*"#"tokenizer_file: \"${4}\""#g ${f}
  sed -i s#"model_parallel: .*"#"model_parallel: ${5}"#g ${f}
  sed -i s#"auto_trans_ckpt: .*"#"auto_trans_ckpt: ${6}"#g ${f}
}

prepare_env
############################ atb a8w8 ############################
# atb a8w8 acc
sed_qconfig "${BASEPATH}/ws/predict_deepseek_r1_671b_qinfer.yaml" ${atb_a8w8_model_path} ${vocab_file} ${tokenizer_file} 16 True
eval "eval atb a8w8 deepseek-r1" "atb-a8w8" 16 "${BASEPATH}/ws/predict_deepseek_r1_671b_qinfer.yaml" "dsquant"

############################ gptq a16w4 ############################
# quant ckpt gptq a16w4
sed_qconfig "${BASEPATH}/ws/predict_deepseek_r1_671b_qckpt.yaml" ${bfp16_model_path} ${vocab_file} ${tokenizer_file} 8 True
quant "quant deepseek-r1 bfp16 to a16w4 by gptq" "gptq-a16w4" 8 "${BASEPATH}/ws/predict_deepseek_r1_671b_qckpt.yaml" "gptq-pergroup" 
# gptq a16w4 acc
sed_qconfig "${BASEPATH}/ws/predict_deepseek_r1_671b_qinfer.yaml" "./output/DeepSeekR1_gptq-pergroup_safetensors/" ${vocab_file} ${tokenizer_file} 8 False
eval "eval gptq-a16w4 deepseek-r1" "gptq-a16w4" 8 "${BASEPATH}/ws/predict_deepseek_r1_671b_qinfer.yaml" "gptq-pergroup"

############################ conda info ############################
conda_path=$(pip show mindspore | grep 'Location' | awk -F ' ' '{print$2}')
echo "mindspore commit:"
cat ${conda_path}/mindspore/.commit_id
echo "mindspore_gs commit:"
cat ${conda_path}/mindspore_gs/.commit_id
echo "mindformers commit:"
head -n 3 ${conda_path}/mindformers/.commit_id

############################ results info ############################
echo_result()
{
  name=$1
  path=$2
  if [ -f "${path}" ]; then
    echo "----------------- ${name} ${ds_type} result -----------------"
    grep "总成绩" ${path}
  fi
}

echo_result "atb a8w8 deepseek-r1" "${BASEPATH}/ws/atb-a8w8_eval_log/worker_0.log"
echo_result "gptq a16w4 deepseek-r1" "${BASEPATH}/ws/gptq-a16w4_eval_log/worker_0.log"