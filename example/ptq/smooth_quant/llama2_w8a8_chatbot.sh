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


SR=$1
ER=$2

export RANK_TABLE_FILE=hccl_2p_${SR}${ER}_10.170.27.69.json
config_file="./run_llama2_7b_910b.yaml"
tokenizer_file="tokenizer.model"

export RANK_ID=0
export DEVICE_ID=${SR}
python llama2_w8a8_chatbot.py -c ${config_file} -k "llama2-w8a8-dev0.ckpt" -d ${DEVICE_ID} -r ${RANK_ID} -t ${tokenizer_file} -q 1 > chat-log${RANK_ID} 2>&1 &
export RANK_ID=1
export DEVICE_ID=${ER}
python llama2_w8a8_chatbot.py -c ${config_file} -k "llama2-w8a8-dev1.ckpt" -d ${DEVICE_ID} -r ${RANK_ID} -t ${tokenizer_file} -q 1 > chat-log${RANK_ID} 2>&1 &

pid=$(ps -u | grep "python llama2_w8a8_chatbot.py -c" | grep -v grep | head -n 1 | awk -F ' ' '{print $2}')
tail -f --pid=$pid chat-log0
