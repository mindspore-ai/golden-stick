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


export RANK_TABLE_FILE=/path/to/hccl.json
export START_DEVICE=0
export END_DEVICE=1

# run
for((i=${START_DEVICE}; i<=${END_DEVICE}; i++))
do
    export RANK_ID=$((i-START_DEVICE))
    export DEVICE_ID=$i
    python ./llama2_w8a16_chatbot.py \
    --config_path /path_to_yaml \
    -d ${DEVICE_ID} \
    -k /path_to_ckpt \
    -t /path_to_tokenizer \
    -p 2 \
    -q 1 > log_"${DEVICE_ID}".txt 2>&1 &
done
