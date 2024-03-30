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


config_file="./run_llama2_7b_910b.yaml"
tokenizer_file="tokenizer.model"
calib_ds_file="wiki.valid.tokens"

rm -rf msrun_log
mkdir msrun_log

msrun --worker_num=4 --local_worker_num=4 --master_port=8118 --log_dir=msrun_log --join=True --cluster_time_out=300 \
      python llama2_w8a16_wikitext2.py -c ${config_file} -k "llama2-w8a16.ckpt" -t ${tokenizer_file} \
                                       -s ${calib_ds_file} -q 1 -p 4
