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
"""ds infer."""

import argparse
from ds_utils import create_network


input_questions = ['介绍下北京故宫', 'I love Beijing, because']


def infer(yaml_file, quant_type):
    """infer"""
    auto_online_trans = quant_type is None or quant_type == 'dsquant'
    tokenizer, network = create_network(yaml_file, quant_type=quant_type, auto_online_trans=auto_online_trans)
    multi_inputs = []
    for question in input_questions:
        message = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': question}
        ]
        input_ids = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, max_length=64)
        multi_inputs.append(input_ids)
    for batch_input in multi_inputs:
        output = network.generate(batch_input, max_length=1024, do_sample=False, top_k=5, top_p=1, max_new_tokens=1024)
        answer = tokenizer.decode(output)
        print("answer:", answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--approach', required=True, type=str,
                        help="awq, smoothquant, dsquant, a16w8, a8dynw8")
    args = parser.parse_args()
    infer(args.config, args.approach)
