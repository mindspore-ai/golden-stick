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
"""gene_strategy"""

import argparse
import os

from ds_utils import create_network
from mindspore.communication import get_rank

from mindformers import logger
from mindformers.experimental.infer.core.utils import generate_state_dict
from mindformers.experimental.parallel_core.pynative.utils import save_strategy_file


def gene_strategy(yaml_file, strategy_ckpt_save_dir, quant_type):
    """gene_strategy"""
    _, network = create_network(yaml_file, quant_type=quant_type)
    os.makedirs(strategy_ckpt_save_dir, exist_ok=True)
    strategy_file_path = os.path.join(strategy_ckpt_save_dir, "ckpt_strategy.ckpt")
    shard_state_dict = generate_state_dict(network)
    if get_rank() == 0:
        save_strategy_file(shard_state_dict, strategy_file_path)
    logger.info(f"Strategy file has been saved in {strategy_file_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--approach', required=True, type=str, help="smoothquant")
    parser.add_argument('--strategy_ckpt_save_dir', required=True, type=str)
    args = parser.parse_args()
    gene_strategy(args.config, args.strategy_ckpt_save_dir, args.approach)
