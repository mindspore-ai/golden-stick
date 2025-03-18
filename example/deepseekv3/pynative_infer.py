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
"""pynative network."""

import time
from typing import Tuple
import argparse
import tqdm
from mindspore.nn import Cell
from mindspore import Tensor, mint, ops

from mindspore_gs.common.utils import offload_network, value_check
from mindspore_gs.ptq.processor import Processor
from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq.quant import InputCatcher

from mindformers.experimental.infer.core.norm import RMSNorm
from mindformers.experimental.infer.core.layers import ColumnParallelLinear
from research.deepseek3.deepseek3_model_infer import DeepseekV3DecodeLayer
from ds_utils import create_network


input_questions = ['介绍下北京故宫', 'I love Beijing, because']


def get_network_layers(network: Cell):
    """
    Get network layers from network.

    Args:
        network (nn.Cell): Network to get network layers.

    Returns:
        A list of tuples (cell_name, `Cell`) of network.
    """
    value_check('network', network, Cell)

    class NetworkWalker(Processor):
        def __init__(self):
            self.layers = []

        def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
            if isinstance(cell, (DeepseekV3DecodeLayer, RMSNorm, ColumnParallelLinear)):
                self.layers.append((cell_name, cell))
                return cell, True
            return cell, False

    walker = NetworkWalker()
    walker.process(network)
    if walker.layers:
        layers = walker.layers
    else:
        layers = [("network", network)]
        logger.warning("No decoder layer found in network.")

    return layers


def get_first_layer_input(network: Cell, layers, input_ids=None):
    """get first layer input"""
    catcher = InputCatcher()
    catcher.patch(layers[0][1])
    try:
        network.generate(input_ids, max_new_tokens=1)
    except GeneratorExit:
        if hasattr(network, "block_mgr") and network.block_mgr:
            network.block_mgr.clear_cache()
    catcher.recover()
    offload_network(network)
    return catcher, network


def generate_input_id(network, layers, input_ids):
    """generate_input_id"""
    start_time = time.time()
    catcher, network = get_first_layer_input(network, layers, input_ids)
    logger.info(f"_get_first_layer_input time cost {time.time() - start_time}")
    arg = catcher.args[0]
    kwargs = catcher.kwargs[0]
    output = None
    for i in tqdm.tqdm(range(len(layers)), desc="each layer infer..."):
        layer_name, layer = layers[i]
        logger.info(f"{i}th layer {layer_name} start infer...")
        start_time = time.time()
        if isinstance(layer, DeepseekV3DecodeLayer):
            output = layer(*arg, **kwargs)
        elif isinstance(layer, RMSNorm):
            output = layer(arg[0])
            batch_valid_length = kwargs["batch_valid_length"]
            batch_valid_length = mint.cumsum(batch_valid_length, 0)
            output = ops.gather(output, ops.sub(batch_valid_length, 1), 1)
        else:
            output = layer(arg[0])
        arg[0] = output[0] if isinstance(output, tuple) else output
        end_time = time.time()
        logger.info(f"{i}th layer infer time cost {end_time - start_time}")
        offload_network(layer)
        logger.info(f"{i}th layer offload network time cost {time.time() - end_time}")
    return output


def pynative_generate(yaml_file, quant_type):
    """pynative_generate"""
    tokenizer, network = create_network(yaml_file, quant_type=quant_type)
    layers = get_network_layers(network)
    multi_inputs = []
    for question in input_questions:
        message = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': question}
        ]
        input_ids = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, max_length=64)
        multi_inputs.append(input_ids)
    for batch_input in multi_inputs:
        output = generate_input_id(network, layers, batch_input)
        output = Tensor(output.reshape((-1, output.shape[2])))
        output = mint.argmax(output, -1)
        answer = tokenizer.decode(output)
        print("answer:", answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--quant_type', default=None, type=str)
    args = parser.parse_args()
    pynative_generate(args.config, args.quant_type)
