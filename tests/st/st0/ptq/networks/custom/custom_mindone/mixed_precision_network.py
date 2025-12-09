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
"""Create mixed precision mindone network for PTQ accuracy test"""

import numpy as np
from mindspore import nn, mint, Tensor
from mindspore import ops as msops

from mindspore_gs.ptq.models.mindone_models.mindone_model import MindOneModel, SmoothLayerInfo


class LinearModule(nn.Cell):
    """Linear module that wraps a single linear layer"""
    def __init__(self, linear_spec):
        super().__init__()
        self.compute_dtype = linear_spec.compute_dtype

        self.pre_layer = nn.Dense(linear_spec.input_size,
                                  linear_spec.output_size,
                                  has_bias=linear_spec.has_bias,
                                  dtype=linear_spec.compute_dtype)

        if linear_spec.linear_type == 'nn.Dense':
            self.linear = nn.Dense(linear_spec.input_size,
                                   linear_spec.output_size,
                                   has_bias=linear_spec.has_bias,
                                   dtype=self.compute_dtype)
        elif linear_spec.linear_type == 'mint.nn.Linear':
            self.linear = mint.nn.Linear(linear_spec.input_size,
                                         linear_spec.output_size,
                                         bias=linear_spec.has_bias,
                                         dtype=self.compute_dtype)
        else:
            raise ValueError(f"Unsupported linear type: {linear_spec.linear_type}")

    def construct(self, x):
        x = x.astype(self.compute_dtype)
        x = self.pre_layer(x)
        output = self.linear(x)
        return output


class DecoderLayer(nn.Cell):
    """Decoder layer that wraps a single linear layer"""
    def __init__(self, linear_specs):
        super().__init__()
        self.linear_modules = nn.CellList(
            [LinearModule(linear_spec) for linear_spec in linear_specs])

    def construct(self, x):
        output_list = []
        for linear_module in self.linear_modules:
            output = linear_module(x)
            output_list.append(output)
        return msops.concat(output_list, axis=-1)


class MixedPrecisionNetwork(nn.Cell):
    """Mixed precision network for PTQ accuracy test"""

    def __init__(self, linear_specs):
        super().__init__()
        self.linear_specs = linear_specs
        self.layer = DecoderLayer(linear_specs)

    def construct(self, x):
        return self.layer(x)

    def get_outputs_dict(self, x):
        """Get outputs as dictionary mapping layer indices to outputs"""
        outputs = {}
        for layer_idx, layer in enumerate(self.layer.linear_modules):
            output = layer(x)
            outputs[layer_idx] = output
        return outputs


def init_network_weights(network, base_seed=42):
    """Initialize network weights with fixed deterministic random values in range [-0.01, 0.01]
    
    Args:
        network: The MixedPrecisionNetwork instance
        base_seed: Base seed for random number generation
    """
    weight_counter = 0

    def init_weight(weight, layer_idx):
        """Initialize a single weight tensor"""
        nonlocal weight_counter
        weight_shape = weight.shape
        weight_dtype = weight.dtype

        # Convert dtype to numpy dtype
        if hasattr(weight_dtype, 'as_numpy_dtype'):
            np_dtype = weight_dtype.as_numpy_dtype()
        else:
            dtype_str = str(weight_dtype)
            if 'float32' in dtype_str:
                np_dtype = np.float32
            elif 'float16' in dtype_str:
                np_dtype = np.float16
            elif 'bfloat16' in dtype_str:
                np_dtype = np.float32
            else:
                np_dtype = np.float32

        # Generate fixed random seed based on layer index and weight counter
        seed = base_seed + layer_idx * 1000 + weight_counter
        weight_counter += 1

        # Generate random weights in range [-0.01, 0.01]
        np.random.seed(seed)
        weight_data = np.random.uniform(-0.01, 0.01, size=weight_shape).astype(np_dtype)

        return Tensor(weight_data, dtype=weight_dtype)

    # Initialize weights for each decoder layer
    for layer_idx, layer in enumerate(network.layer.linear_modules):
        pre_layer = layer.pre_layer
        linear = layer.linear

        # Initialize weight
        if hasattr(pre_layer, 'weight'):
            pre_layer.weight.set_data(init_weight(pre_layer.weight, layer_idx))
        if hasattr(linear, 'weight'):
            linear.weight.set_data(init_weight(linear.weight, layer_idx))

        # Initialize bias if exists
        if hasattr(pre_layer, 'bias') and pre_layer.bias is not None:
            pre_layer.bias.set_data(init_weight(pre_layer.bias, layer_idx))
        if hasattr(linear, 'bias') and linear.bias is not None:
            linear.bias.set_data(init_weight(linear.bias, layer_idx))


class MixedPrecisionMindOneNetwork(MindOneModel):
    """Mixed precision mindone network for PTQ accuracy test"""
    def __init__(self, linear_specs):
        super().__init__()
        self.network = MixedPrecisionNetwork(linear_specs)
        init_network_weights(self.network, base_seed=42)
        self.is_gqa = False
        self.num_attention_heads, self.num_key_value_heads = 0, 0

    def get_layers_for_smooth(self, decoder_layer):
        """Get layers for smooth operation"""
        layers_info = []

        for linear_module in decoder_layer.linear_modules:
            layers_info.append(
                SmoothLayerInfo(
                    prev_layer=linear_module.pre_layer,
                    curr_layer=[linear_module.linear],
                )
            )
        return layers_info

    def _transformer_layers(self) -> tuple[type]:
        return [DecoderLayer]

    def forward(self, inputs):
        return self.network(inputs['input_ids'])
