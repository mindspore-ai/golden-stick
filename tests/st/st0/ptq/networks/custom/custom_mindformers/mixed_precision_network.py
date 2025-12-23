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
"""Mixed precision network for PTQ accuracy test"""
import os
import sys

import numpy as np
from mindspore import nn, Tensor, dtype as msdtype
from mindspore import ops as msops
import mindspore.ops.operations as P

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../mindformers")))
# pylint: disable=wrong-import-position
from mindformers.parallel_core.inference.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
    QKVParallelLinear,
    MergedColumnParallelLinear,
)
from mindformers.parallel_core.inference.tensor_parallel.grouped_layers import (
    ColumnParallelGroupedLinear,
    RowParallelGroupedLinear,
)
from mindformers.parallel_core.inference.parallel_state import (
    default_pgs,
    get_tensor_model_parallel_group,
    is_initialized,
)
from mindformers.parallel_core.transformer_config import TransformerConfig

class ModelSpec:
    """Specification for the mixed precision model"""

    @staticmethod
    def get_default_layer_configs(hidden_dim, num_experts):  # pylint: disable=redefined-outer-name
        """Get default layer configurations"""
        return [
            (msdtype.float16, 'float16', 'ColumnParallelLinear', {}),
            (msdtype.bfloat16, 'bfloat16', 'RowParallelLinear', {}),
            (msdtype.float16, 'float16', 'QKVParallelLinear', {
                'head_size': hidden_dim // 8,
                'total_num_heads': 8,
                'total_num_kv_heads': 8,
            }),
            (msdtype.bfloat16, 'bfloat16', 'MergedColumnParallelLinear', {
                'ffn_hidden_size': hidden_dim,
            }),
            (msdtype.float32, 'float32', 'ColumnParallelGroupedLinear', {
                'num_local_experts': num_experts,
            }),
            (msdtype.float16, 'float16', 'RowParallelGroupedLinear', {
                'num_local_experts': num_experts,
            }),
        ]

    def __init__(self, hidden_dim, num_layers, num_experts, tensor_model_parallel_size,  # pylint: disable=redefined-outer-name
                 linear_specs):
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.tensor_model_parallel_size = tensor_model_parallel_size
        self.linear_specs = linear_specs[:num_layers] if len(linear_specs) >= num_layers else linear_specs


class GroupedLinearWrapper(nn.Cell):
    """Wrapper for grouped linear layers to handle 2D input requirement"""

    def __init__(self, linear, num_experts=2):  # pylint: disable=redefined-outer-name
        super().__init__()
        self.linear = linear
        self.num_experts = num_experts
        self.reshape = P.Reshape()

    def construct(self, x, group_list=None):
        """Forward pass with 2D input requirement handling"""
        original_shape = x.shape

        if len(original_shape) == 3:
            batch_dim, seq_dim, hidden_dim = original_shape
            x_2d = self.reshape(x, (-1, hidden_dim))
            total_elements = batch_dim * seq_dim
        elif len(original_shape) == 2:
            x_2d = x
            total_elements = original_shape[0]
        else:
            raise ValueError(f"Unsupported input shape for grouped linear: {original_shape}")

        if group_list is None:
            expert_counts = np.random.multinomial(
                total_elements,
                np.ones(self.num_experts) / self.num_experts
            )
            group_list = Tensor(expert_counts, dtype=msdtype.int64)

        output_2d = self.linear(x_2d, group_list=group_list)

        if len(original_shape) == 3:
            output_shape = (original_shape[0], original_shape[1], output_2d.shape[-1])
            return self.reshape(output_2d, output_shape)
        return output_2d


class QKVWithProjection(nn.Cell):
    """Composite layer: QKVParallelLinear + RowParallelLinear projection"""

    def __init__(self, qkv_layer, proj_layer):
        super().__init__()
        self.qkv = qkv_layer
        self.proj = proj_layer

    def construct(self, x):
        qkv_out = self.qkv(x)
        return self.proj(qkv_out)


class MergedWithProjection(nn.Cell):
    """Composite layer: MergedColumnParallelLinear + RowParallelLinear projection"""

    def __init__(self, merged_layer, proj_layer):
        super().__init__()
        self.merged = merged_layer
        self.proj = proj_layer

    def construct(self, x):
        merged_out = self.merged(x)
        return self.proj(merged_out)


class DecoderLayer(nn.Cell):
    """Decoder layer that wraps a single linear layer"""
    def __init__(self, linear, layer_idx):
        super().__init__()
        self.linear = linear
        self.layer_idx = layer_idx

    def construct(self, x, group_list=None):
        if isinstance(self.linear, GroupedLinearWrapper):
            return self.linear(x, group_list=group_list)
        return self.linear(x)

    def get_outputs_dict(self, x, group_list=None):
        """Get outputs as dictionary"""
        output_val = self.construct(x, group_list=group_list)
        return {0: output_val}


class MixedPrecisionNetwork(nn.Cell):
    """Mixed precision network for PTQ accuracy test"""

    def __init__(self, model_spec, tp_group):
        super().__init__()
        self.model_spec = model_spec
        self.hidden_size = model_spec.hidden_size
        self.num_experts = model_spec.num_experts
        self.weight_counter = 0
        self.layers = self._build_layers(tp_group)

    @staticmethod
    def _create_transformer_config(linear_spec, num_experts):  # pylint: disable=redefined-outer-name
        is_moe_layer = linear_spec.linear_type in ['ColumnParallelGroupedLinear', 'RowParallelGroupedLinear']
        return TransformerConfig(
            num_attention_heads=8,
            num_layers=1,
            params_dtype=linear_spec.param_dtype_str,
            compute_dtype=linear_spec.param_dtype_str,
            num_moe_experts=num_experts if is_moe_layer else None,
            add_bias_linear=False if is_moe_layer else None,
        )

    def _build_layers(self, tp_group):
        """Build all layers for the network"""
        layers = nn.CellList()
        prev_output_is_parallel = False

        for layer_idx, linear_spec in enumerate(self.model_spec.linear_specs):
            config = self._create_transformer_config(linear_spec, self.num_experts)
            linear, current_output_is_parallel = self._build_single_layer(
                linear_spec, config, tp_group, prev_output_is_parallel, layer_idx
            )

            self._init_weights(linear, layer_idx)

            if linear_spec.linear_type in ['ColumnParallelGroupedLinear', 'RowParallelGroupedLinear']:
                linear = GroupedLinearWrapper(linear, self.num_experts)

            layers.append(linear)
            prev_output_is_parallel = current_output_is_parallel

        return layers

    def _get_layer_sizes(self, linear_spec):
        return {
            'input_size': linear_spec.input_size if linear_spec.input_size is not None else self.hidden_size,
            'output_size': linear_spec.output_size if linear_spec.output_size is not None else self.hidden_size,
            'hidden_size': linear_spec.hidden_size if linear_spec.hidden_size is not None else self.hidden_size,
        }

    def _build_single_layer(self, linear_spec, config, tp_group, prev_output_is_parallel, layer_idx):  # pylint: disable=unused-argument
        """Build a single layer based on linear_spec"""
        layer_type = linear_spec.linear_type
        sizes = self._get_layer_sizes(linear_spec)

        builders = {
            'ColumnParallelLinear': self._build_column_parallel_linear,
            'RowParallelLinear': self._build_row_parallel_linear,
            'ReplicatedLinear': self._build_replicated_linear,
            'QKVParallelLinear': self._build_qkv_layer,
            'MergedColumnParallelLinear': self._build_merged_layer,
            'ColumnParallelGroupedLinear': self._build_column_parallel_grouped_linear,
            'RowParallelGroupedLinear': self._build_row_parallel_grouped_linear,
        }

        builder = builders.get(layer_type)
        if builder is None:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        return builder(linear_spec, config, tp_group, prev_output_is_parallel, sizes)

    def _build_column_parallel_linear(self, linear_spec, config, tp_group, prev_output_is_parallel, sizes):  # pylint: disable=invalid-name,unused-argument
        """Build ColumnParallelLinear layer"""
        linear = ColumnParallelLinear(
            input_size=sizes['input_size'],
            output_size=sizes['output_size'],
            config=config,
            compute_dtype=linear_spec.compute_dtype,
            bias=linear_spec.bias,
            gather_output=linear_spec.gather_output,
            transpose_b=linear_spec.transpose_b,
            tp_group=tp_group,
        )
        return linear, True

    def _build_row_parallel_linear(self, linear_spec, config, tp_group, prev_output_is_parallel, sizes):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Build RowParallelLinear layer"""
        linear = RowParallelLinear(
            input_size=sizes['input_size'],
            output_size=sizes['output_size'],
            config=config,
            compute_dtype=linear_spec.compute_dtype,
            bias=linear_spec.bias,
            input_is_parallel=prev_output_is_parallel,
            transpose_b=linear_spec.transpose_b,
            tp_group=tp_group,
        )
        return linear, False

    def _build_replicated_linear(self, linear_spec, config, tp_group, prev_output_is_parallel, sizes):  # pylint: disable=invalid-name,unused-argument
        """Build ReplicatedLinear layer"""
        linear = ReplicatedLinear(
            input_size=sizes['input_size'],
            output_size=sizes['output_size'],
            config=config,
            compute_dtype=linear_spec.compute_dtype,
            bias=linear_spec.bias,
            transpose_b=linear_spec.transpose_b,
        )
        return linear, False

    def _build_column_parallel_grouped_linear(self, linear_spec, config, tp_group, prev_output_is_parallel, sizes):  # pylint: disable=invalid-name,unused-argument
        """Build ColumnParallelGroupedLinear layer"""
        num_local_experts = (linear_spec.num_local_experts
                             if linear_spec.num_local_experts is not None
                             else self.num_experts)
        linear = ColumnParallelGroupedLinear(
            num_local_experts=num_local_experts,
            input_size=sizes['input_size'],
            output_size=sizes['output_size'],
            config=config,
            bias=linear_spec.bias,
            gather_output=linear_spec.gather_output,
            tp_group=tp_group,
        )
        return linear, True

    def _build_row_parallel_grouped_linear(self, linear_spec, config, tp_group, prev_output_is_parallel, sizes):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Build RowParallelGroupedLinear layer"""
        num_local_experts = (linear_spec.num_local_experts
                             if linear_spec.num_local_experts is not None
                             else self.num_experts)
        linear = RowParallelGroupedLinear(
            num_local_experts=num_local_experts,
            input_size=sizes['input_size'],
            output_size=sizes['output_size'],
            config=config,
            bias=linear_spec.bias,
            input_is_parallel=prev_output_is_parallel,
            tp_group=tp_group,
        )
        return linear, False

    def _create_projection_config(self, param_dtype_str):
        return TransformerConfig(
            num_attention_heads=8,
            num_layers=1,
            params_dtype=param_dtype_str,
            compute_dtype=param_dtype_str,
        )

    def _create_projection_layer(self, input_size, output_size, linear_spec, tp_group):
        proj_config = self._create_projection_config(linear_spec.param_dtype_str)
        return RowParallelLinear(
            input_size=input_size,
            output_size=output_size,
            config=proj_config,
            compute_dtype=linear_spec.compute_dtype,
            bias=linear_spec.bias,
            input_is_parallel=True,
            transpose_b=linear_spec.transpose_b,
            tp_group=tp_group,
        )

    def _calculate_qkv_output_size(self, total_num_heads, total_num_kv_heads, head_size, tp_group):
        tp_size_actual = tp_group.size if is_initialized() else 1
        num_heads_per_partition = total_num_heads // tp_size_actual
        if tp_size_actual >= total_num_kv_heads:
            num_kv_heads_per_partition = 1
        else:
            num_kv_heads_per_partition = total_num_kv_heads // tp_size_actual
        return (num_heads_per_partition + 2 * num_kv_heads_per_partition) * tp_size_actual * head_size

    def _build_qkv_layer(self, linear_spec, config, tp_group, prev_output_is_parallel, sizes):  # pylint: disable=too-many-arguments,too-many-positional-arguments,unused-argument,redefined-outer-name
        """Build QKVParallelLinear layer with projection"""
        hidden_size = sizes['hidden_size']
        head_size = linear_spec.head_size if linear_spec.head_size is not None else hidden_size // 8
        total_num_heads = linear_spec.total_num_heads if linear_spec.total_num_heads is not None else 8
        total_num_kv_heads = linear_spec.total_num_kv_heads if linear_spec.total_num_kv_heads is not None else 8

        qkv_linear = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_size,
            total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
            config=config,
            compute_dtype=linear_spec.compute_dtype,
            bias=linear_spec.bias,
            gather_output=linear_spec.gather_output,
            transpose_b=linear_spec.transpose_b,
            tp_group=tp_group,
        )

        qkv_output_size = self._calculate_qkv_output_size(total_num_heads, total_num_kv_heads, head_size, tp_group)
        proj_linear = self._create_projection_layer(qkv_output_size, hidden_size, linear_spec, tp_group)

        return QKVWithProjection(qkv_linear, proj_linear), False

    def _calculate_merged_output_size(self, ffn_hidden_size, tp_group):
        tp_size_actual = tp_group.size if is_initialized() else 1
        return 2 * ffn_hidden_size // tp_size_actual

    def _calculate_merged_full_output_size(self, ffn_hidden_size):
        return 2 * ffn_hidden_size

    def _build_merged_layer(self, linear_spec, config, tp_group, prev_output_is_parallel, sizes):  # pylint: disable=invalid-name,unused-argument
        """Build MergedColumnParallelLinear layer with projection"""
        hidden_size = sizes['hidden_size']
        ffn_hidden_size = linear_spec.ffn_hidden_size if linear_spec.ffn_hidden_size is not None else hidden_size

        merged_linear = MergedColumnParallelLinear(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            config=config,
            compute_dtype=linear_spec.compute_dtype,
            bias=linear_spec.bias,
            gather_output=linear_spec.gather_output,
            transpose_b=linear_spec.transpose_b,
            tp_group=tp_group,
        )

        merged_full_output_size = self._calculate_merged_full_output_size(ffn_hidden_size)
        proj_linear = self._create_projection_layer(merged_full_output_size, hidden_size, linear_spec, tp_group)

        return MergedWithProjection(merged_linear, proj_linear), False

    def _init_weights(self, linear, layer_idx):
        """Initialize weights with fixed deterministic values"""
        if not hasattr(self, 'weight_counter'):
            self.weight_counter = 0

        def init_weight(weight, weight_name=""):  # pylint: disable=unused-argument
            weight_shape = weight.shape
            weight_dtype = weight.dtype

            seed = 42 + layer_idx * 1000 + self.weight_counter
            self.weight_counter += 1

            np.random.seed(seed)
            weight_data = np.random.uniform(-0.01, 0.01, size=weight_shape)

            return Tensor(weight_data, dtype=weight_dtype)

        if hasattr(linear, 'qkv') and hasattr(linear, 'proj'):
            linear.qkv.weight.set_data(init_weight(linear.qkv.weight, "qkv"))
            linear.proj.weight.set_data(init_weight(linear.proj.weight, "proj"))
        elif hasattr(linear, 'merged') and hasattr(linear, 'proj'):
            linear.merged.weight.set_data(init_weight(linear.merged.weight, "merged"))
            linear.proj.weight.set_data(init_weight(linear.proj.weight, "proj"))
        elif hasattr(linear, 'linear'):
            linear.linear.weight.set_data(init_weight(linear.linear.weight, "linear"))
        else:
            linear.weight.set_data(init_weight(linear.weight, "weight"))

    def construct(self, x, group_list=None):
        """Forward pass: all layers process same input in parallel, outputs are concatenated"""
        output_list = []
        for layer in self.layers:
            if isinstance(layer, GroupedLinearWrapper):
                output = layer(x, group_list=group_list)
            else:
                output = layer(x)
            output_list.append(output)

        concat_output = msops.concat(output_list, axis=-1)
        return concat_output

    def get_outputs_dict(self, x, group_list=None):
        """Get outputs as dictionary mapping layer indices to outputs"""
        outputs = {}
        for layer_idx, layer in enumerate(self.layers):
            if isinstance(layer, GroupedLinearWrapper):
                output = layer(x, group_list=group_list)
            else:
                output = layer(x)
            outputs[layer_idx] = output
        return outputs

    # pylint: disable=unused-argument
    def generate(self, input_ids, do_sample=False, max_new_tokens=1):
        """Generation interface for test framework compatibility"""
        if isinstance(input_ids, np.ndarray):
            input_ids = Tensor(input_ids)
        elif not isinstance(input_ids, Tensor):
            input_ids = Tensor(np.array(input_ids))

        if len(input_ids.shape) == 1:
            input_ids = msops.expand_dims(input_ids, 0)
        elif len(input_ids.shape) == 3:
            input_ids = msops.squeeze(input_ids, 1)

        x = input_ids.astype(msdtype.float32)
        return self.construct(x)


def create_mixed_precision_network(linear_specs, hidden_dim=1024, num_layers=7, num_experts=2,  # pylint: disable=redefined-outer-name
                                    tensor_model_parallel_size=1):
    """Factory function to create mixed precision network"""
    tp_group = get_tensor_model_parallel_group() if is_initialized() else default_pgs

    model_spec = ModelSpec(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_experts=num_experts,
        tensor_model_parallel_size=tensor_model_parallel_size,
        linear_specs=linear_specs
    )

    return MixedPrecisionNetwork(model_spec, tp_group)
