# Copyright 2023 Huawei Technologies Co., Ltd
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
"""
concrete gate and pruning heads for the model
"""
import math
from itertools import compress
import mindspore as ms
from mindspore import ops
from mindspore import nn
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
from mindspore_gs.validator import Validator


class ConcreteGate(nn.Cell):
    """
    A gate made of stretched concrete distribution (using experimental Stretchable Concrete™)
    Can be applied to sparsify neural network activations or weights.
    Example usage: https://gist.github.com/justheuristic/1118a14a798b2b6d47789f7e6f511abd
    :param shape: shape of gate variable. can be broadcasted.
        e.g. if you want to apply gate to tensor [batch, length, units] over units axis,
        your shape should be [1, 1, units]
    :param temperature: concrete sigmoid temperature, should be in (0, 1] range
        lower values yield better approximation to actual discrete gate but train longer
    :param stretch_limits: min and max value of gate before it is clipped to [0, 1]
        min value should be negative in order to compute l0 penalty as in (https://arxiv.org/pdf/1712.01312.pdf)
        however, you can also use tf.nn.sigmoid(log_a) as regularizer if min, max = 0, 1
    :param l0_penalty: coefficient on the regularizer that minimizes l0 norm of gated value
    :param eps: a small additive value used to avoid NaNs
    """

    def __init__(self, shape, temperature=0.33, stretch_limits=(-0.1, 1.1),
                 l0_penalty=1.0, eps=1e-6, gqa_rep=1):

        super(ConcreteGate, self).__init__()
        Validator.check_value_type("shape", shape, [list], self.__class__.__name__)
        Validator.check_value_type("temperature", temperature, [float], self.__class__.__name__)
        Validator.check_value_type("stretch_limits", stretch_limits, [tuple], self.__class__.__name__)
        Validator.check_value_type("l0_penalty", l0_penalty, [float], self.__class__.__name__)
        Validator.check_value_type("eps", eps, [float], self.__class__.__name__)
        Validator.check_equal_int(len(stretch_limits), 2, "stretch_limits size", self.__class__.__name__)
        self.temperature, self.stretch_limits, self.eps = temperature, stretch_limits, eps
        self.l0_penalty = l0_penalty
        self.log_a = Parameter(initializer('xavier_uniform', shape), name="log_a")
        self.sigmoid = ops.Sigmoid()
        self.log = ops.Log()
        self.op = ops.ReduceSum()
        self.uniformreal = ops.UniformReal()
        self.shape = self.log_a.shape
        low, high = self.stretch_limits
        assert low < 0.0, "p_gate_closed can be computed only if lower stretch limit is negative"
        self.bias = Tensor(temperature * math.log(-low / high))
        self.gqa_rep = gqa_rep
        if gqa_rep > 1:
            self.tile_kv = P.Tile()  # Needed only for Grouped Query Attention
            self.reshape = P.Reshape()

    def _repeat_for_gqa(self, gates):
        bs, n_kv_head, seqlen, head_dim = gates.shape
        gates = self.reshape(gates, (bs * n_kv_head, 1, seqlen, head_dim))
        gates = self.tile_kv(gates, (1, self.gqa_rep, 1, 1))
        gates = self.reshape(gates, (bs, n_kv_head * self.gqa_rep, seqlen, head_dim))
        return gates

    def construct(self, values, is_train=True):
        """ applies gate to values, if is_train, adds regularizer to reg_collection """
        Validator.check_value_type("is_train", is_train, [bool], self.__class__.__name__)
        is_train = self.training if is_train is None else is_train
        gates = self.get_gates(is_train)
        if not self.lrp_train:
            gates = self.zero_gates(gates)
        if self.gqa_rep > 1:
            gates = self._repeat_for_gqa(gates)
        return values * gates

    def zero_gates(self, gates):
        return gates*(gates >= 0.5)


    def get_gates(self, is_train):
        """ samples gate activations in [0, 1] interval """
        Validator.check_value_type("is_train", is_train, [bool], self.__class__.__name__)
        concrete = self.sigmoid(self.log_a)

        if is_train:
            noise = Tensor((1.0 - 2 * self.eps) * self.uniformreal(self.shape)) + self.eps
            concrete = self.sigmoid((self.log(noise) - self.log(1.0 - noise) + self.log_a) / self.temperature)
        else:
            concrete = self.sigmoid(self.log_a)

        low, high = self.stretch_limits
        stretched_concrete = concrete * (high - low) + low
        clipped_concrete = ops.clip_by_value(stretched_concrete, 0, 1)
        return clipped_concrete

    def get_penalty(self):
        """
        Computes l0 and l2 penalties. For l2 penalty one must also provide the sparsified values
        (usually activations or weights) before they are multiplied by the gate
        Returns the regularizer value that should to be MINIMIZED (negative logprior)
        """
        # compute p(gate_is_closed) = cdf(stretched_sigmoid < 0)
        p_open = self.sigmoid(self.log_a - self.bias)
        p_open = ops.clip_by_value(p_open, self.eps, 1.0 - self.eps)

        total_reg = self.l0_penalty * self.op(p_open)
        return total_reg


def find_pruneable_heads_and_indices(heads, n_heads: int, head_size: int, already_pruned_heads):
    """
    Finds the heads and their indices taking :obj:`already_pruned_heads` into account.

    Args:
        heads (:obj:`List[int]`): List of the indices of heads to prune.
        n_heads (:obj:`int`): The number of heads in the model.
        head_size (:obj:`int`): The size of each head.
        already_pruned_heads (:obj:`Set[int]`): A set of already pruned heads.

    Returns:
        :obj:`Tuple[Set[int], LongTensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    Validator.check_value_type("heads", heads, [list, tuple], "find_pruneable_heads_and_indices")
    Validator.check_value_type("already_pruned_heads", already_pruned_heads, [set])
    ones = ops.Ones()
    mask = ones((n_heads, head_size), ms.float16)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0

    equal = ops.Equal()
    mask = equal(mask.view(-1), 1.0)
    np_tensor = ms.numpy.arange(len(mask)).numpy()
    index = list(compress(np_tensor, mask))
    index = ms.Tensor(index).long()
    return heads, index


def prune_linear_layer(layer, index, dim: int = 0):
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (:obj:`nn.Linear`): The layer to prune.
        index (:obj:`LongTensor`): The indices to keep in the layer.
        dim (:obj:`int`, `optional`, defaults to 0): The dimension on which to keep the indices.

    Returns:
        :obj:`nn.Linear`: The pruned layer as a new layer with :obj:`requires_grad=True`.
    """
    Validator.check_value_type("index", index, [Tensor], "prune_linear_layer")
    Validator.check_non_negative_int(dim, "dim")
    Validator.check_value_type("layer", layer, [nn.Cell], "prune_linear_layer")
    if dim == 1:
        w = layer.weight.T[index].T
    else:
        w = layer.weight[index]
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias
        else:
            b = layer.bias[index]
    new_size = list(layer.weight.shape)
    new_size[dim] = len(index)
    new_layer = nn.Dense(new_size[1], new_size[0], bias_init=layer.bias is not None).to_float(ms.float16)
    new_layer.weight.requires_grad = False
    new_layer.weight = w.copy()
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias = b.copy()
        new_layer.bias.requires_grad = True

    return new_layer


def prune_heads_for_bert_model(self, heads_to_prune):
    """
        Prune heads for bert
        @heads: heads to prune for model
        """
    for layer, heads in heads_to_prune.items():
        self.bert_encoder.layers[layer].attention.prune_heads(heads)


def prune_heads_for_bert_self_attention(self, heads):
    """
    Prune heads for bert
    @heads: heads to prune for self attention
    """
    if not heads:
        return

    self.pruned_heads = set()

    heads, index = find_pruneable_heads_and_indices(
        heads, self.attention.num_attention_heads, self.attention.size_per_head, self.pruned_heads
    )

    # Prune linear layers
    self.attention.query_layer = prune_linear_layer(self.attention.query_layer, index)
    self.attention.key_layer = prune_linear_layer(self.attention.key_layer, index)
    self.attention.value_layer = prune_linear_layer(self.attention.value_layer, index)
    self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
    self.attention.shape_return = (-1, index.size)

    # Update hyper params and store pruned heads
    self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
    self.attention.all_head_size = self.attention.size_per_head * self.attention.num_attention_heads
    self.pruned_heads = self.pruned_heads.union(heads)


def prune_heads_for_gpt_model(self, heads_to_prune):
    for layer, heads in heads_to_prune.items():
        self.encoder.blocks[layer].attention.prune_heads(heads)


def prune_heads_for_multi_head_attention(self, heads):
    """
    prune_heads_for_multi_head_attention
    """
    if not heads:
        return
    Validator.check_value_type("heads", heads, [list, tuple], self.__class__.__name__)

    self.pruned_heads = set()

    heads, index = find_pruneable_heads_and_indices(
        heads, self.n_head, self.size_per_head, self.pruned_heads
    )

    self.dense1 = prune_linear_layer(self.dense1, index)
    self.dense2 = prune_linear_layer(self.dense2, index)
    self.dense3 = prune_linear_layer(self.dense3, index)
    self.projection = prune_linear_layer(self.projection, index, dim=1)

    # Update hyper params and store pruned heads
    self.n_head = self.n_head - len(heads)
    self.pruned_heads = self.pruned_heads.union(heads)
