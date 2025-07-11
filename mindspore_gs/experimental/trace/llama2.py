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
"""fake llama2 network for st."""

import numpy as np
from mindspore import context
from mindspore import nn, Tensor, Parameter
from mindspore.common.initializer import initializer
from mindspore.ops import operations as ops
from mindspore.ops import functional as func
from mindspore import dtype as msdtype
from mindspore_gs.common import logger


class Linear(nn.Cell):
    """Linear"""
    def __init__(self, ic, oc):
        super().__init__()
        self.ic = ic
        self.oc = oc
        self.weight = Parameter(initializer('ones', (ic, oc), dtype=msdtype.float16))
        self.bias = Parameter(initializer('ones', (oc,), dtype=msdtype.float16))
        self.mm = ops.MatMul()
        self.bias_add = ops.Add()

    def construct(self, x):
        """construct"""
        out_shape = func.shape(x)[:-1] + (self.oc,)
        x = func.reshape(x, (-1, self.ic))
        x = self.mm(x, self.weight)
        x = self.bias_add(x, self.bias)
        return func.reshape(x, out_shape)


class Attention(nn.Cell):
    """Attention"""
    def __init__(self, num_heads, hidden_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        kv_dim = self.head_dim * num_heads
        self.proj_q = Linear(hidden_size, hidden_size)
        self.proj_k = Linear(hidden_size, kv_dim)
        self.proj_v = Linear(hidden_size, kv_dim)
        self.proj_o = Linear(hidden_size, hidden_size)
        self.mm_qk = ops.BatchMatMul(transpose_b=True)
        self.mm_v = ops.BatchMatMul()

    def construct(self, x):
        """construct"""
        bs, seq, _ = func.shape(x)
        q = self.proj_q(x)
        q = func.reshape(q, (bs, seq, self.num_heads, self.head_dim))
        q = func.transpose(q, (0, 2, 1, 3))
        k = self.proj_k(x)
        k = func.reshape(k, (bs, seq, self.num_heads, self.head_dim))
        k = func.transpose(k, (0, 2, 1, 3))
        v = self.proj_v(x)
        v = func.reshape(v, (bs, seq, self.num_heads, self.head_dim))
        v = func.transpose(v, (0, 2, 1, 3))
        score = self.mm_qk(q, k)
        probs = func.softmax(score)
        weight_values = self.mm_v(probs, v)
        weight = func.transpose(weight_values, (0, 2, 1, 3))
        context_layer = func.reshape(weight, (bs, seq, -1))
        return self.proj_o(context_layer)


class RMSNorm(nn.Cell):
    """RMSNorm"""
    def __init__(self, dim):
        super().__init__()
        self.weight = Parameter(initializer('ones', (dim,), dtype=msdtype.float16))
        self.mul = ops.Mul()
        self.square = ops.Square()
        self.mean = ops.ReduceMean(keep_dims=True)
        self.add = ops.Add()
        self.rsqrt = ops.Rsqrt()

    def construct(self, x):
        """construct"""
        norm_factor = self.square(x)
        norm_factor = self.mean(norm_factor, -1)
        norm_factor = self.add(norm_factor, 1e-6)
        norm_factor = self.rsqrt(norm_factor)
        x = self.mul(x, norm_factor)
        output = self.mul(x, self.weight)
        return output


class FeedForward(nn.Cell):
    """FeedForward"""
    def __init__(self, dim, hidden_dim, multi_of=256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multi_of * ((hidden_dim + multi_of - 1) // multi_of)
        self.proj_gate = Linear(dim, hidden_dim)
        self.proj_hidden = Linear(dim, hidden_dim)
        self.proj_w2 = Linear(hidden_dim, dim)

    def construct(self, x):
        """construct"""
        gate = self.proj_gate(x)
        hidden = self.proj_hidden(x)
        hidden = gate * hidden
        return self.proj_w2(hidden)


class DecoderLayer(nn.Cell):
    """DecoderLayer"""
    def __init__(self, num_heads, hidden_size, multi_of=256):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size)
        self.ffn_norm = RMSNorm(hidden_size)
        self.attention = Attention(num_heads=num_heads, hidden_size=hidden_size)
        self.ffn = FeedForward(hidden_size, 4 * hidden_size, multi_of)

    def construct(self, x):
        """construct"""
        x = self.attn_norm(x)
        h = self.attention(x)
        h = self.ffn_norm(h)
        ffn = self.ffn(h)
        return h + ffn


class Embedding(nn.Cell):
    """Embedding"""
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embed_weight = Parameter(initializer('ones', (vocab_size, embed_size), dtype=msdtype.float16))
        self.gather = ops.Gather()

    def construct(self, input_ids):
        """construct"""
        return self.gather(self.embed_weight, input_ids, 0)


class LlamaModel(nn.Cell):
    """LlamaModel"""
    def __init__(self, vocab_size, num_layers, hidden_size, num_heads, multi_of=256, dtype=msdtype.float16):
        super().__init__()
        self.layers = nn.CellList()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dtype = dtype
        self.tok_embeddings = Embedding(vocab_size, hidden_size)
        for _ in range(num_layers):
            self.layers.append(DecoderLayer(num_heads=num_heads, hidden_size=hidden_size, multi_of=multi_of))
        self.norm_out = RMSNorm(hidden_size)

    def construct(self, tokens: Tensor):
        """construct"""
        h = func.cast(self.tok_embeddings(tokens), self.dtype)
        h = func.reshape(h, (1, 1024, self.hidden_size))
        for i in range(self.num_layers):
            h = self.layers[i](h)
        return self.norm_out(h)


def create_llama(num_layers=1):
    """create_llama"""
    hidden_size = 1024
    num_heads = 32
    vocab_size = 32000
    multi_of = 256
    return LlamaModel(vocab_size, num_layers, hidden_size, num_heads, multi_of)


def create_input(batch_size=1, seq_len=1024):
    """create_input"""
    return Tensor(np.ones((batch_size, seq_len), dtype=np.int32), dtype=msdtype.int32)


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE)
    bs, seq = 1, 1024
    network = create_llama()
    output = network(create_input()).asnumpy()
    logger.debug(f'output shape: {output.shape},'
                 f'output dtype: {output.dtype},'
                 f'output value: {output}')
