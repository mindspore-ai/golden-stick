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
"""ptq wrapper cells for mindformers"""
import mindspore as ms
from mindspore import Tensor
from mindspore import ops, mint
from mindspore.nn import Cell
from mindspore.ops.auto_generate import ReshapeAndCache, PagedAttention

from mindspore_gs.common import logger
from mindspore_gs.long_context_compress.razor_attention import RAConfig

from vllm.attention import Attention
from vllm.attention.backends.abstract import AttentionType, AttentionMetadata


DUMMY_INPUT_LENGTH = 2500
REPET_TIMES = 4

class RACompressCell(Cell):
    """razor attention compress kvcache"""
    def __init__(self, layer_name, layer, cfg, **kwargs):
        super().__init__()
        self.layer = layer
        self.layer_name = layer_name
        self.num_heads = layer.num_heads
        self.num_kv_heads = layer.num_kv_heads
        self.head_size = layer.head_size
        self.scale_factor = layer.scale
        self.kv_groups = self.num_heads // self.num_kv_heads

        self.attn_inputs_args = None
        self.attn_inputs_kwargs = None
        self._echo_score = []
        self._induction_score = []

    def _calc_attn_score(self):
        "_calc_attn_score"
        query, key, value = self.attn_inputs_args[0:3]
        
        seq_len = query.shape[-2]
        query = ops.transpose(query.reshape(-1, seq_len, self.num_heads, self.head_size), (0, 2, 1, 3))
        key = ops.transpose(key.reshape(-1, seq_len, self.num_kv_heads, self.head_size), (0, 2, 1, 3))
        value = ops.transpose(value.reshape(-1, seq_len, self.num_kv_heads, self.head_size), (0, 2, 1, 3))

        key = self._repeat_kv(key, self.kv_groups)
        value = self._repeat_kv(value, self.kv_groups)

        attn_score = self._attn_dot_product(query, key)
        logger.debug(f"RACompressCell: attn_score of Layer({self.layer_name}) is {{{attn_score.shape}, {attn_score.dtype}, "
                     f"{attn_score.asnumpy()}}}")
        return attn_score

    def _repeat_kv(self, x, rep):
        """ Expand key, value on num_head dimension. """
        if rep == 1:
            return x
        bs, num_groups, seq_length, head_dim = x.shape()
        # [B, ng, S, D] -> [B, ng, 1, S*D]
        x = x.reshape((bs, num_groups, 1, seq_length * head_dim))
        x = x.tile((1, 1, rep, 1))
        # [B, ng, rep, S*D] -> [B, N, S, D]
        x = x.reshape((bs, num_groups * rep, seq_length, head_dim))
        return x

    def _generate_attn_mask(self, bsz, q_len, kv_len, flatten=False):
        "_generate_attn_mask"
        if flatten:
            raise NotImplementedError("flatten not support yet")
        mask = mint.ones((bsz, 1, q_len, kv_len), dtype=ms.uint8)
        mask = mint.triu(mask, diagonal=1)
        return mask

    def _attn_dot_product(self, query, key):
        """_attn_dot_product"""
        # (B, N, S, D)
        bs, _, seq_len, _ = query.shape
        # score -> (B, N, S, S)
        score = ops.bmm(query, ops.transpose(key, (0, 1, 3, 2)))
        score = mint.mul(score, self.scale_factor)

        attn_mask = self._generate_attn_mask(bs, seq_len, seq_len)
        causal_mask = attn_mask * -1e-5
        masked_input = score + causal_mask
        probs = mint.nn.functional.softmax(masked_input, dim=-1)
        return probs

    def _get_score_mask(self, seq_len):
        "_get_score_mask"
        echo_mask = mint.zeros((seq_len, seq_len), dtype=ms.uint8)
        induction_mask = mint.zeros((seq_len, seq_len), dtype=ms.uint8)
        for i in range(1, seq_len):
            if i // DUMMY_INPUT_LENGTH == 0:
                continue
            for j in range(i % DUMMY_INPUT_LENGTH, i, DUMMY_INPUT_LENGTH):
                echo_mask[i, j] = 1
                induction_mask[i, j+1] = 1
        return echo_mask, induction_mask

    def _calc_score(self, head_i_score):
        "_calc_score"
        seq_len = head_i_score.shape[-2]
        echo_mask, induction_mask = self._get_score_mask(seq_len)
        echo_score = ops.mean(head_i_score * echo_mask)
        induction_score = ops.mean(head_i_score * induction_mask)
        return echo_score, induction_score

    def _get_score(self, attn_score):
        "_get_score"
        for i in range(self.num_heads):
            echo_score, induction_score = self._calc_score(attn_score[:, i, :, :].squeeze(0))
            self._echo_score.append(echo_score)
            self._induction_score.append(induction_score)
            logger.debug(f"RACompressCell: echo_score of Layer({self.layer_name}) head [{i}] is {echo_score.asnumpy()}")
            logger.debug(f"RACompressCell: induction_score of Layer({self.layer_name}) head [{i}] is {induction_score.asnumpy()}")

    def _max_every_group(self, data, n):
        "_max_every_group"
        max_values = [max(data[i:i+n]) for i in range(0, len(data), n)]
        return max_values

    def process(self):
        "process"
        attn_score = self._calc_attn_score()
        self._get_score(attn_score)
        self._echo_score = self._max_every_group(self._echo_score,
                                                 self.kv_groups)
        self._induction_score = self._max_every_group(self._induction_score,
                                                      self.kv_groups)
        logger.debug(f"RACompressCell: echo_score of Layer({self.layer_name}) is {self._echo_score}")
        logger.debug(f"RACompressCell: induction_score of Layer({self.layer_name}) is {self._induction_score}")

    def construct(self, *args, **kwargs):
        # 获取attn的输入，用于计算attn_score
        if not self.attn_inputs_args:
            self.attn_inputs_args = args
            self.attn_inputs_kwargs = kwargs
        return self.layer(*args, **kwargs)


class DeployRACompressCell(Cell):
    """DeployAttentionMgrCell"""
    def __init__(self, layer_name, layer: Attention, cfg: RAConfig, retrieval_head: list):
        super().__init__()
        self.layer = layer
        self.cfg = cfg
        self.num_kv_group = self.layer.num_heads // self.layer.num_kv_heads

        self.kv_retri_heads = None
        self.kv_non_retri_heads = None
        self.q_retri_heads = None
        self.q_non_retri_heads = None
        self._load_retrieval_head(retrieval_head)

        self.paged_attention_retri = PagedAttention(len(self.q_retri_heads), self.layer.scale,
                                                    len(self.kv_retri_heads))
        self.paged_attention_non_retri = PagedAttention(len(self.q_non_retri_heads), self.layer.scale,
                                                        len(self.kv_non_retri_heads))

        self.reshape_and_cache_retri = ReshapeAndCache()
        self.reshape_and_cache_non_retri = ReshapeAndCache()

        self.apply_decoder_num = 0

    def _load_retrieval_head(self, retrieval_head):
        self.kv_retri_heads = retrieval_head
        self.kv_non_retri_heads = [i for i in range(self.layer.num_kv_heads) if i not in self.kv_retri_heads]
        self.q_retri_heads = [i for h in self.kv_retri_heads for i in range(h * self.num_kv_group, 
                                                                            (h+1) * self.num_kv_group)]
        self.q_non_retri_heads = [i for h in self.kv_non_retri_heads for i in range(h * self.num_kv_group,
                                                                                    (h+1) * self.num_kv_group)]

    def _compress(self, cache, use_virtual_token=False):
        cache_len = cache.shape[0]
        if cache_len <= self.cfg.local_capacity:
            return cache
        if not use_virtual_token:
            drop_cache_len = cache_len - self.cfg.local_capacity
            left = cache[:self.cfg.sink_size,]
            right = cache[drop_cache_len + self.cfg.sink_size:,]
            compress_cache = ops.cat([left, right], axis=0)
        else:
            drop_cache_len = cache_len - self.cfg.local_capacity + 1
            left = cache[:self.cfg.sink_size,]
            middle = cache[self.cfg.sink_size:drop_cache_len+self.cfg.sink_size,]
            middle = ops.mean(middle, axis=0, keep_dims=True)
            right = cache[drop_cache_len + self.cfg.sink_size:,]
            compress_cache = ops.cat([left, middle, right], axis=0)
        return compress_cache

    def _is_empty_tensor(self, tensor):
        return any(dim == 0 for dim in tensor.shape)

    def construct(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        kv_cache: Tensor,
        attn_metadata: AttentionMetadata,
        attn_type: str = AttentionType.DECODER,
    ):
        query = query.reshape((-1, self.layer.num_heads, self.layer.head_size))
        key = key.reshape((-1, self.layer.num_kv_heads, self.layer.head_size))
        value = value.reshape((-1, self.layer.num_kv_heads, self.layer.head_size))

        if key is not None:
            if value is None:
                raise ValueError("value should not be None when key is not None.")
        else:
            if value is not None:
                raise ValueError("value should be None when key is None.")

        if attn_type != AttentionType.ENCODER and (
            kv_cache is not None and kv_cache[0].numel() > 0
        ):
            # KV-cache during decoder-self- or
            # encoder-decoder-cross-attention, but not
            # during encoder attention.
            #
            # Even if there are no new key/value pairs to cache,
            # we still need to break out key_cache and value_cache
            # i.e. for later use by paged attention
            key_cache_retri, value_cache_retri, key_cache_non_retri, value_cache_non_retri = \
                self.split_kv_cache(kv_cache, self.layer.num_kv_heads, 
                                          self.layer.head_size)

            if (key is not None) and (value is not None):
                if attn_type == AttentionType.ENCODER_DECODER:
                    # Update cross-attention KV cache (prefill-only)
                    # During cross-attention decode, key & value will be None,
                    # preventing this IF-statement branch from running
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                else:
                    # Update self-attention KV cache (prefill/decode)
                    updated_slot_mapping = attn_metadata.slot_mapping

                key_retri = key[:, self.kv_retri_heads, :]
                value_retri = value[:, self.kv_retri_heads, :]

                key_non_retri = key[:, self.kv_non_retri_heads, :]
                value_non_retri = value[:, self.kv_non_retri_heads, :]

                if self.kv_retri_heads:
                    self.reshape_and_cache_retri(
                        key_retri.view(1, key_retri.shape[0], -1),
                        value_retri.view(1, value_retri.shape[0], -1),
                        key_cache_retri,
                        value_cache_retri,
                        updated_slot_mapping,
                    )

                if self.kv_non_retri_heads:
                    if prefill_meta := attn_metadata.prefill_metadata:
                        updated_slot_mapping = updated_slot_mapping[:self.cfg.local_capacity]
                        key_non_retri = self._compress(key_non_retri, self.cfg.use_virtual_token)
                        value_non_retri = self._compress(value_non_retri, self.cfg.use_virtual_token)
                    else:
                        updated_slot_mapping[0] = self.apply_decoder_num + self.cfg.local_capacity
                    self.reshape_and_cache_non_retri(
                        key_non_retri.view(1, key_non_retri.shape[0], -1),
                        value_non_retri.view(1, value_non_retri.shape[0], -1),
                        key_cache_non_retri,
                        value_cache_non_retri,
                        updated_slot_mapping,
                    )

        if prefill_meta := attn_metadata.prefill_metadata:
            if attn_metadata.seq_lens is None:
                raise ValueError("attn_metadata.seq_lens should not be None")
            if not prefill_meta.prefill_metadata.chunked_prefill:  # type: ignore
                query = query.reshape((query.shape[0], -1))
                key = key.reshape((key.shape[0], -1))
                value = value.reshape((value.shape[0], -1))
                output = self.layer._run_prefill_forward(
                    query, key, value, prefill_meta, attn_type=attn_type
                )
            else:
                # TODO: to support CPP
                raise NotImplementedError("not support CPP yet")

        if decode_meta := attn_metadata.decode_metadata:
            if attn_type == AttentionType.ENCODER_ONLY:
                raise ValueError("Encoder-only models should not have decode metadata.")
            # Decoding run.
            self.apply_decoder_num += 1
            (
                seq_lens_arg,
                max_seq_len_arg,
                block_tables_arg,
            ) = decode_meta.get_seq_len_block_table_args(attn_type)

            alibi_mask = None
            output = self._run_decode_attention(
                query,
                key_cache_retri,
                value_cache_retri,
                key_cache_non_retri,
                value_cache_non_retri,
                seq_lens_arg,
                block_tables_arg,
                alibi_mask,
            )
        return output

    def _run_decode_attention(
        self, query, key_cache_retri, value_cache_retri, 
        key_cache_non_retri, value_cache_non_retri, 
        seq_lens_arg, block_tables, alibi_mask=None
    ):
        # TODO: to support alibi mask
        # if self.use_alibi_mask:
        #     return self.paged_attention(query, key_cache, value_cache, batch_valid_length, block_tables, alibi_mask)

        bsz = block_tables.shape[0]
        seq_len = query.shape[0]

        output_retrieval = ops.zeros((0, 0))
        output_non_retrival = ops.zeros((0, 0))

        query_retri = query[:, self.q_retri_heads, :]
        query_non_retri = query[:, self.q_non_retri_heads, :]

        query_retri = query_retri.reshape((bsz, seq_len, -1))
        query_non_retri = query_non_retri.reshape((bsz, seq_len, -1))

        if self.q_retri_heads:
            output_retrieval = self.paged_attention_retri(
                query_retri, key_cache_retri, value_cache_retri, block_tables, context_lens=seq_lens_arg
            )
        if self.q_non_retri_heads:
            output_non_retrival = self.paged_attention_non_retri(
                query_non_retri, key_cache_non_retri, value_cache_non_retri, block_tables, context_lens=seq_lens_arg
            )
        if self._is_empty_tensor(output_retrieval) or self._is_empty_tensor(output_non_retrival):
            output = output_retrieval if not self._is_empty_tensor(output_retrieval) else output_non_retrival
        else:
            output = ops.zeros((bsz, seq_len, self.layer.num_heads, self.layer.head_size), dtype=query_retri.dtype)
            output_retrieval = output_retrieval.reshape((bsz, seq_len, len(self.q_retri_heads), -1))
            output_non_retrival = output_non_retrival.reshape((bsz, seq_len, len(self.q_non_retri_heads), -1))

            output[:, :, self.q_retri_heads, :] = output_retrieval
            output[:, :, self.q_non_retri_heads, :] = output_non_retrival
        return output.reshape((seq_len, -1))

    def split_kv_cache(
        self,
        kv_cache,
        num_kv_heads: int,
        head_size: int,
    ):
        # TODO: to support view operation on mindspore.
        # num_blocks = kv_cache.shape[1]

        key_cache_retri = kv_cache[0]
        value_cache_retri = kv_cache[1]
        key_cache_non_retri = kv_cache[2]
        value_cache_non_retri = kv_cache[3]
        return key_cache_retri, value_cache_retri, key_cache_non_retri, value_cache_non_retri
