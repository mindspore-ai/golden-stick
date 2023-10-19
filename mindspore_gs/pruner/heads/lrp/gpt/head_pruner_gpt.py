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
"""LRP Head Pruner For GPT"""
from mindspore import load_param_into_net
from mindspore import nn
from mindspore.parallel._transformer.transformer import MultiHeadAttention

from official.nlp.GPT.src.gpt import GPT_Model

from mindspore_gs.pruner.heads.lrp.gpt.archs import GPTModel as GPTModelGated
from mindspore_gs.pruner.heads.lrp.utils import prune_heads_for_gpt_model, prune_heads_for_multi_head_attention
from mindspore_gs.pruner.heads.lrp.abstract_lrp import AbstractHeadPrunerLRP


class ConfigNet(nn.Cell):
    """
    tmp pretrain for checkpoints repackage.
    """
    def __init__(self, model):
        super(ConfigNet, self).__init__()
        self.backbone = model


class HeadPrunerGPTLRP(AbstractHeadPrunerLRP):
    """Head Pruner class"""

    def _init_head(self, model):
        """
        check if model has a head, save the GPT model.
        """

        if isinstance(model, GPT_Model):
            self.has_head = False
            self.origin_model = model
            self.head = ConfigNet(self.origin_model)
        else:
            self.has_head = True
            self.head = model
            self.origin_model = model.backbone

    def _decorate_model(self, l0_penalty=0.0015):
        """
        decorate model, repackage the model with additional functionality.
        Args:
            l0_penalty: penalty value for gate calculation.

        Returns: gated GPT model.

        """
        gpt_gated = GPTModelGated(**self.model_config)
        load_param_into_net(gpt_gated, self.head.parameters_dict())
        gpt_gated.apply_gates(l0_penalty)

        if self.has_head:
            self.head.backbone = gpt_gated
            return self.head

        return gpt_gated

    def _prune_model(self, model, save_dir_path=None):
        """
        Prune the model, after training/fine-tuning.
        Args:
            model: that has been decorated.
            save_dir_path (optional): path to save the models and heads dictionary

        Returns: pruned & clean model.

        """
        if isinstance(model, GPTModelGated):
            model = ConfigNet(model)

        gates = model.backbone.get_gate_values()
        gates_dict = self._mask2dict(gates)

        if save_dir_path:
            self._save_model(model, gates_dict, save_dir_path)

        GPT_Model.prune_heads = prune_heads_for_gpt_model
        MultiHeadAttention.prune_heads = prune_heads_for_multi_head_attention

        load_param_into_net(self.origin_model, model.parameters_dict())

        self.origin_model.prune_heads(gates_dict)

        del GPT_Model.prune_heads
        del MultiHeadAttention.prune_heads

        if self.has_head:
            self.head.backbone = self.origin_model
            return self.head

        return self.origin_model
