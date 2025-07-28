# Copyright 2023-2024 Huawei Technologies Co., Ltd
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
"""ptq quant cells."""

import abc
from mindspore_gs.quantization.quant_cell import QuantCell


class PTQCell(QuantCell):
    """Wrapper Cell to PTQCell with FakeQuantizer"""

    @abc.abstractmethod
    def weight_quantizer(self):
        raise NotImplementedError

    @abc.abstractmethod
    def core_construct(self, *args):
        raise NotImplementedError

    @staticmethod
    def antiquant_strategy(weight_strategy=None):
        """antiquant strategy for w8a16"""
        if weight_strategy is None:
            return None
        strategy_len = len(weight_strategy)
        if strategy_len != 2:
            raise RuntimeError(f'strategy length shall be 2, but got {strategy_len}')
        x_strategy = weight_strategy

        anti_strategy = (x_strategy, (), ())
        return anti_strategy

    @staticmethod
    def qbmm_strategy(act_strategy, weight_strategy, is_transpose=False):
        """parallel strategy for antiquant bmm"""
        if act_strategy is None or weight_strategy is None:
            return None
        if is_transpose:
            scale_strategy = (weight_strategy[0],)
        else:
            scale_strategy = (weight_strategy[1],)
        return act_strategy, weight_strategy, scale_strategy


    @staticmethod
    def wqbmm_strategy(act_strategy, weight_strategy, is_transpose=False):
        """parallel strategy for antiquant bmm"""
        if act_strategy is None or weight_strategy is None:
            return None
        if is_transpose:
            scale_strategy = (weight_strategy[0],)
        else:
            scale_strategy = (weight_strategy[1],)
        offset_strategy = scale_strategy
        return act_strategy, weight_strategy, scale_strategy, offset_strategy

    @staticmethod
    def dynamic_bmm_strategy(act_strategy, weight_strategy, is_transpose=False):
        '''dynamic_bmm_strategy'''
        if act_strategy is None or weight_strategy is None:
            return None
        if is_transpose:
            scale_strategy = (weight_strategy[0],)
        else:
            scale_strategy = (weight_strategy[1],)
        pertoken_strategy = (act_strategy[0],)
        return act_strategy, weight_strategy, scale_strategy, pertoken_strategy
