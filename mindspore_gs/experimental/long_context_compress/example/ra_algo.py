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
"""razor attention for network."""
import argparse

from mindspore_gs.long_context_compress.razor_attention import RAMode, RAConfig
from mindspore_gs.common import BackendTarget
from mindspore_gs.long_context_compress.razor_attention import RazorAttention as RA


def get_args():
    """get_args"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str, required=True)
    return parser.parse_args()


class LLMNetworkHelper:
    """LLMNetworkHelper"""

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.llm = None
        self._load_vllm_ms_plugin()
        self._create_llm()

    # pylint: disable=W0611
    @staticmethod
    def _load_vllm_ms_plugin():
        """_load_vllm_ms_plugin"""
        import vllm_mindspore

    def _create_llm(self):
        """_create_llm"""
        from vllm import LLM
        self.llm = LLM(model=self.model_path, max_model_len=31500)

    def get_network(self):
        """get_network"""
        return self.llm.llm_engine.model_executor.driver_worker.model_runner.model

    def generate(self, *args, **kwargs):
        """generate"""
        from vllm import SamplingParams
        sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=1)
        outputs = self.llm.generate(sampling_params=sampling_params, *args, **kwargs)
        return outputs

if __name__ == "__main__":
    argus = get_args()
    helper = LLMNetworkHelper(argus.model_path)
    ra_config = RAConfig(mode=RAMode.SEARCH_RETRIEVAL,
                         backend=BackendTarget.ASCEND,
                         retrieval_head_path='./head_dict.json',
                         echo_head_ratio=0.01,
                         induction_head_ratio=0.14,
                         sink_size=4,
                         local_capacity=256,
                         use_virtual_token=True)
    ra = RA(ra_config)
    ra.apply(helper)
