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
"""BoolQ dataset."""


import copy
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from mindspore import dtype, Tensor
import mindspore.dataset.transforms as C
from mindspore.dataset import GeneratorDataset
from mindspore_gs.common import logger


TASK2DESC = {
    "computer_network": "计算机网络",
    "operating_system": "操作系统",
    "computer_architecture": "计算机组成",
    "college_programming": "大学编程",
    "college_physics": "大学物理",
    "college_chemistry": "大学化学",
    "advanced_mathematics": "高等数学",
    "probability_and_statistics": "概率论与数理统计",
    "discrete_mathematics": "离散数学",
    "electrical_engineer": "注册电气工程师",
    "metrology_engineer": "注册计量师",
    "high_school_mathematics": "高中数学",
    "high_school_physics": "高中物理",
    "high_school_chemistry": "高中化学",
    "high_school_biology": "高中生物",
    "middle_school_mathematics": "初中数学",
    "middle_school_biology": "初中生物",
    "middle_school_physics": "初中物理",
    "middle_school_chemistry": "初中化学",
    "veterinary_medicine": "兽医学",
    "college_economics": "大学经济学",
    "business_administration": "工商管理",
    "marxism": "马克思主义基本原理",
    "mao_zedong_thought": "毛泽东思想和中国特色社会主义理论概论",
    "education_science": "教育学",
    "teacher_qualification": "教师资格",
    "high_school_politics": "高中政治",
    "high_school_geography": "高中地理",
    "middle_school_politics": "初中政治",
    "middle_school_geography": "初中地理",
    "modern_chinese_history": "近代史纲要",
    "ideological_and_moral_cultivation": "思想道德修养与法律基础",
    "logic": "逻辑学",
    "law": "法学",
    "chinese_language_and_literature": "中国语言文学",
    "art_studies": "艺术学",
    "professional_tour_guide": "导游资格",
    "legal_professional": "法律职业资格",
    "high_school_chinese": "高中语文",
    "high_school_history": "高中历史",
    "middle_school_history": "初中历史",
    "civil_servant": "公务员",
    "sports_science": "体育学",
    "plant_protection": "植物保护",
    "basic_medicine": "基础医学",
    "clinical_medicine": "临床医学",
    "urban_and_rural_planner": "注册城乡规划师",
    "accountant": "注册会计师",
    "fire_engineer": "注册消防工程师",
    "environmental_impact_assessment_engineer": "环境影响评价工程师",
    "tax_accountant": "税务师",
    "physician": "医师资格",
}

choices = ["A", "B", "C", "D"]


def format_example(subject, line, include_answer=True):
    """format_example"""
    example = f"以下是中国关于{TASK2DESC.get(subject, '')}考试的单项选择题，请不要分析过程，\
        直接在A、B、C、D四个选项中选出正确答案。\n\n"
    example = example + line["question"]
    for choice in choices:
        example += f'\n{choice}. {line[f"{choice}"]}'

    if include_answer:
        example += "\n答案：" + line["answer"] + "\n\n"
    else:
        example += "\n答案："
    return example


class CEvalDataset(GeneratorDataset):
    """boolQ dataset."""
    def __init__(self, path: str, mode: str, seq_length: int, tokenizer: callable, ignore_token_id=-100,
                 need_pad=True, n_samples=-1, add_special_tokens=True):
        self.path = os.path.join(path)
        if mode not in ("eval", "train", "test"):
            raise ValueError("Input `mode` should be 'eval', 'test' or 'train', got: ", mode)
        self.mode = mode
        self.seq_len = seq_length
        self.ignore_token_id = ignore_token_id
        self.add_special_tokens = add_special_tokens
        self.tokenizer = tokenizer
        self.need_pad = need_pad
        if mode in ("eval", "test"):
            if hasattr(self.tokenizer, 'add_bos_token'):
                self.tokenizer.add_bos_token = True
            if hasattr(self.tokenizer, 'add_eos_token'):
                self.tokenizer.add_eos_token = False
        else:
            if hasattr(tokenizer, 'add_bos_token'):
                tokenizer.add_bos_token = True
            if hasattr(tokenizer, 'add_eos_token'):
                tokenizer.add_eos_token = True
        self.subjects = []
        self.input_ids = []
        self.labels = []
        self._load(n_samples)
        self.iter_subjects = None
        self.iter_input_ids = None
        self.iter_labels = None
        super().__init__(source=self, column_names=["subjects", "input_ids", "labels"])

    def __len__(self):
        return len(self.input_ids)

    def _load(self, n_samples=-1):
        """Load and preprocess squad dataset."""
        subjects = []
        sources = []
        targets = []

        for subject_name in tqdm(TASK2DESC.keys()):
            if self.path.endswith("/"):
                self.path = self.path[:-1]
            file_path = os.path.join(self.path, f"{subject_name}_{self.path.split('/')[-1]}.csv")
            df = pd.read_csv(file_path)

            for idx, row in df.iterrows():
                input_str = format_example(subject_name, row, include_answer=False)
                subjects.append(TASK2DESC.get(subject_name, ""))
                sources.append(input_str)
                targets.append(row["answer"])
                if (0 < int(n_samples / len(TASK2DESC.keys())) < idx) or (0 < n_samples <= len(sources)):
                    break
            if 0 < n_samples <= len(sources):
                break

        total_items = 0
        total_items = self._dataset_based_on_mode(subjects, sources, targets, total_items)
        logger.info("Find %d total data items", total_items)

    def _dataset_based_on_mode(self, subjects, sources, targets, total_items):
        """create dataset based on mode"""
        self.subjects.clear()
        self.input_ids.clear()
        self.labels.clear()
        pad_mode = 'constant'
        if self.mode == "eval":
            for subject, prompt, answer in zip(subjects, sources, targets):
                total_items += 1
                input_ids = self.tokenizer.encode(prompt, add_special_tokens=self.add_special_tokens)
                label_id = self.tokenizer.encode(answer, add_special_tokens=False)
                if len(input_ids) >= self.seq_len:
                    input_ids = input_ids[:self.seq_len]
                if len(label_id) >= self.seq_len:
                    label_id = label_id[:self.seq_len]

                if self.need_pad:
                    input_ids = np.pad(input_ids, (0, self.seq_len - len(input_ids)), pad_mode,
                                       constant_values=(self.tokenizer.pad_token_id, self.tokenizer.pad_token_id))
                    label_id = np.pad(label_id, (0, self.seq_len - len(label_id)), pad_mode,
                                      constant_values=(self.tokenizer.pad_token_id, self.tokenizer.pad_token_id))

                self.subjects.append(subject)
                self.input_ids.append(Tensor(input_ids, dtype=dtype.int32))
                self.labels.append(Tensor(label_id, dtype=dtype.int32))
        # for train/finetune
        else:
            for subject, prompt, answer in zip(subjects, sources, targets):
                total_items += 1
                concated_qa = prompt + answer
                input_ids = self.tokenizer.encode(concated_qa, add_special_tokens=self.add_special_tokens)
                input_ids = np.array(input_ids)

                prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                prompt_ids = np.array(prompt_ids)
                prompt_length = len(prompt_ids)
                concat_length = len(input_ids)

                if self.need_pad:
                    pad_length = self.seq_len + 1 - concat_length
                    input_ids = np.pad(input_ids, (0, pad_length), pad_mode,
                                       constant_values=(self.tokenizer.pad_token_id, self.tokenizer.pad_token_id))
                label_id_new = copy.deepcopy(input_ids)
                label_id_new[:prompt_length] = self.ignore_token_id
                if self.need_pad:
                    label_id_new[-pad_length:] = self.ignore_token_id

                self.subjects.append(subject)
                self.input_ids.append(Tensor(input_ids, dtype=dtype.int32))
                self.labels.append(Tensor(label_id_new, dtype=dtype.int32))
        return total_items

    def __next__(self):
        return next(self.iter_subjects), next(self.iter_input_ids), next(self.iter_labels)

    def __iter__(self):
        self.iter_subjects = iter(self.subjects)
        self.iter_input_ids = iter(self.input_ids)
        self.iter_labels = iter(self.labels)
        return self


def create_ceval_dataset(ds_path: str, mode: str, bs: int, seq_length: int, tokenizer: callable,
                         ignore_token_id=-100, repeat=1, need_pad=True, n_samples=-1, add_special_tokens=True):
    """create squad dataset"""
    ds = CEvalDataset(ds_path, mode, seq_length, tokenizer, ignore_token_id, need_pad, n_samples, add_special_tokens)
    type_cast_op = C.TypeCast(dtype.int32)
    ds = ds.map(operations=type_cast_op, input_columns="input_ids")
    ds = ds.map(operations=type_cast_op, input_columns="labels")
    ds = ds.batch(bs, drop_remainder=True)
    ds = ds.repeat(repeat)
    return ds
