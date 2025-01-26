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

""" Plot box figure. """

import numpy as np
import matplotlib.pyplot as plt


def plot_box(data: np.ndarray, out_path: str, title='box figure', x_label='x', y_label='y', x_labels=None):
    """plot_box"""
    if len(data.shape) != 2:
        raise ValueError("Only support 2D data.")
    x_len = data.shape[1]
    plt.figure(figsize=(14, 8))
    bplot = plt.boxplot(data, patch_artist=True, boxprops=dict(facecolor='#3274A1'))
    for element in ['medians']:
        plt.setp(bplot[element], color='black')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if not x_labels:
        x_labels = [f'{i + 1}' for i in range(0, x_len)]
    plt.xticks(range(1, x_len + 1), labels=x_labels)
    plt.savefig(out_path)
    plt.close()


def plot_seaborn_box(data: np.ndarray, out_path: str, title='box figure', x_label='x', y_label='y', x_labels=None):
    """plot_seaborn_box"""
    import seaborn as sns
    import pandas as pd
    if len(data.shape) != 2:
        raise ValueError("Only support 2D data.")
    x_len = data.shape[1]
    df = pd.DataFrame(data, columns=[f'{i + 1}' for i in range(x_len)])
    plt.figure(figsize=(14, 8))
    sns.set(style="whitegrid")
    sns.boxplot(data=df, palette=['#3274A1'] * x_len, width=0.5)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if not x_labels:
        x_labels = [f'{i + 1}' for i in range(0, x_len)]
    plt.xticks(range(0, x_len), labels=x_labels)
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    np.random.seed(0)
    samples = np.random.rand(100, 32)

    for i in range(0, 31, 4):
        samples[0, i] = 2
        samples[1, i] = -1

    plot_box(samples, "sample_matplot.png")
    plot_seaborn_box(samples, "sample_seaborn.png")
