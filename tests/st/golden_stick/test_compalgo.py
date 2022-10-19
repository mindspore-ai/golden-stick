# Copyright 2022 Huawei Technologies Co., Ltd
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
"""test CompAlgo."""

import pytest
import numpy as np

from mindspore_gs.comp_algo import CompAlgo, ExportMindIRCallBack
import mindspore
from mindspore import Tensor


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_init():
    """
    Feature: CompAlgo init algorithm.
    Description: Initialize a CompAlgo.
    Expectation: Success.
    """

    algo = CompAlgo({})
    assert algo


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_callback():
    """
    Feature: CompAlgo callback export.
    Description: Initialize a CompAlgo and set export MindIR automatically after training.
    Expectation: Success.
    """

    algo = CompAlgo({})
    algo.set_save_mindir(save_mindir=True)
    algo.set_save_mindir_path(save_mindir_path="test")
    algo.set_save_mindir_inputs(Tensor(np.ones(1), mindspore.float32))
    cb = algo.callbacks()
    assert cb
    assert isinstance(cb[0], ExportMindIRCallBack)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_callback_error():
    """
    Feature: CompAlgo callback export.
    Description: Initialize a CompAlgo and set export MindIR automatically after training.
    Expectation: Expect error.
    """

    algo = CompAlgo({})
    algo.set_save_mindir(save_mindir=True)
    has_error = False
    try:
        _ = algo.callbacks()
        has_error = True
    except RuntimeError:
        pass
    assert not has_error


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_save_mindir():
    """
    Feature: set_save_mindir api of CompAlgo.
    Description: Input invalid value and expect error.
    Expectation: Expect error.
    """
    qat = CompAlgo({})
    has_error = False
    try:
        qat.set_save_mindir(1)
        has_error = True
    except TypeError:
        pass
    assert not has_error


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_save_mindir_path():
    """
    Feature: set_save_mindir_path api of CompAlgo.
    Description: Input invalid value and expect error.
    Expectation: Expect error.
    """
    qat = CompAlgo({})
    has_error = False
    try:
        qat.set_save_mindir(save_mindir=True)
        qat.set_save_mindir_path(1)
        has_error = True
    except TypeError:
        pass
    assert not has_error


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_save_mindir_inputs():
    """
    Feature: set_save_mindir_inputs api of CompAlgo.
    Description: Input invalid value and expect error.
    Expectation: Expect error.
    """
    qat = CompAlgo({})
    has_error = False
    try:
        qat.set_save_mindir(save_mindir=True)
        qat.set_save_mindir_inputs(None)
        has_error = True
    except RuntimeError:
        pass
    assert not has_error

    try:
        qat.set_save_mindir(save_mindir=True)
        qat.set_save_mindir_inputs({"a": 1})
        has_error = True
    except RuntimeError:
        pass
    assert not has_error
