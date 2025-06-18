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
"""cholesky_trans."""

import os
import ctypes
import numpy as np
from mindspore import Tensor, numpy
from mindspore import ops as msops
from mindspore_gs.common import logger


dll = ctypes.cdll.LoadLibrary
env_var = ""
use_kml = False
LIBRARY_PATH = os.environ['LD_LIBRARY_PATH'].split(':')
for path in LIBRARY_PATH:
    path_names = path.split('/')
    for name in path_names:
        if name == 'libklapack_full.so':
            use_kml = True
            env_var = path
if env_var == "":
    logger.info("Notice: The kml library is not used for cholesky decomposition.")
else:
    lib_lapack = dll(env_var)
    spotrf = lib_lapack.spotrf_
    spotrf.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    spotri = lib_lapack.spotri_
    spotri.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]

def cholesky_compute(h, damp_percent=0.01):
    """compute cholesky decomposition"""
    damp = damp_percent * msops.mean(numpy.diag(h))
    if np.isnan(damp.asnumpy()) or np.isinf(damp.asnumpy()):
        raise ValueError("The damping is NaN or Inf.")
    diag = msops.arange(h.shape[0])
    h[diag, diag] += damp
    if use_kml:
        damp_sum = 0.
        flag = True
        threshold = 10
        while flag:
            # Note: cholesky_transform will change H with inplace way
            hinv, flag = cholesky_transform(h.numpy().copy())
            if not flag:
                hinv = Tensor(hinv)
                logger.info(f"Successful! the accumulated damp is {damp_sum}")
                break
            h[diag, diag] += damp
            damp_sum += damp
            logger.info(f"Add more damping, the current damp is {damp}")
            if int(damp_sum / damp) > threshold:
                raise StopIteration(f"Too many damping iterations ({int(damp_sum / damp) - 1})")
    else:
        h = np.linalg.cholesky(h.numpy())
        h = np.linalg.inv(h.T) @ np.linalg.inv(h)
        hinv = Tensor(np.linalg.cholesky(h).T)
    return hinv

def cholesky_transform(h):
    """
        This function is used to compute the Cholesky decomposition result of the Hessian matrix by kml library.
    """
    shape = h.shape
    uplo = ctypes.c_char(b'U')
    n = ctypes.c_int(shape[0])
    lda = ctypes.c_int(shape[0])
    info = ctypes.c_int(0)
    a = h.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    spotrf(ctypes.byref(uplo), ctypes.byref(n), a, ctypes.byref(lda), ctypes.byref(info))
    if info.value != 0:
        return h, True

    info = ctypes.c_int(0)
    spotri(ctypes.byref(uplo), ctypes.byref(n), a, ctypes.byref(lda), ctypes.byref(info))
    if info.value != 0:
        return h, True

    info = ctypes.c_int(0)
    spotrf(ctypes.byref(uplo), ctypes.byref(n), a, ctypes.byref(lda), ctypes.byref(info))
    if info.value != 0:
        return h, True

    tril_h = np.tril(h)
    trans_h = np.transpose(tril_h)
    return trans_h, False
