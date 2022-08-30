#!/bin/bash
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
CURRPATH=$(cd "$(dirname $0)" || exit; pwd)

if [ $# != 1 ] && [ $# != 2 ]; then
  echo "bash runtests.sh DATASET_PATH"
  echo "bash runtests.sh DATASET_PATH ALGO_FILTER"
fi

ds_path=$1

if [ $# == 2 ]; then
  algo=$2
else
  algo="all"
fi

export SELF_CHECK=True
export DATASET_PATH=$ds_path

if [ "x$algo" == "xall" ] || [ "x$algo" == "xsim_qat" ]; then
  echo "============================ start testing sim_qat"
  pytest -vra $CURRPATH/st/quantization/sim_qat/*.py
  RET=$?
  if [ ${RET} -ne 0 ]; then
      echo "============================ testing sim_qat failed"
      exit ${RET}
  fi
  echo "============================ testing sim_qat successfully"
fi

if [ "x$algo" == "xall" ] || [ "x$algo" == "xlsq" ]; then
  echo "============================ start testing lsq"
  pytest -vra $CURRPATH/st/quantization/lsq/*.py
  RET=$?
  if [ ${RET} -ne 0 ]; then
    echo "============================ testing lsq failed"
      exit ${RET}
  fi
  echo "============================ testing lsq successfully"
fi

echo "============================ finish all testcases"
