#!/bin/bash
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

if [ $# -ne 1 ]; then
  script_name=$(basename "$0")
  echo "usage: bash ${script_name} version, available version: master, r0.6.0, r0.5.0, r0.2.0; etc"
  exit 1
fi

version=$1

CUR_DIR=$(cd "$(dirname $0)"; pwd)

# mindspore_gs/pruner/README.md
sed -i "s/\.\/scop\/README_CN\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/zh-CN\/${version}\/pruner\/scop\.html/g" ${CUR_DIR}/../mindspore_gs/pruner/README_CN.md
sed -i "s/\.\/scop\/README\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/en\/${version}\/pruner\/scop\.html/g" ${CUR_DIR}/../mindspore_gs/pruner/README.md

# mindspore_gs/ptq/README.md
sed -i "s/\.\.\/quantization\/README_CN\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/zh-CN\/${version}\/quantization\/overview\.html/g" ${CUR_DIR}/../mindspore_gs/ptq/README_CN.md
sed -i "s/\.\/ptq\/README_CN\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/zh-CN\/${version}\/ptq\/ptq\.html/g" ${CUR_DIR}/../mindspore_gs/ptq/README_CN.md
sed -i "s/\.\/round_to_nearest\/README_CN\.ipynb/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/zh-CN\/${version}\/ptq\/round_to_nearest\.html/g" ${CUR_DIR}/../mindspore_gs/ptq/README_CN.md

sed -i "s/\.\.\/quantization\/README\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/en\/${version}\/quantization\/overview\.html/g" ${CUR_DIR}/../mindspore_gs/ptq/README.md
sed -i "s/\.\/ptq\/README\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/en\/${version}\/ptq\/ptq\.html/g" ${CUR_DIR}/../mindspore_gs/ptq/README.md
sed -i "s/\.\/round_to_nearest\/README\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/en\/${version}\/ptq\/round_to_nearest\.html/g" ${CUR_DIR}/../mindspore_gs/ptq/README.md

# mindspore_gs/ptq/ptq/README.md
sed -i "s/\.\.\/round_to_nearest\/README_CN\.ipynb/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/zh-CN\/${version}\/ptq\/round_to_nearest\.html/g" ${CUR_DIR}/../mindspore_gs/ptq/ptq/README_CN.md
sed -i "s/\.\.\/round_to_nearest\/README\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/en\/${version}\/ptq\/round_to_nearest\.html/g" ${CUR_DIR}/../mindspore_gs/ptq/ptq/README.md

# mindspore_gs/quantization/README.md
sed -i "s/\.\/simulated_quantization\/README_CN\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/zh-CN\/${version}\/quantization\/simulated_quantization\.html/g" ${CUR_DIR}/../mindspore_gs/quantization/README_CN.md
sed -i "s/\.\/slb\/README_CN\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/zh-CN\/${version}\/quantization\/slb\.html/g" ${CUR_DIR}/../mindspore_gs/quantization/README_CN.md
sed -i "s/\.\.\/ptq\/README_CN\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/zh-CN\/${version}\/ptq\/overview\.html/g" ${CUR_DIR}/../mindspore_gs/quantization/README_CN.md

sed -i "s/\.\/simulated_quantization\/README\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/en\/${version}\/quantization\/simulated_quantization\.html/g" ${CUR_DIR}/../mindspore_gs/quantization/README.md
sed -i "s/\.\/slb\/README\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/en\/${version}\/quantization\/slb\.html/g" ${CUR_DIR}/../mindspore_gs/quantization/README.md
sed -i "s/\.\.\/ptq\/README\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/en\/${version}\/ptq\/overview\.html/g" ${CUR_DIR}/../mindspore_gs/quantization/README.md

# mindspore_gs/CONTRIBUTING.md
sed -i "s#\[README\](./README.md)#\[README\](https://atomgit.com/mindspore/golden-stick/tree/${version}/README.md)#g" ${CUR_DIR}/../CONTRIBUTING.md
sed -i "s#\[README\](./README_CN.md)#\[README\](https://atomgit.com/mindspore/golden-stick/tree/${version}/README_CN.md)#g" ${CUR_DIR}/../CONTRIBUTING_CN.md

sed -i "s#\[Architecture Design\](./docs/en/design\.md)#\[Architecture Design\]\(https://www.mindspore.cn/golden_stick/docs/en/${version}/design.html)#g" ${CUR_DIR}/../CONTRIBUTING.md
sed -i "s#\[架构设计\](./docs/zh_cn/design\.md)#\[架构设计\]\(https://www.mindspore.cn/golden_stick/docs/zh-CN/${version}/design.html)#g" ${CUR_DIR}/../CONTRIBUTING_CN.md

sed -i "s#\[Quick Start\](./example)#\[Quick Start\](https://atomgit.com/mindspore/golden-stick/tree/${version}/example/)#g" ${CUR_DIR}/../CONTRIBUTING.md
sed -i "s#\[快速入门\](./example)#\[快速入门\](https://atomgit.com/mindspore/golden-stick/tree/${version}/example/)#g" ${CUR_DIR}/../CONTRIBUTING_CN.md

sed -i "s#./docs/en/install.md#https://www.mindspore.cn/golden_stick/docs/en/${version}/install.html#g" ${CUR_DIR}/../CONTRIBUTING.md
sed -i "s#./docs/zh_cn/install.md#https://www.mindspore.cn/golden_stick/docs/zh-CN/${version}/install.html#g" ${CUR_DIR}/../CONTRIBUTING_CN.md

sed -i "s#./docs/en/design.md#https://www.mindspore.cn/golden_stick/docs/en/${version}/design.html#g" ${CUR_DIR}/../CONTRIBUTING.md
sed -i "s#./docs/zh_cn/design.md#https://www.mindspore.cn/golden_stick/docs/zh-CN/${version}/design.html#g" ${CUR_DIR}/../CONTRIBUTING_CN.md

sed -i "s#./scripts/pre_push/README.md#https://atomgit.com/mindspore/golden-stick/tree/${version}/scripts/pre_push/README.md#g" ${CUR_DIR}/../CONTRIBUTING.md
sed -i "s#./scripts/pre_push/README_CN.md#https://atomgit.com/mindspore/golden-stick/tree/${version}/scripts/pre_push/README_CN.md#g" ${CUR_DIR}/../CONTRIBUTING_CN.md

sed -i '/<!-- TOC -->/,/<!-- \/TOC -->/d' ${CUR_DIR}/../CONTRIBUTING_CN.md
sed -i '/<!-- TOC -->/,/<!-- \/TOC -->/d' ${CUR_DIR}/../CONTRIBUTING.md
