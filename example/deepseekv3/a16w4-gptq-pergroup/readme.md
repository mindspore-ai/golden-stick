# DeepSeekR1网络GPTQ A16W4量化算法指南

本指南基于单机8卡，如果使用单机8卡，请将msrun命令替换为单机8卡形式。

运行前请检查yaml配置中的tp并行数，load_checkpoint配置，tokenizer配置是否合理。

## 1. 进行算法校准阶段

使用数据集进行量化校准。命令如下，默认配置为单机8卡，且使用的yaml文件为当前目录下的predict_deepseek_r1_671b_calibrate.yaml，需要修改yaml文件的load_checkpoint、vocab_file、tokenizer_file参数，确保output目录下有足够的空间存放量化后的权重，约370GB：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash calibrate.sh /path/to/mindformers
```

## 2. 测试量化网络对话精度

命令如下，默认配置为单机8卡，且使用的yaml文件为当前目录下的predict_deepseek_r1_671b_qinfer.yaml，load_checkpoint参数路径已设置为./output/DeepSeekR1-gptq-pergroup_safetensors，需要修改yaml文件的vocab_file、tokenizer_file参数：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash quant_infer.sh /path/to/mindformers
```

## 3. 量化网络ceval数据集评测

命令如下，默认配置为单机8卡，需要传入ceval数据集路径，使用的yaml文件与第二步相同，为当前目录下的predict_deepseek_r1_671b_qinfer.yaml，load_checkpoint参数路径已设置为默认路径./output/DeepSeekR1-gptq-pergroup_safetensors，需要修改yaml文件的vocab_file、tokenizer_file参数：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash eval_ceval.sh /path/to/mindformers /path/to/ceval_dataset_path
```

## 4. 生成的safetensors合一

首先生成策略文件，生成的策略文件ckpt_strategy.ckpt在当前目录下:

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash gen_strategy.sh /path/to/mindformers
```

然后进行合一，需要传入原始safetensors目录和目标safetensors目录。

```bash
bash unify_safetensors.sh /path/to/src_safetensors /path/to/dst_safetensors
```

注: gptq量化时需要进行cholesky分解，mindspore不支持在Ascend环境进行cholesky分解，因此需要安装鲲鹏数学库kml来进行cholesky分解计算。

# 鲲鹏数学库的安装说明

## 下载鲲鹏数学库(KML)及依赖的第三方包

KML库2.4.0版本ZIP包：https://kunpeng-repo.obs.cn-north-4.myhuaweicloud.com/Kunpeng%20HPCKit/Kunpeng%20HPCKit%2024.0.RC2/KML_2.4.0.zip
KML适配器最新版本ZIP包：https://gitee.com/openeuler/kml_adapter/repository/archive/99ab7ca00eca858d152fd1879247820dca6faf5a.zip
LAPACK包3.10.0版本ZIP包：https://codeload.github.com/Reference-LAPACK/lapack/zip/refs/tags/v3.10.0

鲲鹏数学库的安装与配置

## 1、创建kml目录，在该目录下执行以下命令安装kml库(kml.sh脚本代码在下方)

```sourcce
kml.sh
```

## 2、加入以下环境变量来使能KML库

```export
export LD_LIBRARY_PATH={kml库路径}/kml/lib/libklapack_full.so:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH={kml库路径}/kml/lib/kblas/omp:$LD_LIBRARY_PATH
```

## 附录：附上kml.sh代码参考，注意执行rpm -ivh kml-*.arrch64.rpm时需要取得root权限，安装该包请确保具有root权限。

```source
# 下载鲲鹏数学库(KML)及相关依赖包
# 加入KML_ROOT环境变量
export KML_ROOT=/usr/local/kml

# 在当前目录创建KML目录
mkdir KML
cd KML

# 解压KML库和第三方包
mv ../KML_2.4.0.zip . && unzip KML_2.4.0.zip
mv ../kml_adapter-99ab7ca00eca858d152fd1879247820dca6faf5a.zip . && \
unzip kml_adapter-99ab7ca00eca858d152fd1879247820dca6faf5a.zip && \
mv kml_adapter-99ab7ca00eca858d152fd1879247820dca6faf5a kml_adapter
mv ../lapack-3.10.0.zip . && unzip lapack-3.10.0.zip

# 安装KML，将在/usr/local/kml/lib下生成共享库文件
rpm -ivh kml-2.4.0-1.aarch64.rpm

# 加入LAPACK_SRC_DIR环境变量以指定LAPACK包的位置
export LAPACK_SRC_DIR=$PWD/lapack-3.10.0

# 在kml_adapter/lapack-adapt/build-full-klapack.sh文件的ldflags中加入如下链接选项以支持OpenMP和Kunpeng BLAS
sed -i '146a \ \ \ \ -fopenmp\n \ \ \ -L\/usr\/local\/kml\/lib\/kblas\/omp\n \ \ \ -lkblas' \
kml_adapter/lapack-adapt/build-full-klapack.sh

# 将会在kml_adapter/lapack_adapt/tmp-build-lapack/lib/下得到libklapack_full.so共享库文件
cd kml_adapter/lapack-adapt && \
./build-full-klapack.sh && cd ../../../

mv KML/kml_adapter/lapack-adapt/tmp-build-lapack/lib/libklapack_full.so $KML_ROOT/lib/
```