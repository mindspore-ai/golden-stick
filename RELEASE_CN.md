# MindSpore Golden Stick Release Notes

[View English](./RELEASE.md)

## MindSpore Golden Stick 0.3.0 Release Notes

### 主要特性和增强

* [stable] SLB（Searching for Low-Bit Weights in Quantized Neural Networks）感知量化算法支持BatchNorm矫正能力。可以通过`set_enable_bn_calibration`接口来配置使能。对于存在BatchNorm层的网络，BatchNorm矫正能力减少SLB量化算法产生的网络准确率下降。([!150](https://gitee.com/mindspore/golden-stick/pulls/150))
* [stable] 验证了SimQAT（Simulated Quantization Aware Training）算法和SLB算法在ResNet网络，Imagenet2012数据集上的量化效果，详细效果参见[MindSpore Models仓readme](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ResNet#%E7%BB%93%E6%9E%9C-4)。

### API变更

#### 非兼容性变更

#### 兼容性变更

* SLB算法新增`set_enable_bn_calibration`接口，用于配置是否需要使能BatchNorm矫正能力。([!150](https://gitee.com/mindspore/golden-stick/pulls/150))
* 算法基类CompAlgo新增 `set_save_mindir` 接口，配置在训练后自动导出MindIR，具体使用方法详见[模型部署文档](https://www.mindspore.cn/golden_stick/docs/zh-CN/master/deployment/convert.html#%E9%85%8D%E7%BD%AE%E7%AE%97%E6%B3%95%E8%87%AA%E5%8A%A8%E5%AF%BC%E5%87%BAmindir)。([!168](https://gitee.com/mindspore/golden-stick/pulls/168/files))

### Bug修复

### 贡献者

感谢以下人员作出的贡献：

liuzhicheng01, fuzhongqian, hangangqiang, yangruoqi713, kevinkunkun.

欢迎以任意形式对项目提供贡献!

## MindSpore Golden Stick 0.2.0 Release Notes

MindSpore Golden Stick是华为诺亚团队和华为MindSpore团队联合设计开发的一个模型压缩算法集，提供了一套统一的算法应用接口，让用户能够统一方便地使用例如量化、剪枝等等模型压缩算法，同时提供前端网络修改能力，降低算法接入成本。MindSpore Golden Stick当前版本提供了三个算法。

### 主要特性和增强

* [BETA] 提供一个名为SimQAT（Simulated Quantization Aware Training）的感知量化算法，是一种最基本的感知量化算法。
* [BETA] 提供一个名为SLB（Searching for Low-Bit Weights in Quantized Neural Networks）的感知量化算法，是一种非线性、高精度的感知量化算法，在低比特量化上优势明显。
* [STABLE] 提供一个名为SCOP（Scientific Control for Reliable Neural Network Pruning）的剪枝算法，是一种高精度的结构化剪枝算法，当前主要应用于CV网络上。

### 贡献者

感谢以下人员作出的贡献：

ghostnet, liuzhicheng01, fuzhongqian, hangangqiang, cjh9368.

欢迎以任意形式对项目提供贡献!

## MindSpore Golden Stick 0.1.0 Release Notes

MindSpore Golden Stick是华为诺亚团队和华为MindSpore团队联合设计开发的一个模型压缩算法集，提供了一套统一的算法应用接口，让用户能够统一方便地使用例如量化、剪枝等等模型压缩算法，同时提供前端网络修改能力，降低算法接入成本。MindSpore Golden Stick当前版本提供了三个算法。

### 主要特性和增强

* [BETA] 提供一个名为SimQAT（Simulated Quantization Aware Training）的感知量化算法，是一种最基本的感知量化算法。
* [BETA] 提供一个名为SLB（Searching for Low-Bit Weights in Quantized Neural Networks）的感知量化算法，是一种非线性、高精度的感知量化算法，在低比特量化上优势明显。
* [STABLE] 提供一个名为SCOP（Scientific Control for Reliable Neural Network Pruning）的剪枝算法，是一种高精度的结构化剪枝算法，当前主要应用于CV网络上。

### 贡献者

感谢以下人员作出的贡献：

ghostnet, liuzhicheng01, fuzhongqian, hangangqiang, cjh9368.

欢迎以任意形式对项目提供贡献!