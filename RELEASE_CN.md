# MindSpore Golden Stick Release Notes

[View English](./RELEASE.md)

## MindSpore Golden Stick 0.1.0 Release Notes

MindSpore Golden Stick是华为诺亚团队和华为MindSpore团队联合设计开发的一个模型压缩算法集，提供了一套统一的算法应用接口，让用户能够统一方便地使用例如量化、剪枝等等模型压缩算法，同时提供前端网络修改能力，降低算法接入成本。MindSpore Golden Stick当前版本提供了三个算法。

### 主要特性和增强

- [BETA] 提供一个名为SimQAT（Simulated Quantization Aware Training）的感知量化算法，是一种最基本的感知量化算法。
- [BETA] 提供一个名为SLB（Searching for Low-Bit Weights in Quantized Neural Networks）的感知量化算法，是一种非线性、高精度的感知量化算法，在低比特量化上优势明显。
- [STABLE] 提供一个名为SCOP（Scientific Control for Reliable Neural Network Pruning）的剪枝算法，是一种高精度的结构化剪枝算法，当前主要应用于CV网络上。

### 接口变更

#### 后向兼容变更

##### Python接口

### 贡献者

感谢以下人员作出的贡献：

ghostnet, liuzhicheng01, fuzhongqian, hangangqiang, cjh9368.

欢迎以任意形式对项目提供贡献!

[View English](./RELEASE_CN.md)
