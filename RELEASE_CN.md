# MindSpore Golden Stick Release Notes

[View English](./RELEASE.md)

## MindSpore Golden Stick 1.0.0 Release Notes

### 主要特性和增强

* 训练后量化算法 `PTQ` 支持 `GPTQ` 量化算法，可对权重进行8bit或4bit量化。`GPTQ` 已添加至精度恢复算法集中，可通过 `PTQConfig` 中的 `precision_recovery` 来进行算法选择，当前精度恢复算法仅 `GPTQ` 算法可选。
* 训练后量化算法 `PTQ` 支持 `AWQ` 量化算法，通过新增一种异常值抑制方法来使能 `AWQ` ，对权重进行4bit量化。可通过 `PTQConfig` 中的 `outliers_suppression` 来进行异常值抑制方法选择，当前可选 `smooth` 和 `awq` 两种方法。
* 训练后量化算法 `PTQ` 支持激活per-token动态量化，实现对激活的在线量化。可通过 `PTQConfig` 中的 `act_quant_granularity=QuantGranularity.PER_TOKEN`进行配置。

### API变更

* `RoundToNearest` 和 `SmoothQuant` 量化方法已经被废弃，请使用 `PTQ` 进行代替。

### 贡献者

感谢以下人员做出的贡献:

huangzhuo, zhangminli, ccsszz, yyyyrf, hangangqiang

欢迎以任何形式对项目提供贡献！

## MindSpore Golden Stick 0.6.0 Release Notes

### 主要特性和增强

* `RoundToNearest` 支持将MindFormers的kvcache即 `PagedAttentionMgr` 类量化成int8，主要针对Llama2系列网络。
* 新增训练后量化算法 `PTQ`，支持SmoothQuant、A16W8、KVCacheInt8以及他们之间的组合算法，比如A16W8组合KVCacheInt8，SmoothQuant组合KVCacheInt8等，可以通过配置PTQConfig获取相应的算法能力。该算法主要支持MindFormers社区的ParallelLlama2网络。

### API变更

* `PTQConfig`新增如下三个参数:
    * `act_quant_dtype`：mindspore.dtype类型，默认为None，可选输入及含义如下：

        |  act_quant_dtype  | mindspore.dtype.int8  | None（默认）  |
        |  ----  | ----  | ----  |
        | 含义  | 将激活量化成int8 | 不进行激活量化 |
    * `weight_quant_dtype`：mindspore.dtype类型，默认为mindspore.dtype.int8，可选输入及含义如下：

        |  weight_quant_dtype  | mindspore.dtype.int8（默认）  | None  |
        |  ----  | ----  | ----  |
        | 含义  | 将权重量化成int8 | 不进行权重量化 |
    * `kvcache_quant_dtype`：mindspore.dtype类型，默认为None，可选输入及含义如下：

        |  kvcache_quant_dtype  | mindspore.dtype.int8  | None（默认）  |
        |  ----  | ----  | ----  |
        | 含义  | 将kvcache量化成int8 | 不进行kvcache量化 |
    * `outliers_suppression`：OutliersSuppressionType类型，默认为OutliersSuppressionType.NONE，可选输入及含义如下：

        |  outliers_suppression  | OutliersSuppressionType.SMOOTH  | OutliersSuppressionType.NONE（默认）  |
        |  ----  | ----  | ----  |
        | 含义  | 使用smooth方法对权重和激活进行异常值抑制 | 不进行异常值抑制 |

### 贡献者

感谢以下人员做出的贡献:

ccsszz, yyyyrf, hangangqiang

欢迎以任何形式对项目提供贡献！

## MindSpore Golden Stick 0.5.0 Release Notes

### 主要特性和增强

* [DEMO] 新增了主要针对Llama2网络的训练后量化W8A8算法 `SmoothQuant`。

### API变更

* `CompAlgo`类的`apply`新增`kwargs`参数作为子类的可扩展入参。
* 添加了 `NetworkHelper` 抽象类作为适配器，用于算法和框架之间的解耦。
* 添加了 `MFLlama2Helper` 类作为算法和 MindFormers 之间的适配器。
* [DEMO] 新增 `SmoothQuant` 类作为SmoothQuant算法的入口。
* 新增参数检查，确认 `RoundToNearest` 算法仅支持BackendTarget.ASCEND作为后端。

### 贡献者

感谢以下人员做出的贡献:

ccsszz, yyyyrf, hangangqiang

欢迎以任何形式对项目提供贡献！

## MindSpore Golden Stick 0.4.1 Release Notes

### 主要特性和增强

* 优化`RoundToNearest`算法量化checkpoint的耗时。
* 优化`RoundToNearest`量化网络推理时的图编译耗时。

### 贡献者

感谢以下人员做出的贡献:

ccsszz, yyyyrf, hangangqiang

欢迎以任何形式对项目提供贡献！

## MindSpore Golden Stick 0.4.0 Release Notes

### 主要特性和增强

* 新增W8A16训练后权重量化算法`RoundToNearest`，实现对Llama2 7B/13B/70B、Baichuan2 13B网络无损压缩参数40%+。

### API变更

* 新增`PTQConfig`，用于配置训练后量化算法。
* 新增`PTQMode`枚举类，可以在`PTQConfig`中进行配置，用于区分量化算法的两个阶段：量化阶段和部署阶段。
* 新增`BackendTarget`枚举类，可以在`PTQConfig`中进行配置，表达量化的网络最终要部署到什么后端，比如`BackendTarget.Ascend`表示最终要部署到MindSpore的昇腾后端上。

### 贡献者

感谢以下人员做出的贡献:

zhuxiaoxiong, hangangqiang

欢迎以任何形式对项目提供贡献！

## MindSpore Golden Stick 0.3.0 Release Notes

### Bug修复

* 修复SCOP剪枝算法训练无法收敛的问题。

### 贡献者

感谢以下人员作出的贡献：

hangangqiang, yangruoqi713, kevinkunkun

欢迎以任意形式对项目提供贡献!

## MindSpore Golden Stick 0.3.0-alpha Release Notes

### 主要特性和增强

* [stable] SLB（Searching for Low-Bit Weights in Quantized Neural Networks）感知量化算法支持BatchNorm矫正能力。可以通过`set_enable_bn_calibration`接口来配置使能。对于存在BatchNorm层的网络，BatchNorm矫正能力减少SLB量化算法产生的网络准确率下降。([!150](https://gitee.com/mindspore/golden-stick/pulls/150))
* [stable] 验证了SimQAT（Simulated Quantization Aware Training）算法和SLB算法在ResNet网络，Imagenet2012数据集上的量化效果，详细效果参见[MindSpore Models仓readme](https://gitee.com/mindspore/models/tree/r2.0.0-alpha/official/cv/ResNet#%E7%BB%93%E6%9E%9C-4)。
* [stable] 打通了SimQAT算法在Lite上的部署流程，并验证了LeNet网络的部署效果，详细效果参见[MindSpore官网SimQAT量化算法推理部署效果](https://www.mindspore.cn/golden_stick/docs/zh-CN/master/quantization/simqat.html#%E9%83%A8%E7%BD%B2%E6%95%88%E6%9E%9C)。

### API变更

#### 兼容性变更

* SLB算法新增`set_enable_bn_calibration`接口，用于配置是否需要使能BatchNorm矫正能力。([!150](https://gitee.com/mindspore/golden-stick/pulls/150))
* 算法基类CompAlgo新增 `convert` 接口，用于在训练后将网络转换为推理网络，推理网络将被导出为MindIR进行推理部署，具体使用方法详见[模型部署文档](https://www.mindspore.cn/golden_stick/docs/zh-CN/r0.3.0-alpha/deployment/convert.html#%E8%AE%AD%E7%BB%83%E5%90%8E%E5%AF%BC%E5%87%BAmindir)。([!176](https://gitee.com/mindspore/golden-stick/pulls/176/files))
* 算法基类CompAlgo新增 `set_save_mindir` 接口，配置在训练后自动导出MindIR，具体使用方法详见[模型部署文档](https://www.mindspore.cn/golden_stick/docs/zh-CN/r0.3.0-alpha/deployment/convert.html#%E9%85%8D%E7%BD%AE%E7%AE%97%E6%B3%95%E8%87%AA%E5%8A%A8%E5%AF%BC%E5%87%BAmindir)。([!168](https://gitee.com/mindspore/golden-stick/pulls/168/files))

### Bug修复

* [STABLE] 重构SimQAT算法代码，解决量化过程中激活算子丢失、pre-trained参数丢失、伪量化算子冗余等问题。

### 贡献者

感谢以下人员作出的贡献：

liuzhicheng01, fuzhongqian, hangangqiang, yangruoqi713, kevinkunkun.

欢迎以任意形式对项目提供贡献!

## MindSpore Golden Stick 0.2.0 Release Notes

### 主要特性和增强

* [STABLE] SLB（Searching for Low-Bit Weights in Quantized Neural Networks）感知量化算法通过内置温度调节机制来简化算法的应用方式，用户训练脚本中不在需要手动编写温度调节的逻辑，通过算法配置接口即可实现原来的温度调节功能。

### Bug修复

* [STABLE] 解决多卡训练时AllReduce算子的一个bug，从而SLB感知量化算法得以支持多卡训练。

### API变更

#### 非兼容性变更

#### Python API

* 算法基类CompAlgo新增`callbacks`接口，返回算法在训练过程中的回调逻辑，为了方便不同算法实现各自的回调逻辑，该算法为变参输入。([!117](https://gitee.com/mindspore/golden-stick/pulls/117))
* SLB算法新增`set_epoch_size`接口，用于配置当前训练的总epoch数，用于温度调节逻辑的实现。([!117](https://gitee.com/mindspore/golden-stick/pulls/117))
* SLB算法新增`set_has_trained_epoch`接口，如果训练中使用了预训练的checkpoing，请通过该接口配置当前训练中使用的预训练checkpoint对应的预训练轮数，用于温度调节逻辑的实现。([!117](https://gitee.com/mindspore/golden-stick/pulls/117))
* SLB算法新增`set_t_start_val`接口，用于配置温度调节机制中温度的初始值，用于温度调节逻辑的实现。([!117](https://gitee.com/mindspore/golden-stick/pulls/117))
* SLB算法新增`set_t_start_time`接口，用于配置温度调节机制开始生效的时间点，用于温度调节逻辑的实现。([!117](https://gitee.com/mindspore/golden-stick/pulls/117))
* SLB算法新增`set_t_end_time`接口，用于配置温度调节机制停止生效的时间点，用于温度调节逻辑的实现。([!117](https://gitee.com/mindspore/golden-stick/pulls/117))
* SLB算法新增`set_t_factor`接口，用于配置温度调节机制中的温度调节因子，用于温度调节逻辑的实现。([!117](https://gitee.com/mindspore/golden-stick/pulls/117))

### 贡献者

感谢以下人员作出的贡献：

ghostnet, liuzhicheng01, fuzhongqian, hangangqiang, cjh9368, yangruoqi713, kevinkunkun.

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
