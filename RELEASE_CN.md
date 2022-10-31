# MindSpore Golden Stick Release Notes

[View English](./RELEASE.md)

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
