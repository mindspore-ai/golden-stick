# MindSpore Golden Stick Release Notes

[查看中文](./RELEASE_CN.md)

## MindSpore Golden Stick 0.3.0 Release Notes

### Major Features and Improvements

* [stable] SLB（Searching for Low-Bit Weights in Quantized Neural Networks）QAT algorithm now support BatchNorm calibration. we can invoke `set_enable_bn_calibration` api to enable BatchNorm calibration. For a network with a BatchNorm layer, the BatchNorm calibration can reduces the decrease in network accuracy caused by the SLB quantization algorithm. ([!150](https://gitee.com/mindspore/golden-stick/pulls/150))
* [stable] We verified the quantization effect of the SimQAT(Simulated Quantization Aware Training) algorithm and the SLB algorithm on the ResNet network and the Imagenet2012 dataset. For details, please refer to[MindSpore Models仓readme](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ResNet#%E7%BB%93%E6%9E%9C-4)。

### API Change

#### Backwards Incompatible Change

#### Backwards Compatible Change

* The SLB algorithm adds the `set_enable_bn_calibration` interface to enable or disable BatchNorm calibration.([!117](https://gitee.com/mindspore/golden-stick/pulls/117))
* Add `set_save_mindir` interface to the algorithm base class, which is configured to automatically export MindIR after training. For details, please refer to [Model Deployment](https://www.mindspore.cn/golden_stick/docs/en/master/deployment/convert.html#configure-the-algorithm-to-automatically-export-mindir).([!168](https://gitee.com/mindspore/golden-stick/pulls/168/files))

### Bug fixes

### Contributors

Thanks goes to these wonderful people:

liuzhicheng01, fuzhongqian, hangangqiang, yangruoqi713, kevinkunkun.

Contributions of any kind are welcome!

## MindSpore Golden Stick 0.2.0 Release Notes

MindSpore Golden Stick is a model compression algorithm set jointly designed and developed by Huawei's Noah team and Huawei's MindSpore team. MindSpore Golden Stick provides an unified user interface allowing users to apply model compression algorithms such as quantization and pruning in a unified and convenient manner. MindSpore Golden Stick also provides front-end network modification capabilities to reduce algorithm development costs. MindSpore Golden Stick provides three algorithms in current version.

### Major Features and Improvements

* [BETA] Provides a quantization aware training algorithm named SimQAT (Simulated Quantization Aware Training), which is the most basic quantization aware training algorithm.
* [BETA] Provides a quantization aware training algorithm called SLB (Searching for Low-Bit Weights in Quantized Neural Networks), which is a nonlinear, high-precision quantization aware training algorithm with obvious advantages in low-bit quantization.
* [STABLE] Provides a pruning algorithm named SCOP (Scientific Control for Reliable Neural Network Pruning), which is a high-precision structured pruning algorithm and is mainly used in CV networks at present.

### Contributors

Thanks goes to these wonderful people:

ghostnet, liuzhicheng01, fuzhongqian, hangangqiang, cjh9368.

Contributions of any kind are welcome!

## MindSpore Golden Stick 0.1.0 Release Notes

MindSpore Golden Stick is a model compression algorithm set jointly designed and developed by Huawei's Noah team and Huawei's MindSpore team. MindSpore Golden Stick provides an unified user interface allowing users to apply model compression algorithms such as quantization and pruning in a unified and convenient manner. MindSpore Golden Stick also provides front-end network modification capabilities to reduce algorithm development costs. MindSpore Golden Stick provides three algorithms in current version.

### Major Features and Improvements

* [BETA] Provides a quantization aware training algorithm named SimQAT (Simulated Quantization Aware Training), which is the most basic quantization aware training algorithm.
* [BETA] Provides a quantization aware training algorithm called SLB (Searching for Low-Bit Weights in Quantized Neural Networks), which is a nonlinear, high-precision quantization aware training algorithm with obvious advantages in low-bit quantization.
* [STABLE] Provides a pruning algorithm named SCOP (Scientific Control for Reliable Neural Network Pruning), which is a high-precision structured pruning algorithm and is mainly used in CV networks at present.

### Contributors

Thanks goes to these wonderful people:

ghostnet, liuzhicheng01, fuzhongqian, hangangqiang, cjh9368.

Contributions of any kind are welcome!
