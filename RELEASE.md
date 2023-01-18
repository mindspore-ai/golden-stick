# MindSpore Golden Stick Release Notes

[查看中文](./RELEASE_CN.md)

## MindSpore Golden Stick 0.2.0 Release Notes

### Major Features and Improvements

* [STABLE] SLB(Searching for Low-Bit Weights in Quantized Neural Networks) QAT algorithm implements a built-in temperature adjustment callback to simplify the use of the algorithm. Users no longer need to manually write the temperature adjustment logic int the training script, and the original temperature adjustment function can be realized through the algorithm configuration interface. Note that this is an incompatible change.

### Bug fixes

* [STABLE] Solve a bug of AllReduce during distributed training, so that the SLB QAT algorithm can support distributed training.

### API Change

#### Backwards Incompatible Change

#### Python API

* Added `callbacks` interface to the algorithm base class, which returns the callback logic of the algorithm which will be called during the training process. In order to facilitate different algorithms to implement their own callback logic, this method has variable parameter inputs.([!117](https://gitee.com/mindspore/golden-stick/pulls/117))
* The SLB algorithm adds the `set_epoch_size` interface, which is used to configure the total number of epochs of training, and is used to implement the temperature adjustment callback logic.([!117](https://gitee.com/mindspore/golden-stick/pulls/117))
* The SLB algorithm adds the `set_has_trained_epoch` interface. If a pre-trained checkpoint is used in training, it is used to configure the number of pre-trained epochs corresponding to the pre-trained checkpoint used in the current training, which is used to implement the temperature adjustment callback logic.([!117](https://gitee.com/mindspore/golden-stick/pulls/117))
* The SLB algorithm adds the `set_t_start_val` interface, which is used to configure the initialization value of the temperature in the temperature adjustment mechanism, and is used to implement the temperature adjustment callback logic.([!117](https://gitee.com/mindspore/golden-stick/pulls/117))
* The SLB algorithm adds the `set_t_start_time` interface, which is used to configure the time when the temperature adjustment mechanism start to work, and is used to implement the temperature adjustment callback logic.([!117](https://gitee.com/mindspore/golden-stick/pulls/117))
* The SLB algorithm adds the `set_t_end_time` interface, which is used to configure the time when the temperature adjustment mechanism stop to work, and is used to implement the temperature adjustment callback logic.([!117](https://gitee.com/mindspore/golden-stick/pulls/117))
* The SLB algorithm adds the `set_t_factor` interface, which is used to configure the temperature adjustment factor in the temperature adjustment mechanism, and is used to implement the temperature adjustment callback logic.([!117](https://gitee.com/mindspore/golden-stick/pulls/117))

### Contributors

Thanks goes to these wonderful people:

ghostnet, liuzhicheng01, fuzhongqian, hangangqiang, cjh9368, yangruoqi713, kevinkunkun.

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
