# MindSpore Golden Stick Release Notes

[查看中文](./RELEASE_CN.md)

## MindSpore Golden Stick 0.1.0 Release Notes

MindSpore Golden Stick is a model compression algorithm set jointly designed and developed by Huawei's Noah team and Huawei's MindSpore team. MindSpore Golden Stick provides an unified user interface allowing users to apply model compression algorithms such as quantization and pruning in a unified and convenient manner. MindSpore Golden Stick also provides front-end network modification capabilities to reduce algorithm development costs. MindSpore Golden Stick provides three algorithms in current version.

### Major Features and Improvements

- [BETA] Provides a quantization aware training algorithm named SimQAT (Simulated Quantization Aware Training), which is the most basic quantization aware training algorithm.
- [BETA] Provides a quantization aware training algorithm called SLB (Searching for Low-Bit Weights in Quantized Neural Networks), which is a nonlinear, high-precision quantization aware training algorithm with obvious advantages in low-bit quantization.
- [STABLE] Provides a pruning algorithm named SCOP (Scientific Control for Reliable Neural Network Pruning), which is a high-precision structured pruning algorithm and is mainly used in CV networks at present.

### API Change

#### Backwards Compatible Change

##### Python API

### Contributors

Thanks goes to these wonderful people:

ghostnet, liuzhicheng01, fuzhongqian, hangangqiang, cjh9368.

Contributions of any kind are welcome!
