# Installing MindSpore Golden Stick

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/golden-stick/blob/master/docs/en/install.md)

[查看中文](../docs_zh_cn/install.md)

## Environmental Restrictions

The following table lists the environment required for installing, compiling and running MindSpore Golden Stick:

| Software |  Version   |
| :-----: | :-----: |
| OS  | openEuler/Ubuntu/Linux |
| Python  |  >=3.9, <3.12 |

> For other third-party dependencies, please refer to the [requirements file](https://gitee.com/mindspore/golden-stick/blob/master/requirements.txt).

## Version Lifecycle and Version Compatibility Strategy

MindSpore Golden Stick version has the following five maintenance phases:

| **Status**        | **Duration** | **Description**                                                                                                             |
|-------------------|--------------|-----------------------------------------------------------------------------------------------------------------------------|
| Plan              | 1-3 months   | Planning function.                                                                                                          |
| Develop           | 3 months     | Build function.                                                                                                             |
| Preserve          | 6 months     | Incorporate all solved problems and release new versions.                                                                   |
| No Preserve       | 0—3 months   | Incorporate all the solved problems, there is no full-time maintenance team, and there is no plan to release a new version. |
| End of Life (EOL) | N/A          | The branch is closed and no longer accepts any modifications.                                                               |

### Historical Version Lifecycle

| MindSpore Golden Stick Version | Current Status  |   Release Date  |         Next Status         |   EOL Date   |
| :---------------------: | :--------: | :--------: | :--------------------: | :--------: |
|          1.3.0          |  Maintenance       | 2025-10-23 | Expected to end maintenance on 2026-04-23   | 2026-07-23 |
|          1.2.0          |  Maintenance       | 2025-08-13 | Expected to end maintenance on 2026-02-13   | 2026-05-13 |
|          1.1.0          |  Maintenance       | 2025-05-21 | Expected to end maintenance on 2026-11-21   | 2026-02-21 |
|          1.0.0          | Unmaintained      | 2025-02-13 | Expected EOL on 2025-11-23| 2025-11-13 |
|          0.6.0          | End of Life | 2024-10-30 |                        | 2025-07-30 |
|          0.5.0          | End of Life | 2024-08-01 |                        | 2025-05-01 |
|          0.4.1          | End of Life | 2024-07-15 |                        | 2025-04-15 |
|          0.4.0          | End of Life | 2024-03-30 |                        | 2024-12-30 |
|          0.3.0          | End of Life | 2023-06-15 |                        | 2024-03-15 |
|       0.3.0-alpha       | End of Life | 2023-02-01 |                        | 2023-11-01 |
|          0.2.0          | End of Life | 2022-10-26 |                        | 2023-07-26 |
|          0.1.0          | End of Life | 2022-07-29 |                        | 2023-04-29 |

### Historical Version Dependencies

MindSpore Golden Stick depends on MindSpore and MindSpore Transformers repositories. Please install the corresponding versions of MindSpore and MindSpore Transformers according to the relationships shown in the table below:

> Starting from version 1.3.0, MindSpore and MindSpore Transformers are officially included in MindSpore Golden Stick's requirements.txt, so users no longer need to manually install these two dependencies.

| MindSpore Golden Stick Version |                             Branch                                  | MindSpore Version | MindSpore Transformers Version |
| :---------------------: | :-----------------------------------------------------------------: | :----------: | :------------------------: |
|          1.3.0          | [r1.3](https://gitee.com/mindspore/golden-stick/tree/r1.3.0/)       |   2.7.1      |       1.7.0                |
|          1.2.0          | [r1.2](https://gitee.com/mindspore/golden-stick/tree/r1.2.0/)       |   2.7.0      |       1.6.0                |
|          1.1.0          | [r1.1](https://gitee.com/mindspore/golden-stick/tree/r1.1.0/)       |   2.6.0      |       1.5.0                |
|          1.0.0          | [r1.0](https://gitee.com/mindspore/golden-stick/tree/r1.0.0/)       |   2.5.0      |       1.4.0-beta2          |
|          0.6.0          | [r0.6](https://gitee.com/mindspore/golden-stick/tree/r0.6.0/)       |   2.4.0      |       1.3.0                |
|          0.5.0          | [r0.5](https://gitee.com/mindspore/golden-stick/tree/r0.5.0/)       |   2.3.1      |       1.2.0                |
|          0.4.1          | [r0.4](https://gitee.com/mindspore/golden-stick/tree/r0.4.1/)       |   2.3.0      |       1.2.0                |
|          0.4.0          | [r0.4](https://gitee.com/mindspore/golden-stick/tree/r0.4/)         |   2.3.0-rc1  |        NA                  |
|          0.3.0          | [r0.3](https://gitee.com/mindspore/golden-stick/tree/r0.3/)         |   2.0.0-rc1, 2.0.0   |    NA              |
|       0.3.0-alpha       | [r0.3](https://gitee.com/mindspore/golden-stick/tree/v0.3.0-alpha/) |   2.0.0-alpha        |        NA          |
|          0.2.0          | [r0.2](https://gitee.com/mindspore/golden-stick/tree/r0.2/)         |   1.9.0      |        NA                  |
|          0.1.0          | [r0.1](https://gitee.com/mindspore/golden-stick/tree/r0.1/)         |   1.8.0      |        NA                  |

> Early versions of Golden Stick do not support MindSpore Transformers, hence marked as NA in the table.

## Installing MindSpore Golden Stick

After installing MindSpore and MindSpore Transformers, proceed to install MindSpore Golden Stick. You can use either pip installation or source code compilation installation.

## Installing from pip Command

We maintain the [MindSpore Golden Stick project](https://pypi.org/project/mindspore-gs/) on PyPI, which can be installed directly using the pip command.

```shell
pip install mindspore-gs
```

## Installing from Source Code

Download the [source code](https://gitee.com/mindspore/golden-stick), then enter the `golden-stick` directory after downloading.

```shell
git clone https://gitee.com/mindspore/golden-stick.git
cd golden-stick
bash build.sh
```

After successful compilation, a whl package will be generated in the `output` directory, which can be installed using pip.

## Verifying Installation Success

Execute the following commands to verify the installation result. If importing the Python modules does not report errors, the installation is successful:

```python
import mindspore_gs
import mindspore_gs.ptq
```
