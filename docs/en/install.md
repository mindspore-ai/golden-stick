# Installing MindSpore Golden Stick

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/golden-stick/blob/master/docs/en/install.md)

[查看中文](../zh_cn/install.md)

## Environmental Restrictions

The following lists the system environment required for installing, compiling and running MindSpore Golden Stick:

- OS: openEuler/Ubuntu/Linux
- Python: >=3.9, <3.12

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
|          1.4.1          |  Maintenance       | 2026-01-08 | Expected to end maintenance on 2026-07-08   | 2026-10-08 |
|          1.4.0          |  Maintenance       | 2026-01-04 | Expected to end maintenance on 2026-07-04   | 2026-10-04 |
|          1.3.0          |  Maintenance       | 2025-10-23 | Expected to end maintenance on 2026-04-23   | 2026-07-23 |
|          1.2.0          |  Maintenance       | 2025-08-13 | Expected to end maintenance on 2026-02-13   | 2026-05-13 |
|          1.1.0          |  Unmaintained       | 2025-05-21 | Expected EOL on 2026-02-21                 | 2026-02-21 |
|          1.0.0          | End of Life | 2025-02-13 |                        | 2025-11-13 |
|          0.6.0          | End of Life | 2024-10-30 |                        | 2025-07-30 |
|          0.5.0          | End of Life | 2024-08-01 |                        | 2025-05-01 |
|          0.4.1          | End of Life | 2024-07-15 |                        | 2025-04-15 |
|          0.4.0          | End of Life | 2024-03-30 |                        | 2024-12-30 |
|          0.3.0          | End of Life | 2023-06-15 |                        | 2024-03-15 |
|       0.3.0-alpha       | End of Life | 2023-02-01 |                        | 2023-11-01 |
|          0.2.0          | End of Life | 2022-10-26 |                        | 2023-07-26 |
|          0.1.0          | End of Life | 2022-07-29 |                        | 2023-04-29 |

## Version Dependency Mapping

MindSpore Golden Stick has version dependencies on MindSpore, MindOne, and MindFormers, as shown in the table below:

| MindSpore Golden Stick Version |                             Branch                                  | MindSpore Version | MindSpore Transformers Version |
| :---------------------: | :-----------------------------------------------------------------: | :----------: | :------------------------: |
|          1.4.1          | [r1.4](https://gitee.com/mindspore/golden-stick/tree/r1.4.0/)       | 2.7.1.post1  |       1.7.0                |
|          1.4.0          | [r1.4](https://gitee.com/mindspore/golden-stick/tree/r1.4.0/)       |   2.7.1      |       1.7.0                |
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

> Early versions of Golden Stick do not involve MindSpore Transformers, hence marked as NA in the table.

## Installing CANN

1. Refer to the version dependency mapping table to find the corresponding MindSpore version;
2. Check the recommended CANN version on the [MindSpore official website](https://www.mindspore.cn/versions);
3. Download and install drivers and firmware from the [CANN Developer Website](https://www.hiascend.com/hardware/firmware-drivers/community);
4. Install and configure the CANN environment according to the [CANN Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=netconda&OS=openEuler&Software=cannToolKit).

## Installing MindSpore

- **Version 1.4.0 and later**: MindSpore is included in MindSpore Golden Stick's `requirements.txt`, so users no longer need to manually install this dependency.
- **Before version 1.4.0**: Users need to install MindSpore using pip according to the version dependency mapping table, or download the corresponding installation package from the [MindSpore official website](https://www.mindspore.cn/install).

## Installing MindFormers

- **Version 1.4.0 and later**: MindFormers (formerly MindSpore Transformers) is no longer a required dependency of MindSpore Golden Stick. Only when quantizing models from the MindFormers ecosystem do users need to install the corresponding MindFormers version according to the version dependency mapping table.
- **Before version 1.4.0**: Users need to install the corresponding MindFormers version using pip according to the version dependency mapping table.

```bash
pip install mindformers==1.7.0
```

## Installing MindOne

**Version 1.4.0 and later**: MindSpore Golden Stick supports quantizing models from the MindOne ecosystem. Since different models in MindOne may be supported in different versions, please install the corresponding MindOne version according to the README of specific models in [Golden Stick Model Card](https://gitee.com/mindspore/golden-stick/tree/master/example/model_cards).

## Installing MindSpore Golden Stick

MindSpore Golden Stick can be installed via pip or by compiling from source code.

### Installing from pip Command

We maintain the [MindSpore Golden Stick project](https://pypi.org/project/mindspore-gs/) on PyPI, which can be installed directly using the pip command.

```shell
pip install mindspore-gs
```

### Installing from Source Code

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
