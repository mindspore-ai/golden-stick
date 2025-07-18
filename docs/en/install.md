# Installing MindSpore Golden Stick

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/golden-stick/blob/master/docs/en/install.md)

[查看中文](../docs_zh_cn/install.md)

## Environmental Restrictions

The following table lists the environment required for installing, compiling and running MindSpore Golden Stick:

| software | version  |
| :-----: | :-----: |
| Ubuntu  |  18.04  |
| Python  |  3.9-3.10 |

> Please refer to [requirements](https://gitee.com/mindspore/golden-stick/blob/master/requirements.txt) for other third party dependencies.
> MindSpore Golden Stick can only run on Ubuntu18.04.

## Version Dependency

The MindSpore Golden Stick depends on the MindSpore training and inference framework, please refer to the table below and [MindSpore Installation Guide](https://mindspore.cn/install) to install the corresponding MindSpore verision:

| MindSpore Golden Stick Version |                         Branch                               | MindSpore Version |
| :---------------------: | :-----------------------------------------------------------------: | :-------: |
|          1.1.0          | [r1.1](https://gitee.com/mindspore/golden-stick/tree/r1.1.0/)       |   2.6.0   |
|          1.0.0          | [r1.0](https://gitee.com/mindspore/golden-stick/tree/r1.0.0/)       |   2.5.0   |
|          0.6.0          | [r0.6](https://gitee.com/mindspore/golden-stick/tree/r0.6.0/)       |   2.4.0   |
|          0.5.0          | [r0.5](https://gitee.com/mindspore/golden-stick/tree/r0.5.0/)       |   2.3.1   |
|          0.4.1          | [r0.4](https://gitee.com/mindspore/golden-stick/tree/r0.4.1/)       |   2.3.0   |
|          0.4.0          | [r0.4](https://gitee.com/mindspore/golden-stick/tree/r0.4/)         |   2.3.0-rc1   |
|          0.3.0          | [r0.3](https://gitee.com/mindspore/golden-stick/tree/r0.3/)         |   2.0.0-rc1, 2.0.0   |
|       0.3.0-alpha       | [r0.3](https://gitee.com/mindspore/golden-stick/tree/v0.3.0-alpha/) |   2.0.0-alpha   |
|          0.2.0          | [r0.2](https://gitee.com/mindspore/golden-stick/tree/r0.2/)         |   1.9.0   |
|          0.1.0          | [r0.1](https://gitee.com/mindspore/golden-stick/tree/r0.1/)         |   1.8.0   |

After MindSpore is installed, you can use pip or source code build for MindSpore Golden Stick installation.

## Installing from pip Command

If you use the pip command, please download the whl package from [MindSpore Golden Stick](https://www.mindspore.cn/versions/en) page and install it.

```shell
pip install  https://ms-release.obs.cn-north-4.myhuaweicloud.com/{MindSpore_version}/GoldenStick/any/mindspore_gs-{mg_version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - Installing whl package will download MindSpore Golden Stick dependencies automatically (detail of dependencies is shown in requirement.txt),  other dependencies should install manually.
> - `{MindSpore_version}` stands for the version of MindSpore. For the version matching relationship between MindSpore and MindSpore Golden Stick, please refer to [page](https://www.mindspore.cn/versions).
> - `{ms_version}` stands for the version of MindSpore Golden Stick. For example, if you would like to download version 1.0.0, you should fill 2.5.0 in `{MindSpore_version}` and fill 1.0.0 in `{mg_version}`.

## Installing from Source Code

Download [source code](https://gitee.com/mindspore/golden-stick), then enter the `golden-stick` directory.

```shell
git clone https://gitee.com/mindspore/golden-stick.git
cd golden-stick
bash build.sh
pip install output/mindspore_gs-{mg_version}-py3-none-any.whl
```

## Verification

If you can successfully execute following command, then the installation is completed.

```python
import mindspore_gs
```
