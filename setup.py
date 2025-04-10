#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""setup package."""
import os
import sys
import re
import stat
import shutil
import time
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py


pwd = os.path.dirname(os.path.realpath(__file__))
build_path = os.path.join(pwd, 'build')
pkg_dir = os.path.join(build_path, 'lib')


class BuildPy(build_py):
    """BuildPy."""

    @staticmethod
    def _write_commit_file(path):
        """ Write commit info into `file`. """
        ret_code = os.system(f"git log --format='[sha1]:%h,[branch]:%d' --abbrev=8 -1 > {path}")
        if ret_code != 0:
            sys.stdout.write("Warning: Can not get commit id information. Please make sure git is available.")
            os.system(f"echo 'git is not available while building.' > {path}")

    @staticmethod
    def _write_extra_info():
        """ Write extra info into `file` such as commit info. """
        mindspore_dir = os.path.join(pkg_dir, 'mindspore_gs')
        BuildPy._update_permissions(mindspore_dir)
        mindspore_dir = os.path.join(pwd, 'mindspore_gs')
        BuildPy._update_permissions(mindspore_dir)
        mindspore_dir = os.path.join(pkg_dir, 'mindspore_gs', '.commit_id')
        BuildPy._update_permissions(mindspore_dir)
        BuildPy._write_commit_file(mindspore_dir)
        mindspore_dir = os.path.join(pwd, 'mindspore_gs', '.commit_id')
        BuildPy._update_permissions(mindspore_dir)
        BuildPy._write_commit_file(mindspore_dir)

    @staticmethod
    def _update_permissions(path):
        """
        Update permissions of `path`.

        Args:
            path (str): Target directory path.
        """

        for dirpath, dirnames, filenames in os.walk(path):
            for dirname in dirnames:
                dir_fullpath = os.path.join(dirpath, dirname)
                os.chmod(dir_fullpath, stat.S_IREAD | stat.S_IWRITE |
                         stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)
            for filename in filenames:
                file_fullpath = os.path.join(dirpath, filename)
                os.chmod(file_fullpath, stat.S_IREAD | stat.S_IWRITE)

    @staticmethod
    def _copy_ops_files():
        """Copy ops files to pkg."""
        src_path = os.path.join(pwd, 'mindspore_gs')
        target_dir_name = 'gpu'
        for dirpath, dirnames, _ in os.walk(src_path):
            if target_dir_name in dirnames:
                src_dir_path = os.path.join(dirpath, target_dir_name)
                dst_dir_path = os.path.join(pkg_dir, 'mindspore_gs',
                                            dirpath.split('mindspore_gs/')[-1], target_dir_name)
                if os.path.exists(dst_dir_path):
                    shutil.rmtree(dst_dir_path)
                shutil.copytree(src_dir_path, dst_dir_path)

    def run(self):
        super().run()
        BuildPy._copy_ops_files()
        BuildPy._write_extra_info()


def version_from_file():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    version_py_file = os.path.join(current_dir, 'mindspore_gs', 'version.py')
    version_pattern = r"^__version__.*=.*'([0-9,.]+)'"
    with open(version_py_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(version_pattern, line.strip())
            if match:
                return match.group(1)
    raise ValueError(f"Version not found in {version_py_file}")


version_num = version_from_file()
date_str = time.strftime('%Y%m%d', time.localtime())
version = f"{version_num}.dev{date_str}"


setup(
    name='mindspore_gs',
    version=version,
    author='The MindSpore Authors',
    author_email='contact@mindspore.cn',
    description='A MindSpore model optimization algorithm set..',
    url='https://www.mindspore.cn',
    packages=find_packages(include=['mindspore_gs*']),
    cmdclass={'build_py': BuildPy},
    install_requires=[
        'numpy >= 1.17.0',
        'scipy >= 1.5.2',
        'pyyaml>=6.0',
        'matplotlib',
        'prettytable',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.7',
    license='Apache 2.0',
    keywords='mindspore optimization quantization golden-stick',
)
