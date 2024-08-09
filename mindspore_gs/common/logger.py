# Copyright 2024 Huawei Technologies Co., Ltd
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
""" Logger for golden-stick """

import os
import sys
import traceback
import io
import logging


if hasattr(sys, '_getframe'):
    # pylint: disable=protected-access
    currentframe = lambda: sys._getframe(4)
else: #pragma: no cover
    def currentframe():
        """Return the frame object for the caller's stack frame."""
        try:
            raise Exception
        # pylint: disable=broad-except
        except Exception:
            return sys.exc_info()[2].tb_frame.f_back


# pylint: disable=unused-argument
def _find_real_caller(stack_info=False, stacklevel=1):
    """
    Find the stack frame of the caller so that we can note the source
    file name, line number and function name.
    """
    f = currentframe()
    log_file = os.path.normcase(f.f_code.co_filename)
    if f is not None:
        f = f.f_back
    rv = "(unknown file)", 0, "(unknown function)", None
    while hasattr(f, "f_code"):
        co = f.f_code
        filename = os.path.normcase(co.co_filename)
        if filename == log_file:
            f = f.f_back
            continue
        sinfo = None
        if stack_info:
            sio = io.StringIO()
            sio.write('Stack (most recent call last):\n')
            traceback.print_stack(f, file=sio)
            sinfo = sio.getvalue()
            if sinfo[-1] == '\n':
                sinfo = sinfo[:-1]
            sio.close()
        rv = (co.co_filename, f.f_lineno, co.co_name, sinfo)
        break
    return rv


class GSLogFormatter(logging.Formatter):
    """_GSLogFormatter"""
    def format(self, record):
        """format"""
        # handle pathname
        ms_install_home_path = 'mindspore_gs'
        idx = record.pathname.rfind(ms_install_home_path)
        if idx >= 0:
            # Get the relative path of the file
            record.filepath = record.pathname[idx:]
        else:
            record.filepath = record.pathname

        # handle funcName
        if record.funcName == "<module>":
            record.funcname = ""
        else:
            record.funcname = record.funcName

        return super().format(record)


class Logger:
    """Logger for GoldenStick."""
    def __init__(self):
        self.logger = logging.getLogger("GoldenStick")
        self.logger.findCaller = _find_real_caller
        if not self.logger.hasHandlers():
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(level=logging.INFO)
            format_str = '[%(levelname)s] %(name)s(%(process)s):%(asctime)s [%(filepath)s:%(lineno)d %(funcname)s] - ' \
                         '%(message)s'
            console.setFormatter(GSLogFormatter(format_str))
            self.logger.addHandler(console)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def set_level(self, level):
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)

    def debug(self, *args, **kwargs):
        """Add debug level log."""
        self.logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        """Add info level log."""
        self.logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        """Add warning level log."""
        self.logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        """Add error level log."""
        self.logger.error(*args, **kwargs)


logger = Logger()
