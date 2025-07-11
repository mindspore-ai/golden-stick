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
import logging
import inspect
import threading
import atexit


def _find_real_caller():
    """
    Find the stack frame of the caller so that we can note the source
    file name, line number and function name.
    """
    current_file = os.path.normcase(__file__)
    stack = inspect.stack()
    for frame_info in stack:
        frame = frame_info.frame
        filename = os.path.normcase(frame.f_code.co_filename)
        if filename == current_file:
            continue
        return (
            filename,
            frame.f_lineno,
            frame.f_code.co_name,
            None
        )
    return "unknown", 0, "unknown", None


class GSLogFormatter(logging.Formatter):
    """_GSLogFormatter"""
    def format(self, record):
        idx = record.pathname.rfind('mindspore_gs')
        record.filepath = record.pathname[idx:] if idx >= 0 else record.pathname
        record.funcname = record.funcName if record.funcName != "<module>" else ""
        return super().format(record)


class Logger:
    """Logger for GoldenStick."""
    def __init__(self):
        self.logger = logging.getLogger("GoldenStick")
        log_level = self._get_init_log_level()

        # Initialize handlers if not exist
        if not self.logger.hasHandlers():
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(log_level)
            formatter = GSLogFormatter(
                '[%(levelname)s] %(name)s(%(process)s):%(asctime)s '
                '[%(filepath)s:%(lineno)d %(funcname)s] - %(message)s'
            )
            console.setFormatter(formatter)
            self.logger.addHandler(console)

        self.logger.setLevel(log_level)
        self.logger.propagate = False

        # Log merging mechanism
        self._lock = threading.RLock()
        self.last_signature = None  # (msg, level, file, line, func)
        self.repeat_count = 0
        self.last_exc_info = None
        self.last_extra = None
        self.last_sinfo = None
        self.last_msg = None
        self.last_level = None
        self.first_occurrence = None

        # Ensure final flush on exit
        atexit.register(self._flush)


    @staticmethod
    def _get_init_log_level():
        level_map = {
            '0': logging.DEBUG,
            '1': logging.INFO,
            '2': logging.WARNING,
            '3': logging.ERROR
        }
        return level_map.get(os.environ.get("GSLOG", "2"), logging.WARNING)

    def set_level(self, level):
        """set_level"""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)

    def _flush(self):
        """_flush"""
        if self.last_msg is None:
            return

        # use first log location
        filename, lineno, funcname = self.first_occurrence
        merged_msg = f"{self.last_msg} (repeated {self.repeat_count} times)" if self.repeat_count > 1 else self.last_msg

        record = self.logger.makeRecord(
            name=self.logger.name,
            level=self.last_level,
            fn=filename,
            lno=lineno,
            msg=merged_msg,
            args=(),
            exc_info=self.last_exc_info,
            func=funcname,
            sinfo=self.last_sinfo,
            extra=self.last_extra
        )
        self.logger.handle(record)

        # reset state
        self.last_msg = None
        self.last_level = None
        self.first_occurrence = None
        self.repeat_count = 0

    def _log(self, level, msg, *args, **kwargs):
        """_log"""
        with self._lock:
            filename, lineno, funcname, sinfo = _find_real_caller()

            # if same log, increment repeat_count
            if msg == self.last_msg and level == self.last_level:
                self.repeat_count += 1
                # save first raise info
                if 'exc_info' in kwargs and kwargs['exc_info'] and not self.last_exc_info:
                    self.last_exc_info = kwargs.get('exc_info')
                return
            # if not same log, output buffered log
            self._flush()

            # record new msg info
            self.repeat_count = 1
            self.last_msg = msg
            self.last_level = level

            self.first_occurrence = (filename, lineno, funcname)

            self.last_exc_info = kwargs.get('exc_info')
            self.last_extra = kwargs.get('extra')
            self.last_sinfo = sinfo

    # Public logging methods
    def debug(self, msg, *args, **kwargs):
        """debug"""
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """info"""
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """warning"""
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """error"""
        self._log(logging.ERROR, msg, *args, **kwargs)


logger = Logger()
