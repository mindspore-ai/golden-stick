#!/usr/bin/env python
# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test scheduler for running multiple tests in parallel or sequentially."""

import os
import time
import subprocess
import threading
import argparse
import tempfile
from tests.st.test_utils import get_available_port


class Task:
    """Task class for encapsulating test information and operations."""

    def __init__(self, name, script_path, num_cards, device_str=None, log_dir=None):
        """Initialize a test task.

        Args:
            name: Name of the test for logging purposes
            script_path: Path to the test script to run
            num_cards: Number of cards to use for this test
            device_str: Device string to use for this test
            log_dir: Directory for storing logs (optional)
        """
        self.name = name
        self.script_path = script_path
        self.num_cards = num_cards
        self.device_str = device_str
        self.log_dir = log_dir or f"./test_{name}_logs"
        self.log_file = f"{name.lower().replace(' ', '_')}_test.log"
        self.return_code = None
        self.start_time = None
        self.end_time = None
        self.port = None

    def prepare_environment(self):
        """Prepare environment variables and ports for the test."""
        # Set environment variables
        os.environ['HCCL_CONNECT_TIMEOUT'] = "1800"
        with open(self.log_file, 'w') as log_file:
            # Set HCCL_NPU_SOCKER_PORT_RANGE environment variable if not already set
            # Use test name hash to create a unique port range for each test to avoid conflicts
            port_offset = hash(self.name) % 1000
            port_start = 60000 + port_offset
            port_end = port_start + 5
            os.environ['HCCL_HOST_SOCKET_PORT_RANGE'] = f"{port_start}-{port_end}"
            log_file.write(f"[{self.name}] Setting HCCL_HOST_SOCKET_PORT_RANGE={port_start}-{port_end}")
            port_start += 50
            port_end += 50
            os.environ['HCCL_NPU_SOCKET_PORT_RANGE'] = f"{port_start}-{port_end}"
            log_file.write(f"[{self.name}] Setting HCCL_NPU_SOCKET_PORT_RANGE={port_start}-{port_end}")

            port_start = 20000 + port_offset
            port_end = port_start + 5
            lcal_port = get_available_port(port_start, port_end)
            os.environ['LCAL_COMM_ID'] = f"127.0.0.1:{lcal_port}"
            log_file.write(f"[{self.name}] Setting LCAL_COMM_ID=127.0.0.1:{lcal_port}")

            # Get a free port for the distributed training
            # Use test name hash to create a unique port range for each test to avoid conflicts
            port_offset = hash(self.name) % 1000
            port_start = 10000 + port_offset
            port_end = port_start + 5
            self.port = get_available_port(port_start, port_end)
            log_file.write(f"[{self.name}] get acailable port {self.port}")
            os.system(f"kill -9 $(lsof -i:{self.port} | " + "awk '{print $2}')")
            time.sleep(1.0)

            # Set device IDs if provided
            if self.device_str:
                os.environ['ASCEND_RT_VISIBLE_DEVICES'] = self.device_str
                log_file.write(f"[{self.name}] Setting ASCEND_RT_VISIBLE_DEVICES={self.device_str}")
            os.makedirs(self.log_dir, exist_ok=True)
            log_file.flush()

    def build_command(self):
        """Build the command to run the test."""
        # Construct the msrun command
        # Check if script_path already contains command line arguments
        if " " in self.script_path:
            script_parts = self.script_path.split(" ", 1)
            script_file = script_parts[0]
            script_args = script_parts[1]
            command = (
                f"msrun --worker_num={self.num_cards} --local_worker_num={self.num_cards} "
                f"--master_addr=127.0.0.1 --master_port={self.port} --join=True "
                f"--log_dir={self.log_dir} python {script_file} {script_args}"
            )
        else:
            command = (
                f"msrun --worker_num={self.num_cards} --local_worker_num={self.num_cards} "
                f"--master_addr=127.0.0.1 --master_port={self.port} --join=True "
                f"--log_dir={self.log_dir} python {self.script_path}"
            )

        return command

    def print_task_info(self):
        """Print task information."""
        device_info = f"devices {self.device_str}"
        print(f"\n[{self.name}] STARTING TEST on {device_info}")
        print(f"[{self.name}] Log file: {os.path.abspath(self.log_file)}")
        print(f"[{self.name}] Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[{self.name}] Using port: {self.port}")
        print(f"[{self.name}] Running on devices: {os.environ.get('ASCEND_RT_VISIBLE_DEVICES', 'unknown')}")
        print(f"[{self.name}] Script path: {self.script_path}")
        print(f"[{self.name}] Number of cards: {self.num_cards}")

    def write_log_header(self, command):
        """Write header information to log file."""
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else '.', exist_ok=True)

        with open(self.log_file, 'a') as log_file:
            log_file.write(f"=== TEST: {self.name} ===\n")
            log_file.write(f"Command: {command}\n")
            log_file.write(f"Devices: {self.device_str}\n")
            log_file.write(f"Port: {self.port}\n")
            log_file.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"===========================\n\n")
            log_file.flush()

    def write_log_footer(self, status):
        """Write footer information to log file."""
        with open(self.log_file, 'a') as log_file:
            log_file.write(f"\n===========================\n")
            log_file.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
            hours, remainder = divmod(duration, 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
            log_file.write(f"Duration: {duration_str}\n")
            log_file.write(f"Status: {status}\n")
            log_file.flush()

    def print_completion_info(self):
        """Print completion information."""
        status = "PASSED" if self.return_code == 0 else f"FAILED (code: {self.return_code})"
        print(f"\n[{self.name}] TEST {status}")
        print(f"[{self.name}] End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            hours, remainder = divmod(duration, 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
            print(f"[{self.name}] Duration: {duration_str}")

    def run(self):
        """Run the test and return the result code."""
        self.start_time = time.time()

        # Prepare environment and build command
        self.prepare_environment()
        command = self.build_command()

        # Print task information
        self.print_task_info()
        print(f"[{self.name}] Executing command: {command}")

        # Write log header
        self.write_log_header(command)

        # Run the command
        with open(self.log_file, 'a') as log_file:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # Monitor the process output
            for line in process.stdout:
                # Add task name prefix to each line for better identification in parallel execution
                log_file.write(line)
                log_file.flush()

            # Wait for the process to complete
            self.return_code = process.wait()
            self.end_time = time.time()
            # Log completion information
            status = "PASSED" if self.return_code == 0 else f"FAILED (code: {self.return_code})"
            self.print_completion_info()
            self.write_log_footer(status)
            return self.return_code


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--task_name', type=str, required=True)
    args.add_argument('--script_path', type=str, required=True)
    args.add_argument('--num_cards', type=int, required=True)
    args.add_argument('--device_str', type=str, required=True)
    args.add_argument('--log_dir', type=str, required=True)
    args.add_argument('--tmp_file', type=str, required=True)
    args = args.parse_args()
    with open(args.tmp_file, 'w') as result_file:
        current_task = Task(args.task_name, args.script_path, args.num_cards, args.device_str, args.log_dir)
        print(f'Process for task {current_task.name} started')
        exit_code = current_task.run()
        print(f'Task {current_task.name} completed with code: {{exit_code}}')
        result_file.write(str(exit_code))


class TestScheduler:
    """Scheduler for running multiple tests with device allocation."""

    def __init__(self, total_available_devices=None):
        """Initialize the test scheduler.

        Args:
            total_available_devices: List of available device IDs (default: [0,1,2,3,4,5,6,7])
        """
        self.total_available_devices = total_available_devices or list(range(8))  # Default: 8 cards (0-7)
        self.tasks = []

    def add_test(self, test_name, script_path, num_cards, log_dir=None):
        """Add a test to the scheduler.

        Args:
            test_name: Name of the test
            script_path: Path to the test script
            num_cards: Number of cards required for this test
            log_dir: Directory for storing logs (optional)
        """
        self.tasks.append(Task(test_name, script_path, num_cards, log_dir=log_dir))

    def run_all_parallel(self):
        """Run all tests in parallel with automatic device allocation."""
        print("\n===== Starting Test Scheduler in Parallel Mode =====")

        # Check if we have enough devices
        total_cards_needed = sum(task.num_cards for task in self.tasks)
        print(f"Available devices: {self.total_available_devices}")
        print(f"Total cards needed: {total_cards_needed}, Available: {len(self.total_available_devices)}")

        if total_cards_needed > len(self.total_available_devices):
            raise ValueError(
                f"Not enough devices. Need {total_cards_needed} but only "
                f"{len(self.total_available_devices)} available."
            )

        # Allocate devices to tasks
        allocated_devices = []

        print("\n===== Device Allocation =====")
        for task_item in self.tasks:
            # Allocate the next available devices
            device_ids = []
            for _ in range(task_item.num_cards):
                for device_id in self.total_available_devices:
                    if device_id not in allocated_devices:
                        device_ids.append(device_id)
                        allocated_devices.append(device_id)
                        break

            device_str = ','.join(map(str, device_ids))
            task_item.device_str = device_str

        # Print summary of all tasks before starting
        print("\n===== Task Summary =====")
        for i, task_item in enumerate(self.tasks):
            print(f"Task {i+1}: '{task_item.name}'")
            print(f"  - Script: {task_item.script_path}")
            print(f"  - Devices: {task_item.device_str}")
            print(f"  - Cards: {task_item.num_cards}")
            print(f"  - Log directory: {task_item.log_dir}")

        # Run all tasks in parallel
        processes = []
        result_files = []

        print("\n===== Starting Test Processes =====")
        for i, task_item in enumerate(self.tasks):
            # Create a temporary file to store the return code
            fd, result_path = tempfile.mkstemp(prefix=f"test_result_{i}_", suffix=".txt")
            os.close(fd)  # Close the file descriptor
            result_files.append(result_path)

            # Start the process
            process = self._start_task_process(task_item, result_path)
            processes.append(process)
            print(f"Started process for task '{task_item.name}'")

        # Wait for all processes to complete
        print("\n===== Waiting for All Test Processes to Complete =====")
        return_codes = self._wait_for_processes(processes, result_files)

        # Check for failures and print summary
        return self._print_summary(return_codes)

    def _start_task_process(self, task_obj, result_path):
        """Start a process to run a task."""
        # 构建命令字符串而不是列表，确保参数正确传递
        cmd = f"python {__file__} \
            --task_name \"{task_obj.name}\" \
            --script_path \"{task_obj.script_path}\" \
            --num_cards {task_obj.num_cards} \
            --device_str \"{task_obj.device_str}\" \
            --log_dir \"{task_obj.log_dir}\" \
            --tmp_file '{result_path}'"

        print(f"Executing command: {cmd}")

        # 使用shell=True确保命令字符串被正确解析
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            shell=True
        )

        return process

    def _wait_for_processes(self, processes, result_files):
        """Wait for all processes to complete and collect return codes."""
        return_codes = []

        for i, process in enumerate(processes):
            # Monitor process output and forward it to console
            # Use a separate thread to read output to avoid deadlocks
            def read_output(proc, task_name):
                for line in proc.stdout:
                    print(f"[{task_name}] {line.strip()}")

            output_thread = threading.Thread(
                target=read_output,
                args=(process, self.tasks[i].name)
            )
            output_thread.daemon = True
            output_thread.start()

            # Wait for process to complete
            process.wait()
            output_thread.join(timeout=1.0)  # Give the output thread a chance to finish
            print(f"Process for '{self.tasks[i].name}' completed with return code {process.returncode}")

            # Read the return code from the temporary file
            try:
                with open(result_files[i], 'r') as result_fp:
                    return_code = int(result_fp.read().strip())
                return_codes.append(return_code)
                # Clean up the temporary file
                try:
                    os.remove(result_files[i])
                except OSError as e:
                    print(f"Warning: Could not remove temporary file {result_files[i]}: {e}")
            except (FileNotFoundError, ValueError) as e:
                print(f"Error reading result for '{self.tasks[i].name}': {e}")
                # If the file doesn't exist, the process might have failed to create it
                # Use the process return code as a fallback
                process_return_code = process.returncode
                print(f"Using process return code as fallback: {process_return_code}")
                return_codes.append(process_return_code if process_return_code is not None else 1)
        return return_codes

    def _print_summary(self, return_codes):
        """Print execution summary and return overall status."""
        # Check for failures
        failures = []
        passed_count = 0
        failed_count = 0

        for i, code in enumerate(return_codes):
            if code != 0:
                failures.append(self.tasks[i].name)
                failed_count += 1
            else:
                passed_count += 1

        # Print execution summary
        print("\n===== Test Execution Summary =====")
        for i, task_item in enumerate(self.tasks):
            status = "PASSED" if return_codes[i] == 0 else f"FAILED (code: {return_codes[i]})"
            print(f"Task {i+1}: '{task_item.name}'")
            print(f"  - Status: {status}")
            print(f"  - Script: {task_item.script_path}")
            print(f"  - Devices: {task_item.device_str}")
            print(f"  - Cards: {task_item.num_cards}")
            print(f"  - Log file: {task_item.log_file}")
            print()

        print("\n===== Overall Result =====")
        print(f"Total tests: {len(return_codes)}")
        print(f"Passed: {passed_count}")
        print(f"Failed: {failed_count}")

        if failures:
            failure_str = ", ".join(failures)
            print(f"\nTest failures: {failure_str}")
        print("\nAll tests completed successfully")
        return failures

    def run_all_sequential(self):
        """Run all tests sequentially using all available devices for each test."""
        for task_item in self.tasks:
            # Use all available devices up to the number needed
            task_item.device_ids = self.total_available_devices[:task_item.num_cards]

            return_code = task_item.run()
            if return_code != 0:
                print(f"Task {task_item.name} failed with return code {return_code}")
                return 1

        print("All tests completed successfully")
        return 0


def run_combined_tests(test_configs):
    """Run combined tests using the test scheduler.

    Args:
        test_configs: List of test configurations, each containing:
            - name: Test name
            - script: Path to test script
            - num_cards: Number of cards required
            - log_dir: Directory for logs

    Returns:
        int: Return code (0 for success, non-zero for failure)
    """
    # Create scheduler with all available devices
    scheduler = TestScheduler()

    # Add all tests to the scheduler
    for config in test_configs:
        scheduler.add_test(
            config['name'],
            config['script'],
            config['num_cards'],
            config['log_dir']
        )

    # Run all tests in parallel
    return scheduler.run_all_parallel()
