# coding=utf-8
# Copyright 2023-present International Business Machines Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class PrintLogger:
    def __init__(self, logger):
        self.logger = logger
        self.buffer = ""

    def write(self, message):
        # Add message to buffer
        self.buffer += message

        # Process complete lines in buffer
        if "\n" in self.buffer:
            lines = self.buffer.split("\n")
            # Keep the last piece if it doesn't end with newline
            self.buffer = lines.pop()

            # Log each complete line
            # stacklevel=2 skips past this write() frame so the logging
            # framework attributes the record to the actual caller instead
            # of print_logger.py.
            for line in lines:
                if line.strip():
                    self.logger.info(line.strip(), stacklevel=2)

    def flush(self):
        # Log any remaining content in buffer when flush is called
        if self.buffer.strip():
            self.logger.info(self.buffer.strip(), stacklevel=2)
            self.buffer = ""

    def isatty(self):
        return False
