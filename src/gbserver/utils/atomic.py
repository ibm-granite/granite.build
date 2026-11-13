# Copyright Granite.Build Authors
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

import abc
import threading


class AtomicInteger(abc.ABC):
    _counter: int
    _lock: threading.Lock

    def __init__(self, initial_value: int = 0):
        self._counter = initial_value
        self._lock = threading.Lock()

    def fetch_and_add(self) -> int:
        """Get the current value and then increment the internal value"""
        with self._lock:  # Acquire the lock before modifying counter
            value = self._counter
            self._counter += 1
        return value

    def get(self) -> int:
        """Atomically read the current value."""
        with self._lock:
            return self._counter
