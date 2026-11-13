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

"""Clone progress reporter, kept apart from ``utils`` so gitpython loads only on clone."""

from git import RemoteProgress
from tqdm import tqdm


class CloneProgress(RemoteProgress):
    def __init__(self, update_bar):
        super().__init__()
        self.update_bar = update_bar
        if not update_bar:
            self.pbar = tqdm(leave=False)

    def update(self, op_code, cur_count, max_count=None, message=""):
        if self.update_bar:
            # convert to step size for specified total
            step_size = 100 / max_count
            self.update_bar(
                callback_event="preparing_contents", callback_args={"steps": step_size}
            )
        else:
            self.pbar.total = max_count
            self.pbar.n = cur_count
            self.pbar.refresh()
