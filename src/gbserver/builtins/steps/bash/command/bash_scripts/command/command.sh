#!/bin/sh

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

# Run an arbitrary command supplied via the build.yaml step config
# (config.bash.env.LLMB_COMMAND). The step's exit status is the command's exit
# status, so a failing command (e.g. "exit 1") hard-FAILS the target.
echo "command step start: ${LLMB_COMMAND:-<no LLMB_COMMAND set>}"
exec sh -c "${LLMB_COMMAND}"
