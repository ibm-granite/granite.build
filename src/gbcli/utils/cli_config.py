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

import os
from pathlib import Path


def configureGBWorkingEnv():
    if "GB_CONFIG" not in os.environ:
        os.environ["GB_CONFIG"] = os.path.join(os.path.expanduser("~"), ".gbcli")

    if "DMF_CACHE" not in os.environ:
        os.environ["DMF_CACHE"] = os.path.join(os.environ["GB_CONFIG"], "dmf_cache")

    if "GB_CACHE" not in os.environ:
        os.environ["GB_CACHE"] = os.path.join(os.environ["GB_CONFIG"], "workdir")


def get_local_gb_config():
    return Path(os.path.abspath(os.environ["GB_CONFIG"]))


def get_local_build_cache():
    return Path(os.path.abspath(os.environ["GB_CACHE"]))
