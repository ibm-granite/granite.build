#!/usr/bin/env python3

# Copyright LLM.build Authors
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

"""Utility functions for Kubernetes related stuff."""

import re
import subprocess

SEMVER_REGEX = re.compile(
    r"""^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"""
)


def get_helm_version() -> str:
    """
    Returns the installed helm version.
    Example: v4.1.1+g5caf004
    """

    try:
        proc = subprocess.run(
            ["helm", "version", "--short"], check=True, capture_output=True, text=True
        )
        return proc.stdout
    except Exception as e:
        print("failed to get helm version:", e)
        return ""


def is_helm_v4_or_higher(version: str = "") -> bool:
    """Returns True if the installed helm version is >= 4"""
    if not version:
        version = get_helm_version()
    if not version:
        return False
    version = version.removeprefix("v")
    match = SEMVER_REGEX.match(version)
    if not match:
        return False
    major_version = int(match.group(1), base=10)
    return major_version >= 4
