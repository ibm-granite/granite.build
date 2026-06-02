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

"""Utility functions."""

import base64
import hashlib
import json
import random
import re
import string
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from uuid import uuid4

import yaml

from gbserver.types.localsecretsconfig import SpacesConfig
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


import requests

from gbserver.types.constants import GB_ENVIRONMENT_CONFIG


def get_uuid() -> str:
    """Return a new UUID."""
    return str(uuid4())


def get_sha256sum(my_input: Union[str, bytes]) -> str:
    """Returns the SHA 256 hash of the inputs as hex digits."""
    input_bytes = my_input.encode("utf-8") if isinstance(my_input, str) else my_input
    hash_object = hashlib.sha256(input_bytes)
    return hash_object.hexdigest()


def get_time() -> datetime:
    """Return the current local time (timezone aware)."""
    return datetime.now().astimezone()


def get_utc_time() -> datetime:
    """Return a datetime in utc timezone. UTC seems to be reuired by Iceberg."""
    t1 = datetime.now(timezone.utc)
    return t1


def normalize_to_filename(value: str, allow_unicode: bool = False) -> str:
    """
    https://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename

    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    assert isinstance(value, str)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def short_alphanumeric_lower_hash(input_string: str) -> str:
    """Returns a short (8 character) hash of the input string."""
    hash_object = hashlib.sha256(input_string.encode("utf-8"))
    base64_encoded = base64.b64encode(hash_object.digest()).decode("utf-8")
    base64_encoded = "".join(c for c in base64_encoded if c.isalnum())
    return base64_encoded[:8].lower()


def cmd_safe_join(cmd: List[str]) -> str:
    """Preserves arguments with spaces in them."""
    return " ".join(f"'{x}'" if " " in x else x for x in cmd)


def random_string(length: int = 8):
    """Returns a random string of the given length."""
    characters = string.ascii_lowercase + string.digits
    return "".join(random.choice(characters) for i in range(length))


def get_build_status_link(build_id: str) -> str:
    """Get the link to the build status/lineage given the build ID."""
    dmf_ui = GB_ENVIRONMENT_CONFIG.dmf_ui
    if not dmf_ui:
        return "(lineage is not available in standalone mode)"
    return f"{dmf_ui}/gb/builds/{build_id}"


def get_dashboard_link(build_id: str) -> str:
    """Get the link to the dashboard given the build ID."""
    dashboard_instance = GB_ENVIRONMENT_CONFIG.dashboard_instance
    if not dashboard_instance:
        return "(dashboard is not available in standalone mode)"
    return f"{dashboard_instance}/dashboard/build/{build_id}"


def download_file(url: str, output_path: Path, timeout: int = 10 * 60) -> None:
    """
    Download a file from the given URL.
    https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
    """
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                # if chunk:
                f.write(chunk)


def get_common_ancestor(ps: List[Path]) -> Path:
    """Get the longest common ancestor/parent directory."""
    assert len(ps) > 0
    if len(ps) == 1:
        return ps[0]
    p_strs = [str(p) for p in ps]
    sorted_p_strs = sorted(p_strs)
    first = sorted_p_strs[0]
    last = sorted_p_strs[-1]
    smaller = first if len(first) <= len(last) else last
    until = len(smaller)
    i = 0
    while i < until:
        if first[i] != last[i]:
            break
        i += 1
    answer = smaller[:i]
    return Path(answer)


def create_temp_file_name(suffix=""):
    """Create a temporary file name."""
    with tempfile.NamedTemporaryFile(
        delete=False, dir="/tmp", suffix=suffix
    ) as temp_file:
        temp_file_name = temp_file.name
    return temp_file_name


def get_differing_attributes(object1: Any, object2: Any) -> Dict[str, Tuple[Any, Any]]:
    """
    Does the following:
    - Find differing attributes present in both objects
    - Find attributes only in obj1
    - Find attributes only in obj2

        and returns a dictionary: attr_name -> (value1, value2)
    """
    differing_attributes = {}
    attributes1 = vars(object1)
    attributes2 = vars(object2)
    assert isinstance(attributes1, dict), f"invalid attributes1: {attributes1}"
    assert isinstance(attributes2, dict), f"invalid attributes2: {attributes2}"

    # Find differing attributes present in both objects
    for attr_name, value1 in attributes1.items():
        assert isinstance(attr_name, str), f"invalid attr_name: {attr_name}"
        if attr_name in attributes2:
            value2 = attributes2[attr_name]
            if value1 != value2:
                differing_attributes[attr_name] = (value1, value2)

    # Find attributes only in obj1
    for attr_name in attributes1:
        assert isinstance(attr_name, str), f"invalid attr_name: {attr_name}"
        if attr_name not in attributes2:
            differing_attributes[attr_name] = (
                attributes1[attr_name],
                "Attribute not in second object",
            )

    # Find attributes only in obj2
    for attr_name in attributes2:
        assert isinstance(attr_name, str), f"invalid attr_name: {attr_name}"
        if attr_name not in attributes1:
            differing_attributes[attr_name] = (
                "Attribute not in first object",
                attributes2[attr_name],
            )
    return differing_attributes


def write_local_secrets_file(
    *,
    secrets_file: Path,
    space_name: str,
    secrets: Dict[str, Dict[str, Any]],
):
    """
    Create or update a local secrets file in the format:
    spaces:
      <space>:
        secrets:
          NAME:
            payload: <base64>
            labels: [...]
            secret_group: <group>
    """

    secrets_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing file if present
    if secrets_file.exists():
        with open(secrets_file, "r", encoding="utf-8") as f:
            if secrets_file.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(f) or {}
            elif secrets_file.suffix.lower() == ".json":
                data = json.load(f) or {}
            else:
                data = {}
    else:
        data = {}

    spaces = data.setdefault("spaces", {})
    space = spaces.setdefault(space_name, {})
    space_secrets = space.setdefault("secrets", {})

    for name, secret in secrets.items():
        assert "payload" in secret, f"Missing payload for secret '{name}'"

        raw_payload = secret["payload"]
        payload = base64.b64encode(raw_payload.encode("utf-8")).decode("utf-8")

        entry = {"payload": payload}

        if "labels" in secret and secret["labels"] is not None:
            entry["labels"] = secret["labels"]

        if "secret_group" in secret and secret["secret_group"] is not None:
            entry["secret_group"] = secret["secret_group"]

        space_secrets[name] = entry

    # validating using model validate with the data
    SpacesConfig.model_validate(data)

    with open(secrets_file, "w", encoding="utf-8") as f:
        if secrets_file.suffix.lower() in [".yaml", ".yml"]:
            yaml.safe_dump(data, f, sort_keys=False)
        elif secrets_file.suffix.lower() == ".json":
            json.dump(data, f, indent=2)
