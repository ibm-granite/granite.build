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

import base64
import hashlib
import random
import re
import string
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from uuid import uuid4

# from gbcommon.types.constants import (
#     FETCH_CLOUD_LOGS_MAX_RETRIES,
#     GB_ENVIRONMENT_CONFIG,
# )
# from gbcommon.types.errors import LogMonitoringFailedException
# from gbcommon.utils.cloud_logquery import log_manager
# from gbcommon.utils.logger import get_logger
# import requests


# logger = get_logger(__name__)


# def get_uuid() -> str:
#     """Return a new UUID."""
#     return str(uuid4())


# def get_time() -> datetime:
#     """Return the current local time (timezone aware)."""
#     return datetime.now().astimezone()


# def get_utc_time() -> datetime:
#     """Return a datetime in utc timezone. UTC seems to be reuired by Iceberg."""
#     t1 = datetime.now(timezone.utc)
#     return t1


# def normalize_to_filename(value: str, allow_unicode: bool = False) -> str:
#     """
#     https://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename

#     Taken from https://github.com/django/django/blob/master/django/utils/text.py
#     Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
#     dashes to single dashes. Remove characters that aren't alphanumerics,
#     underscores, or hyphens. Convert to lowercase. Also strip leading and
#     trailing whitespace, dashes, and underscores.
#     """
#     assert isinstance(value, str)
#     if allow_unicode:
#         value = unicodedata.normalize("NFKC", value)
#     else:
#         value = (
#             unicodedata.normalize("NFKD", value)
#             .encode("ascii", "ignore")
#             .decode("ascii")
#         )
#     value = re.sub(r"[^\w\s-]", "", value.lower())
#     return re.sub(r"[-\s]+", "-", value).strip("-_")


def short_alphanumeric_lower_hash(input_string):
    hash_object = hashlib.sha256(input_string.encode("utf-8"))
    base64_encoded = base64.b64encode(hash_object.digest()).decode("utf-8")
    base64_encoded = "".join(c for c in base64_encoded if c.isalnum())
    return base64_encoded[:8].lower()


# def random_string(length: int = 8):
#     characters = string.ascii_lowercase + string.digits
#     return "".join(random.choice(characters) for i in range(length))


# def get_lineage_link(build_id: str) -> str:
#     """Get the link to the lineage given the build ID."""
#     DMF_URL = gb_environment_config()["dmf_ui"]
#     build_lineage_url = f"{DMF_URL}/gb/builds/{build_id}/lineage"
#     return build_lineage_url


# def unwrap_errors(e: Exception) -> str:
#     """Unwrap nested Exception(Group)s to create a readable message."""
#     assert isinstance(
#         e, Exception
#     ), f"unwrap_errors called with non-exception type: {type(e)} {e}"
#     if isinstance(e, ExceptionGroup):
#         err_strs = []
#         for exc in e.exceptions:
#             err_strs.append(unwrap_errors(exc))
#         return "\n".join(err_strs)
#     if e.__cause__ is not None:
#         assert isinstance(e.__cause__, Exception)
#         return unwrap_errors(e.__cause__)
#     if isinstance(e, LogMonitoringFailedException):
#         build_id = e.build_id
#         if FETCH_CLOUD_LOGS_MAX_RETRIES <= 0:
#             return "log monitoring failed (fetching build logs is disabled): " + str(e)
#         if log_manager is not None and build_id != "":
#             try:
#                 logs_str = log_manager.get_build_logs(build_id=build_id)
#                 return (
#                     "log monitoring failed: fetched the step logs:\n\n```\n"
#                     + logs_str
#                     + "\n```\n\n"
#                 )
#             except Exception as logfetche:
#                 logger.error(
#                     "failed to fetch the logs for the build %s : %s",
#                     build_id,
#                     logfetche,
#                 )
#         return "log monitoring failed (also failed to fetch build logs): " + str(e)
#     elif isinstance(e, ValueError):
#         return "value error: " + str(e)
#     return str(e)


# def get_pr_error_message(e: Exception, err_stack: str) -> str:
#     """Get a readable error message to post to the pull request."""
#     logger.debug("get_pr_error_message start")
#     readable_error = unwrap_errors(e)
#     body = f"""
# The run failed due to exception(s):
# {readable_error}

# <details>

# <summary>See more details</summary>

# ### Full Stack Trace

# ```
# {err_stack}
# ```

# </details>
# """
#     logger.debug("get_pr_error_message end")
#     return body


# def download_file(url: str, output_path: Path, timeout: int = 10 * 60) -> None:
#     """
#     Download a file from the given URL.
#     https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
#     """
#     # NOTE the stream=True parameter below
#     with requests.get(url, stream=True, timeout=timeout) as r:
#         r.raise_for_status()
#         output_path.parent.mkdir(exist_ok=True, parents=True)
#         with open(output_path, "wb") as f:
#             for chunk in r.iter_content(chunk_size=8192):
#                 # If you have chunk encoded response uncomment if
#                 # and set chunk_size parameter to None.
#                 # if chunk:
#                 f.write(chunk)


# def get_common_ancestor(ps: List[Path]) -> Path:
#     """Get the longest common ancestor/parent directory."""
#     assert len(ps) > 0
#     if len(ps) == 1:
#         return ps[0]
#     p_strs = [str(p) for p in ps]
#     sorted_p_strs = sorted(p_strs)
#     first = sorted_p_strs[0]
#     last = sorted_p_strs[-1]
#     smaller = first if len(first) <= len(last) else last
#     until = len(smaller)
#     i = 0
#     while i < until:
#         if first[i] != last[i]:
#             break
#         i += 1
#     answer = smaller[:i]
#     return Path(answer)
