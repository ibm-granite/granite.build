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

"""
Map the GitHub username to the LSF username.
"""

import json
import os
import pwd
from argparse import ArgumentParser, Namespace
from pathlib import Path


def get_file_owner(p: Path) -> str:
    """Get the owner of a file."""
    x = os.stat(p)
    uid = x.st_uid
    owner_name = pwd.getpwuid(uid).pw_name
    return owner_name


def process_user_mapping(args: Namespace) -> None:
    """Map the username using the given mapping file."""
    user_mapping_file: Path = args.user_mapping_file.resolve()
    input_username: str = args.input_username
    input_default_username: str = args.input_default_username
    skip_ownership_check: bool = args.skip_ownership_check
    expected_user_mapping_file_owner: str = args.user_mapping_file_owner
    if expected_user_mapping_file_owner == "":
        expected_user_mapping_file_owner = input_default_username
    assert (
        user_mapping_file.is_file()
    ), f"invalid user_mapping_file: {user_mapping_file}"
    assert (
        input_default_username
    ), f"invalid input_default_username: {input_default_username}"
    assert (
        expected_user_mapping_file_owner
    ), f"invalid expected_user_mapping_file_owner: {expected_user_mapping_file_owner}"
    if not skip_ownership_check:
        actual_user_mapping_file_owner = get_file_owner(user_mapping_file)
        assert actual_user_mapping_file_owner == expected_user_mapping_file_owner, (
            f"invalid owner for user mapping file at {user_mapping_file}"
            + f" expected: {expected_user_mapping_file_owner} actual: {actual_user_mapping_file_owner}"
        )
    with open(user_mapping_file, "r", encoding="utf-8") as f:
        user_mapping_file_data = json.load(f)
    assert isinstance(
        user_mapping_file_data, dict
    ), f"invalid user_mapping_file_data: {user_mapping_file_data} at path {user_mapping_file}"
    assert (
        "user_mapping" in user_mapping_file_data
    ), f"invalid user_mapping_file_data: {user_mapping_file_data} at path {user_mapping_file}"
    user_mapping = user_mapping_file_data["user_mapping"]
    assert isinstance(
        user_mapping, dict
    ), f"invalid user_mapping: {user_mapping} at path {user_mapping_file}"
    output_username = user_mapping.get(input_username, input_default_username)
    assert isinstance(
        output_username, str
    ), f"invalid output_username: {output_username} in {user_mapping}"
    print(output_username, end=None)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--user-mapping-file",
        dest="user_mapping_file",
        type=Path,
        required=True,
        help="input path to the user mapping file",
    )
    parser.add_argument(
        "--input-username",
        dest="input_username",
        type=str,
        required=True,
        help="input username to be mapped",
    )
    parser.add_argument(
        "--input-default-username",
        dest="input_default_username",
        type=str,
        required=True,
        help="default username to use if the user isn't found",
    )
    parser.add_argument(
        "--user-mapping-file-owner",
        dest="user_mapping_file_owner",
        type=str,
        required=False,
        default="",
        help="the expected owner of the user mapping file",
    )
    parser.add_argument(
        "--skip-ownership-check",
        dest="skip_ownership_check",
        required=False,
        default=False,
        action="store_true",
        help="skip checking ownership of the user mapping file",
    )
    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    process_user_mapping(args=args)


if __name__ == "__main__":
    main()
