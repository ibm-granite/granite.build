#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Types used by the REST API for build validation."""

from enum import StrEnum, auto


class BuildValidateRequestType(StrEnum):
    """The type of validation to perform."""

    STATIC = auto()
    DYNAMIC = auto()
