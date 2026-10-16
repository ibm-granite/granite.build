# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Canonical comparison helper for config_data.

The UX-stored config_data (dict written to MySQL JSON) and the template's
config_data (loaded from a YAML file) can differ in trivial ways (key order,
int vs float). ``normalized`` produces a comparable canonical form so those
non-meaningful differences do not trigger a spurious suffix on the bridge.
"""

from numbers import Real
from typing import Any


def normalized(value: Any) -> Any:
    """Return an order-insensitive, numeric-canonical representation of ``value``.

    - dicts -> sorted tuple of (key, normalized(child)) pairs
    - lists/tuples -> tuple of normalized children (order preserved)
    - bool -> kept distinct from numbers
    - int/float -> coerced to float so 1 == 1.0
    - everything else -> returned as-is
    """
    if isinstance(value, dict):
        return tuple(sorted((k, normalized(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(normalized(v) for v in value)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, Real):
        return float(value)
    return value
