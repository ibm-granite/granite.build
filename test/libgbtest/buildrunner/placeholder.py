# SPDX-License-Identifier: Apache-2.0

"""Shared sentinel for `gbtest render` skeleton values the user must fill in.

Kept in its own dependency-free leaf module so both the generator
(``buildtest_gen``) and the spec loader (``buildtest``) can import the single
source of truth without pulling in each other's transitive deps.
"""

PLACEHOLDER = "FIXME"
"""Token emitted for non-derivable expectation values (e.g. ``step_count``) and
rejected at spec-load time if left unreplaced."""
