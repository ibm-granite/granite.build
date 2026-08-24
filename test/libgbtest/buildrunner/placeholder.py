# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared sentinel for `gbtest render` skeleton values the user must fill in.

Kept in its own dependency-free leaf module so both the generator
(``buildtest_gen``) and the spec loader (``buildtest``) can import the single
source of truth without pulling in each other's transitive deps.
"""

PLACEHOLDER = "FIXME"
"""Token emitted for non-derivable expectation values (e.g. ``step_count``) and
rejected at spec-load time if left unreplaced."""
