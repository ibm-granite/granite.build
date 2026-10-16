#!/usr/bin/env python3
"""Prepend the Apache-2.0 SPDX header to backend Python files that lack it."""

from __future__ import annotations

import pathlib

HEADER = "# Copyright IBM Corp. 2024-2026\n# SPDX-License-Identifier: Apache-2.0\n"

for path in pathlib.Path("src/autotunex").rglob("*.py"):
    text = path.read_text(encoding="utf-8")
    if "SPDX-License-Identifier" in text:
        continue
    path.write_text(HEADER + text, encoding="utf-8")
    print(f"headered {path}")
