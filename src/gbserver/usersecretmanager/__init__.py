#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Pluggable per-user secret management.

Mirrors the :mod:`gbserver.spacesecretmanager` package but scopes secrets to a
``user_id`` instead of a space. Backends are auto-discovered by filename, the
same way space secret managers are.
"""
