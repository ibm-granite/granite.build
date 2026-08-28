# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Declarative base for all ORM tables."""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base. Import this in Alembic's ``env.py``."""
