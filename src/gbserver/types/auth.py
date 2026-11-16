#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from gbserver.utils.utils import get_time


class User(BaseModel):
    """Info about the user."""

    login: str
    id: int
    url: str
    html_url: str
    name: str
    email: str = ""
    auth_provider: str = "github"
    gbserver_created_at: datetime = Field(default_factory=get_time)

    @field_validator("email", mode="before")
    @classmethod
    def _coerce_null_email(cls, v):
        """GitHub API may return null for email; coerce to empty string."""
        return v or ""
