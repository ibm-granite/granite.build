# Copyright Granite.Build Authors
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
import logging
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from gbserver.types.validation import GBValidationErrors

logger = logging.getLogger(__name__)


class SecretConfig(BaseModel):
    """Definition of a single secret."""

    payload: str = Field(
        ...,
        description="Base64-encoded secret payload",
    )
    labels: Optional[List[str]] = Field(
        default=None,
        description="Optional Label Names (e.g. encode:base64, decode:json)",
    )
    secret_group: Optional[str] = Field(
        default=None,
        description="Secret Group Name (e.g. gbspace-public, gbspace-wca4a)",
    )

    @field_validator("payload")
    def payload_must_be_base64(cls, v: str) -> str:
        errors = GBValidationErrors()

        if not v or not v.strip():
            errors.add(
                err="Secret payload must not be empty",
            )
        else:
            try:
                base64.b64decode(
                    v, validate=True
                )  # checks whether v is a valid base64-encoded string
            except Exception:
                errors.add(
                    err="Secret payload must be valid base64",
                )

        if not errors.is_valid():
            raise ValueError("\n".join(str(e) for e in errors))

        return v

    @field_validator("labels", mode="before")
    def validate_labels_format(cls, v):
        if v is None:
            return v

        errors = GBValidationErrors()

        for label in v:
            if ":" not in label:
                errors.add(
                    err=(
                        f"Invalid label '{label}'. "
                        "Expected format 'action:value' (e.g. encode:base64)"
                    )
                )

        if not errors.is_valid():
            raise ValueError("\n".join(str(e) for e in errors))

        return v


class SpaceSecretsConfig(BaseModel):
    """Secrets belonging to a single space."""

    secrets: Dict[str, SecretConfig]

    @model_validator(mode="after")
    def ensure_at_least_one_secret(self):
        errors = GBValidationErrors()

        if not self.secrets:
            errors.add(err="Each space must define at least one secret")

        if not errors.is_valid():
            raise ValueError("\n".join(str(e) for e in errors))

        return self


class SpacesConfig(BaseModel):
    """Top-level spaces mapping."""

    spaces: Dict[str, SpaceSecretsConfig]

    @model_validator(mode="after")
    def ensure_spaces_exist(self):
        errors = GBValidationErrors()

        if not self.spaces:
            errors.add(err="At least one space must be defined")

        if not errors.is_valid():
            raise ValueError("\n".join(str(e) for e in errors))

        return self
