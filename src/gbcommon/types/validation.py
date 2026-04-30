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
Types related to validation.
"""

import builtins
import logging
from enum import StrEnum, auto
from typing import Any, Dict, List, Optional, Self, Type, Union

from pydantic import BaseModel, Field

from gbcommon.utils.errors import gather_specific_exception_type, simple_unwrap_errors

logger = logging.getLogger(__name__)


class GBValidatorConfig(BaseModel):
    """Config for intializaing a GBValidator."""

    type: str
    config: Dict = Field(default_factory=dict)


class GBValidationErrorType(StrEnum):
    """The type of validation error."""

    GENERIC = auto()
    DEPRECATED = auto()
    EXCEPTION = auto()
    NOT_EXIST = auto()
    RECOMMENDATION = auto()


class GBValidationError(BaseModel):
    """The validation error."""

    type: GBValidationErrorType = GBValidationErrorType.GENERIC
    error: str
    solution: str = ""  # a recommendation for the user to follow

    @classmethod
    def from_exception(cls: Type[Self], e: Exception) -> Self:
        """Create a validation error from a caught exception."""
        se = simple_unwrap_errors(e)
        self = cls(type=GBValidationErrorType.EXCEPTION, error=se)
        return self

    def add_prefix(self: Self, prefix: str) -> None:
        """Add a prefix to the error message."""
        self.error = prefix + self.error

    def __str__(self: Self) -> str:
        s = f"[ validation error - {self.type} ]: {self.error}"
        if self.solution != "":
            s += f" ; solution: {self.solution}"
        return s


class GBValidationWarningType(StrEnum):
    """The type of validation warning."""

    GENERIC = auto()
    DEPRECATED = auto()
    RECOMMENDATION = auto()


class GBValidationWarning(BaseModel):
    """The validation warning."""

    type: GBValidationWarningType = GBValidationWarningType.GENERIC
    warning: str
    solution: str = ""  # a recommendation for the user to follow

    def add_prefix(self: Self, prefix: str) -> None:
        """Add a prefix to the warning message."""
        self.warning = prefix + self.warning

    def __str__(self: Self) -> str:
        s = f"[ validation warning - {self.type} ]: {self.warning}"
        if self.solution != "":
            s += f" ; solution: {self.solution}"
        return s


class GBValidationErrorsException(Exception):
    """An exception contains the errors."""

    def __init__(self: Self, errors: Any, **kwargs: dict):
        """'errors' must be an instance of GBValidationErrors"""
        # Commented to avoid circular dependency
        # assert isinstance(errors, GBValidationErrors)
        self.errors = errors
        super().__init__(str(errors), **kwargs)


class GBValidationErrors(BaseModel):
    """A list of validation errors."""

    errors: List[GBValidationError] = Field(default_factory=list)
    warnings: List[GBValidationWarning] = Field(default_factory=list)

    def add(
        self: Self,
        err: Union[None, str, Exception, GBValidationError, Self, List],
        type: GBValidationErrorType = GBValidationErrorType.GENERIC,
        solution: str = "",
        prefix: str = "",
    ) -> None:
        """Create an error and add it to the list of errors."""
        if err is None:
            logger.warning("called with None, ignoring...")
            return
        if isinstance(err, str):
            error = GBValidationError(type=type, error=err, solution=solution)
            error.add_prefix(prefix=prefix)
            self.errors.append(error)
            return
        if isinstance(err, Exception):
            error = GBValidationError.from_exception(e=err)
            self.errors.append(error)
            return
        if isinstance(err, GBValidationError):
            error = err
            error.add_prefix(prefix=prefix)
            self.errors.append(error)
            return
        if isinstance(err, list):
            for curr_err in err:
                assert isinstance(
                    curr_err, (self.__class__, GBValidationError)
                ), f"invalid type: {builtins.type(curr_err)} {curr_err}"
                self.add(err=curr_err, prefix=prefix)
            return
        assert isinstance(err, self.__class__), f"invalid type: {builtins.type(err)} {err}"
        for error in err.errors:
            self.add(err=error, prefix=prefix)
        for warning in err.warnings:
            self.add_warning(warning=warning, prefix=prefix)

    def add_warning(
        self: Self,
        warning: Union[None, str, GBValidationWarning],
        type: GBValidationWarningType = GBValidationWarningType.GENERIC,
        solution: str = "",
        prefix: str = "",
    ) -> None:
        """Add a warning to the list of warnings."""
        if warning is None:
            logger.warning("called with None, ignoring...")
            return
        if isinstance(warning, str):
            warn = GBValidationWarning(type=type, warning=warning, solution=solution)
            warn.add_prefix(prefix=prefix)
            self.warnings.append(warn)
            return
        assert isinstance(
            warning, GBValidationWarning
        ), f"invalid type: {builtins.type(warning)} {warning}"
        warn = warning
        warn.add_prefix(prefix=prefix)
        self.warnings.append(warn)

    def is_valid(self: Self, check_warnings: bool = False) -> bool:
        """Returns True if there are no validation errors."""
        if len(self.errors) > 0:
            return False
        if len(self.warnings) > 0:
            if check_warnings:
                return False
            logger.warning("%s", self)
        return True

    def raise_if_invalid(self: Self, check_warnings: bool = False) -> None:
        """Returns True if there are no validation errors."""
        if not self.is_valid(check_warnings=check_warnings):
            raise GBValidationErrorsException(errors=self)

    def __str__(self: Self) -> str:
        err_str = ""
        if len(self.errors) > 0:
            err_str += "\n".join(str(error) for error in self.errors)
        if len(self.warnings) > 0:
            if err_str != "":
                err_str += "\n"
            err_str += "\n".join(str(warning) for warning in self.warnings)
        return err_str

    def __len__(self: Self) -> int:
        """Returns the number of errors (excluding warnings)."""
        return len(self.errors)


def gather_val_errors_from_exception(e: BaseException) -> Optional[GBValidationErrors]:
    """Gather validation errors from nested exceptions."""
    all_excs = gather_specific_exception_type(e, GBValidationErrorsException)
    if len(all_excs) == 0:
        return None
    errors = GBValidationErrors()
    for exc in all_excs:
        assert isinstance(exc, GBValidationErrorsException)
        assert isinstance(exc.errors, GBValidationErrors)
        errors.add(err=exc.errors)
    return errors
