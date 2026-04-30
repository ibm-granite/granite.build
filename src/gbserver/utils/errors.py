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

"""Functions for dealing with errors."""

from typing import List, Type


def simple_unwrap_errors(e: BaseException) -> str:
    """Flatten nested exceptions."""
    assert isinstance(e, BaseException), f"called with non-exception type: {type(e)} {e}"
    prefix = e.__class__.__name__
    if isinstance(e, ExceptionGroup):
        err_strs = []
        for exc in e.exceptions:
            err_strs.append(simple_unwrap_errors(exc))
        child_error = "\n".join(err_strs)
        return f"{prefix} {e.message}:\n{child_error}"
    if e.__cause__ is not None:
        return simple_unwrap_errors(e.__cause__)
        # child_error = simple_unwrap_errors(e.__cause__)
        # return f"{prefix}: {child_error}"
    # elif isinstance(e, ValueError):
    #     return "value error: " + str(e)
    return f"{prefix}: {e}"


def gather_specific_exception_type(
    e: BaseException, cls: Type[BaseException]
) -> List[BaseException]:
    """
    Check for nested exceptions and gather
    every exception of the given 'cls' type.

    Returns: List[cls]
    """
    assert isinstance(e, BaseException), f"called with non-exception type: {type(e)} {e}"
    all_excs: List[BaseException] = []
    if isinstance(e, cls):
        all_excs.append(e)
    if e.__cause__ is None:
        # leaf node
        pass
    else:
        # inner node
        if isinstance(e, ExceptionGroup):
            # list of exceptions
            for exc in e.exceptions:
                curr_excs = gather_specific_exception_type(exc, cls)
                all_excs.extend(curr_excs)
        else:
            # not a list
            curr_excs = gather_specific_exception_type(e.__cause__, cls)
            all_excs.extend(curr_excs)
    return all_excs
