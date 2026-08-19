# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Render a parameterized build.yaml into its executable form (issue #278).

``gb`` is the single source of truth for parameter application; this exposes the
existing ``apply_parameters`` engine as a standalone render (no build submission).
"""

import tempfile
from pathlib import Path
from typing import List, Optional

from jinja2 import UndefinedError

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters, parse_params
from gbcli.utils.gbconstants import BUILD_PARAMETERS_FILE


class RenderError(Exception):
    """A build.yaml could not be rendered (e.g. a parameter was not supplied)."""


def render_build_yaml(
    build_yaml_path: Path,
    cli_params: List[str],
    parameters_path: Optional[Path] = None,
) -> str:
    """Return the executable build.yaml text for a parameterized build file.

    Base parameters come from ``parameters_path`` if given, else a sibling
    ``parameters.yaml`` if present; ``--param KEY=VALUE`` entries override both.

    Args:
        build_yaml_path: The parameterized build.yaml to render.
        cli_params: ``KEY=VALUE`` overrides (highest precedence).
        parameters_path: Optional parameters.yaml; defaults to the sibling file.

    Returns:
        The rendered (executable) build.yaml text.

    Raises:
        FileNotFoundError: if ``build_yaml_path`` is not a file.
        RenderError: if a ``$${...}`` placeholder has no value.
    """
    build_yaml_path = Path(build_yaml_path)
    if not build_yaml_path.is_file():
        raise FileNotFoundError(f"not a file: {build_yaml_path}")

    params_file = (
        Path(parameters_path)
        if parameters_path
        else build_yaml_path.parent / BUILD_PARAMETERS_FILE
    )
    params_from_file = (
        get_params_from_file(str(params_file)) if params_file.is_file() else {}
    )
    params_dict = parse_params(list(cli_params), params_from_file)

    with tempfile.TemporaryDirectory() as scratch:
        try:
            return apply_parameters(
                build_yaml_path.read_text(encoding="utf-8"), [], params_dict, scratch
            )
        except UndefinedError as e:
            raise RenderError(f"missing parameter: {e.message}") from e
