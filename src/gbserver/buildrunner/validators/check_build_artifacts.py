#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0


from gbserver.storage.artifact_registry import IArtifactRegistry
from gbserver.types.buildconfig import BuildConfig
from gbserver.types.validation import GBValidationErrors, GBValidationErrorType


def check_build_input_artifacts_registered(
    artifact_registry: IArtifactRegistry,
    build_config: BuildConfig,
    space_name: str,
) -> GBValidationErrors:
    """Check if the build's input artifacts are registered."""
    assert isinstance(
        artifact_registry, IArtifactRegistry
    ), f"expected artifact registry {type(artifact_registry)} {artifact_registry}"
    errors = GBValidationErrors()
    solution = (
        "You can use the `gb artifact push` to push a new dataset/model to Lakehouse"
        + " or the `gb artifact register` command to register something already in Lakehouse."
        + " Use the `--help` option to learn more details."
    )
    for target_name, target in build_config.targets.items():
        if target.inputs is None:
            continue
        for t_input_name, t_input in target.inputs.items():
            input_artifact_uri_str = t_input.uri
            if input_artifact_uri_str is None:
                continue
            input_artifact = artifact_registry.get_by_uri(
                uri=input_artifact_uri_str, space_name=space_name
            )
            if input_artifact is not None:
                continue
            err_prefix = f"Target `{target_name}` Input `{t_input_name}`:"
            errors.add(
                err=f"There is no artifact registered for the URI: `{input_artifact_uri_str}` in the space: `{space_name}`",
                type=GBValidationErrorType.NOT_EXIST,
                solution=solution,
                prefix=err_prefix + " ",
            )
    return errors
