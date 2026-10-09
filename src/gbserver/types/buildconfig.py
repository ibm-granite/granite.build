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
The main parser for the build.yaml file.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Type
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

from gbcommon.types.gbenvconfig import is_standalone
from gbcommon.uri.env import is_relative_env_uri
from gbcommon.uri.uri import URI
from gbserver.types.artifact import ArtifactType
from gbserver.types.config import Config
from gbserver.types.constants import (
    BUILD_YAML_BASE_KEYS,
    CODE_GBSERVER_BUILTINS_STEPS_GBSTEP_URI,
    CURRENT_BUILD_YAML_VERSION,
    FILE_SCHEME,
)
from gbserver.types.validation import GBValidationErrors, GBValidationErrorType
from gbserver.utils.logger import get_logger
from gbserver.utils.template import fill_template

logger = get_logger(__name__)

BUILD_FILENAME = "build.yaml"
BUILD_RUN_YAML_FILENAME = "run.yaml"


def reject_host_file_uri(uri: Optional[str], field: str) -> None:
    """Refuse a step.yaml URI naming a file on the gbserver host, unless STANDALONE.

    The step.yaml counterpart of BuildConfig's build.yaml ``file:`` check: a step
    can come from the user's own git repo, so its ``validator_uri`` or monitor
    ``ref`` could otherwise name e.g. ``file:///home/gbserver/.kube`` and have it
    synced into the step. These are resolved with a ``file`` default scheme, so
    a bare path counts too. ``space://`` refs are allowed (they resolve through
    the space's admin-controlled base_uris, contained by SpaceURI).

    Args:
        uri: The URI as written in step.yaml.
        field: The field it came from, used in the error message.

    Raises:
        ValueError: If the server is not STANDALONE and the URI is, or may render
            to, a ``file:`` URI.
    """
    if not is_standalone() and _may_be_file_uri(uri, FILE_SCHEME):
        raise ValueError(
            f"{field} '{uri}': file: URIs are only allowed on a standalone server"
        )


def _may_be_file_uri(uri: Optional[str], default_scheme: str) -> bool:
    """Return True if a build.yaml URI is, or may render to, a ``file:`` URI.

    Args:
        uri: The URI as written in build.yaml (None/empty means "not set").
        default_scheme: The scheme its consumer applies to a bare path
            (``"file"`` for step/environment URIs, ``"git"`` for inputs/outputs).

    Returns:
        bool: True if the scheme is ``file`` (or ``default_scheme`` is ``file``
        and none is given), or if any Jinja syntax (``{{``, ``{%`` or ``{#``)
        appears before the first ``:``/``/`` -- the scheme is then only decided
        when it is rendered (possibly with run-time values), so it cannot be
        shown not to be ``file``.
    """
    if not uri:
        return False
    prefix = re.split(r"[:/]", uri, maxsplit=1)[0]
    if any(tag in prefix for tag in ("{{", "{%", "{#")):
        return True
    return urlparse(uri, default_scheme).scheme == FILE_SCHEME


def _has_parent_segment(uri: Optional[str]) -> bool:
    """Return True if a URI contains a ``..`` path segment.

    The URI is checked both as written and after filling in space-config
    templates (as URI.get_uri will), so ``{{ '..' }}`` cannot hide one. Segments
    are split on ``/`` and on the ``:``/``#``/``?``/``&``/``=`` separators, so
    ``file:../x`` and a git ``#subdirectory=../x`` are caught too; ``..`` inside
    a name (``a..b``) is not a segment.

    Args:
        uri: The URI as written in build.yaml (None/empty means "not set").

    Returns:
        bool: True if any segment of the raw or rendered URI is exactly ``..``.
    """
    if not uri:
        return False
    candidates = [uri]
    try:
        candidates.append(fill_template(uri, URI.get_space_config()))
    except (ValueError, RuntimeError):
        pass  # an unrenderable template fails later in URI.get_uri anyway
    return any(".." in re.split(r"[:/#?&=]", c) for c in candidates)


class InvalidTarget(Exception):
    """Indicates that the step is invalid."""


class BuildFailure(Exception):
    """Indicates that the build failed."""


class BuildTargetInputMetadataConfig(Config):
    """Metadata regarding an input artifact."""

    checksum: str = ""
    other_locations: List[str] = Field(default_factory=list)


class BuildTargetInputConfig(Config):
    """Input Artifact definition"""

    uri: Optional[str] = None
    metadata: Optional[BuildTargetInputMetadataConfig] = None
    binding: Optional[str] = None
    # If True, the target will wait for push to be finished before triggering
    wait_for_push: Optional[bool] = False
    event: Optional[str] = None

    def get_binding_parts(self: Self) -> List[str]:
        """
        Given a binding: tunedmodel.tuned_checkpoint
        Returns: ["tunedmodel", "tuned_checkpoint"]
        """
        if self.binding is None:
            raise ValueError("no binding specified")
        return self.binding.split(".")


class BuildTargetOutputEventSelectorsConfig(BaseModel):
    """
    Select the artifact event(s) that triggers this output.
    We match against the fields of the event's payload.
    """

    field_name: str = ""
    field_value: str = ""
    field_value_regex: str = ""
    # computed fields
    _field_value_regex: Optional[re.Pattern] = None

    def model_post_init(self: Self, context: Any, /) -> None:
        if self.field_name == "":
            raise ValueError("field name is missing")
        if self.field_value == "" and self.field_value_regex == "":
            raise ValueError("neither field value and regex were provided")
        if self.field_value != "" and self.field_value_regex != "":
            raise ValueError("both field value and regex were provided")
        if self.field_value_regex != "":
            self._field_value_regex = re.compile(self.field_value_regex)


class BuildTargetOutputPushConfig(Config):
    """Push configuration for an output artifact (build.yaml ``store_push`` block).

    Attributes:
        mode: The push mode. Optional and rarely needed — the store is inferred
            from the output ``uri`` scheme.
        config: Store-specific push configuration, keyed by store (e.g. the
            ``hf`` block for HuggingFace). Interpreted by the store's push path.
    """

    mode: Optional[str] = None
    config: Dict = Field(default_factory=dict)


class BuildTargetOutputConfig(Config):
    """Output Artifact definition

    Attributes:
        uri: Output artifact URI template.
        type: Optional artifact type (e.g. ``dataset``/``model``/``fileset``).
            Applied to the registered artifact when the store pushes inline (no
            push step emits one), e.g. the ``env_local`` store. Defaults to
            ``UNDEFINED`` when omitted.
        public: Store-specific push visibility flag (currently HuggingFace only:
            publish the pushed repo). Convenience alias for the store's own
            ``store_push`` visibility key; interpreted by the store's push path,
            which also validates it against the ``uri`` scheme.
        event_selectors: Event selector rules for matching artifact events.
        store_push: Optional per-output push configuration from build.yaml.
        space_name: Build space name; set at runtime, not parsed from build.yaml.
    """

    uri: Optional[str] = None
    type: Optional[ArtifactType] = None
    public: Optional[bool] = None
    event_selectors: List[BuildTargetOutputEventSelectorsConfig] = Field(
        default_factory=list
    )
    store_push: Optional[BuildTargetOutputPushConfig] = None
    # Populated at runtime from Build.space — not a build.yaml field.
    space_name: Optional[str] = None


class BuildTargetStepConfig(Config):
    """A single target step in a build file."""

    step_uri: Optional[str] = Field(
        default=None
    )  # defaults to "" to support empty step uri if `step_uri` field is present
    launcher: Optional[str] = Field(default=None)
    config: Optional[dict[str, Any]] = Field(default_factory=dict)
    config_dir: Optional[str] = Field(default=None)
    retry_enabled: Optional[bool] = None
    retry_transparently: Optional[bool] = None

    # FIELD VALIDATOR
    # Runs when step_uri as a field EXISTS in YAML, even if empty string or None.
    @field_validator("step_uri", mode="before")
    def apply_default_for_empty_value(cls, v):
        # If value is missing, empty string, or just whitespace -> use base step
        if v is None or (isinstance(v, str) and v.strip() == ""):
            default_step_uri = CODE_GBSERVER_BUILTINS_STEPS_GBSTEP_URI
            logger.info(
                f"[FIELD VALIDATOR - BuildTargetStepConfig] EMPTY STEP URI PROVIDED. DEFAULTING TO: {default_step_uri} ======="
            )
            return default_step_uri
        return v

    # MODEL VALIDATOR
    # Runs ALWAYS — even when step_uri field is missing entirely from build.yaml
    @model_validator(mode="after")
    def fill_missing_step_uri(self):
        if not self.step_uri:
            self.step_uri = CODE_GBSERVER_BUILTINS_STEPS_GBSTEP_URI
            logger.info(
                f"[MODEL VALIDATOR - BuildTargetStepConfig] STEP URI OMITTED IN BUILD.YAML, DEFAULTING TO {self.step_uri}"
            )
        return self

    # -------------------------------------------------------------------------
    # This utility validates the k8s env section inside the build.yaml config.
    # K8s treats unquoted env VALUE integers differently,
    # numeric must actually be strings, else it throws error.
    # This validator recursively walks the entire k8s config tree and raises errors if
    # k8s.env value is an integer where a string is expected.
    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_k8s_env_section(self):
        """Extract k8s config if present; if absent, nothing to validate
        Validate only the `k8s.env` section: ensure env values are STRINGS
        """
        k8s_cfg = self.config.get("k8s") if self.config else None
        if not k8s_cfg:
            return self

        errors = GBValidationErrors()

        # Nothing to validate if env not present
        if "env" not in k8s_cfg:
            return self

        env_cfg = k8s_cfg["env"]

        # --------------------------------------------------------------------
        # Case 1: dict env: (EXAMPLE)
        # env:
        #   NCCL_TIMEOUT:
        #     value: "10800000"
        # --------------------------------------------------------------------

        if isinstance(env_cfg, dict):
            for env_name, env_val in env_cfg.items():
                if isinstance(env_val, dict) and "value" in env_val:
                    v = env_val["value"]
                    if isinstance(v, int):
                        errors.add(
                            err=(
                                f"k8s env variable `env.{env_name}.value` must be a STRING, "
                                f"not `{type(v).__name__}`."
                            )
                        )

        elif isinstance(env_cfg, list):
            for i, env_val in enumerate(env_cfg):
                if isinstance(env_val, dict) and "value" in env_val:
                    v = env_val["value"]
                    if isinstance(v, int):
                        errors.add(
                            err=(
                                f"k8s env variable `env[{i}].value` must be a STRING, "
                                f"not `{type(v).__name__}`."
                            )
                        )

        if not errors.is_valid():
            raise ValueError(
                "Invalid k8s config found in build.yaml: Kubernetes env requires numeric values to be quoted.\n"
                + "\n".join(str(e) for e in errors)
            )

        return self


class BuildTargetConfig(Config):
    """A single target in a build file."""

    environment_uri: str
    inputs: Optional[Dict[str, BuildTargetInputConfig]] = Field(default_factory=dict)
    outputs: Optional[Dict[str, BuildTargetOutputConfig]] = Field(default_factory=dict)
    steps: List[BuildTargetStepConfig]
    dependency: Optional[set[str]] = Field(default_factory=set)


class BuildRetryConfig(Config):
    """Retry configuration for a build."""

    max_retries: int = 0
    target_reuse_enabled: bool = True


class BuildConfig(Config):
    """A single build config."""

    version: str = CURRENT_BUILD_YAML_VERSION
    name: str = ""
    targets: dict[str, BuildTargetConfig]
    retries: BuildRetryConfig = Field(default_factory=BuildRetryConfig)

    @classmethod
    def from_yaml(
        cls: Type[Self],
        path: Path,
        basekey: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs,
    ) -> Self:
        if "basekeys" not in kwargs:
            kwargs["basekeys"] = BUILD_YAML_BASE_KEYS
        self = super().from_yaml(path=path, basekey=basekey, context=context, **kwargs)
        validate = kwargs.get("validate", True)
        assert isinstance(validate, bool), f"invalid validate flag: {validate}"
        if validate:
            self.my_validate().raise_if_invalid()
        return self

    def __validate_step_uris(self: Self) -> GBValidationErrors:
        logger.info("validating the step URIs of the build")
        errors = GBValidationErrors()
        for target_name, target in self.targets.items():
            logger.info("checking the env of the target: %s", target_name)
            logger.debug("target: %s", target)  # Avoid blowing travis log limits
            if target.environment_uri == "":
                errors.add(err=f"Target `{target_name}` the env URI is empty")
            logger.info("checking the steps of the target: %s", target_name)
            for i, step in enumerate(target.steps):
                if step.step_uri == "":
                    errors.add(err=f"Target `{target_name}` Step `{i}` URI is empty")
        return errors

    def __validate_target_inputs(self: Self) -> GBValidationErrors:
        logger.info("validating the inputs of the build")
        errors = GBValidationErrors()
        for target_name, target in self.targets.items():
            err_prefix = f"Target `{target_name}`:"
            logger.info("checking inputs of the target: %s", target_name)
            logger.debug("target: %s", target)  # Avoid blowing travis log limits
            if target.outputs is None:
                warning = f"{err_prefix} the target has no outputs"
                logger.warning(warning)
                errors.add_warning(warning=warning)
            if target.inputs is None:
                logger.warning("the target %s has no inputs", target_name)
                continue
            for target_input_name, target_input in target.inputs.items():
                err_prefix = f"Target `{target_name}` Input `{target_input_name}`:"
                logger.info("checking: %s %s", err_prefix, target_input)
                if target_input.uri is None and target_input.binding is None:
                    errors.add(f"{err_prefix} the input has no URI or binding")
                    continue
                if target_input.uri is not None and target_input.binding is not None:
                    errors.add(f"{err_prefix} the input has both a URI and a binding")
                    continue
                if target_input.uri is not None:
                    logger.info("checking the input URI: %s", target_input.uri)
                    if target_input.uri == "":
                        errors.add(f"{err_prefix} the target input URI is empty")
                    continue
                if target_input.binding is not None:
                    logger.info(
                        "checking if input binding is valid: %s", target_input.binding
                    )
                    binding_target_name, binding_target_output_name = (
                        target_input.get_binding_parts()
                    )
                    if binding_target_name not in self.targets:
                        errors.add(
                            err=f"{err_prefix} binding to a non-existent target `{binding_target_name}`",
                            type=GBValidationErrorType.NOT_EXIST,
                        )
                        continue
                    build_config_target = self.targets[binding_target_name]
                    if (
                        build_config_target.outputs is None
                        or binding_target_output_name not in build_config_target.outputs
                    ):
                        errors.add(
                            err=f"{err_prefix} binding to a non-existent output `{binding_target_output_name}` of the target `{binding_target_name}`",
                            type=GBValidationErrorType.NOT_EXIST,
                        )
                    continue
        return errors

    def __validate_env_uris(self: Self) -> GBValidationErrors:
        """Reject relative ``env:`` URIs on any target input or output.

        The ``env:`` store performs no transfer, so a relative path has no defined
        resolution root (see :func:`gbcommon.uri.env.is_relative_env_uri`). Catch
        it at load time — for both inputs and outputs — so a misauthored build
        fails fast with a clear message instead of registering a dangling path.
        """
        errors = GBValidationErrors()
        for target_name, target in self.targets.items():
            for input_name, target_input in (target.inputs or {}).items():
                if target_input.uri and is_relative_env_uri(target_input.uri):
                    errors.add(
                        f"Target `{target_name}` Input `{input_name}`: relative "
                        f"env:// URI '{target_input.uri}' is not supported — use an "
                        "absolute path (env:///...)."
                    )
            for output_name, target_output in (target.outputs or {}).items():
                if target_output.uri and is_relative_env_uri(target_output.uri):
                    errors.add(
                        f"Target `{target_name}` Output `{output_name}`: relative "
                        f"env:// URI '{target_output.uri}' is not supported — use an "
                        "absolute path (env:///...)."
                    )
        return errors

    def __validate_no_file_uris(self: Self) -> GBValidationErrors:
        """Reject ``file:`` URIs in a build.yaml unless the server is STANDALONE.

        A ``file:`` URI names a path on the build host itself. On a shared server
        that would let a build.yaml sync the server's own files to compute
        (``step_uri: file:///home/gbserver/.kube``, an ``environment_uri``) or
        read/write them through a ``file:`` input/output. A STANDALONE server
        runs on the user's own machine, so it keeps accepting local paths.

        Checked: every target's ``environment_uri``, each step's ``step_uri``
        (except the server's own builtin gbstep default, which an empty/missing
        ``step_uri`` is rewritten to), and each input/output ``uri``.

        Returns:
            GBValidationErrors: one error per offending URI (none in STANDALONE).
        """
        errors = GBValidationErrors()
        if is_standalone():
            return errors
        hint = "file: URIs are only allowed on a standalone server"
        for target_name, target in self.targets.items():
            prefix = f"Target `{target_name}`"
            # environment_uri/step_uri are resolved with a "file" default scheme.
            if _may_be_file_uri(target.environment_uri, FILE_SCHEME):
                errors.add(
                    f"{prefix} environment_uri '{target.environment_uri}': {hint}"
                )
            for i, step in enumerate(target.steps):
                uri = step.step_uri or ""
                if uri != CODE_GBSERVER_BUILTINS_STEPS_GBSTEP_URI and (
                    _may_be_file_uri(uri, FILE_SCHEME)
                ):
                    errors.add(f"{prefix} Step `{i}` step_uri '{uri}': {hint}")
            # Inputs/outputs go through URI.get_uri, whose default scheme is git.
            for name, target_input in (target.inputs or {}).items():
                if _may_be_file_uri(target_input.uri, "git"):
                    errors.add(f"{prefix} Input `{name}` '{target_input.uri}': {hint}")
            for name, target_output in (target.outputs or {}).items():
                if _may_be_file_uri(target_output.uri, "git"):
                    errors.add(
                        f"{prefix} Output `{name}` '{target_output.uri}': {hint}"
                    )
        return errors

    def __validate_no_parent_segments(self: Self) -> GBValidationErrors:
        """Reject ``..`` path segments in step_uri and environment_uri.

        A ``..`` segment could climb out of whatever the URI resolves against (a
        space's base_uris, a git checkout's ``#subdirectory=``, a local path), so
        it is rejected in every mode, STANDALONE included.

        Returns:
            GBValidationErrors: one error per offending URI.
        """
        errors = GBValidationErrors()
        hint = "'..' path segments are not allowed"
        for target_name, target in self.targets.items():
            prefix = f"Target `{target_name}`"
            if _has_parent_segment(target.environment_uri):
                errors.add(
                    f"{prefix} environment_uri '{target.environment_uri}': {hint}"
                )
            for i, step in enumerate(target.steps):
                if _has_parent_segment(step.step_uri):
                    errors.add(
                        f"{prefix} Step `{i}` step_uri '{step.step_uri}': {hint}"
                    )
        return errors

    def __validate_output_push(self: Self) -> GBValidationErrors:
        """Validate each output's push config via the owning store's own rules.

        Store-specific push rules live with the store, not here — this only walks
        the outputs and collects what each store's validator reports (currently
        only the HF push guard). Lazily imported to avoid a types→spaces cycle.
        """
        from gbserver.spaces.hf_push_config import validate_output_push

        errors = GBValidationErrors()
        for target_name, target in self.targets.items():
            for output_name, output in (target.outputs or {}).items():
                err = validate_output_push(output_name, output)
                if err:
                    errors.add(f"Target `{target_name}`: {err}")
        return errors

    def __validate_lh_output_uris(self: Self) -> GBValidationErrors:
        """Parse literal lh:// output URIs so LhURI's rules (e.g. reserved table
        names) fail at submit time. Templated URIs are checked at push time.
        """
        from gbcommon.uri.lh import LH_URI_SCHEME

        errors = GBValidationErrors()
        for target_name, target in self.targets.items():
            for output_name, output in (target.outputs or {}).items():
                uri = output.uri
                if not uri or not uri.startswith(f"{LH_URI_SCHEME}://") or "{{" in uri:
                    continue
                try:
                    URI.get_uri(uri)
                except ValueError as e:
                    errors.add(f"Target `{target_name}` Output `{output_name}`: {e}")
        return errors

    def my_validate(self: Self) -> GBValidationErrors:
        """Validate the build config."""
        logger.info("validating the build config")
        errors = GBValidationErrors()
        if len(self.targets) == 0:
            errors.add("the build has no targets specified")
            return errors
        errors.add(self.__validate_step_uris())
        errors.add(self.__validate_target_inputs())
        errors.add(self.__validate_env_uris())
        errors.add(self.__validate_no_file_uris())
        errors.add(self.__validate_no_parent_segments())
        errors.add(self.__validate_output_push())
        errors.add(self.__validate_lh_output_uris())
        logger.info("validated the build config and found %d errors", len(errors))
        return errors


class BuildRunConfig(Config):
    """A single build run config."""

    targets_to_run: dict[str, Any]
