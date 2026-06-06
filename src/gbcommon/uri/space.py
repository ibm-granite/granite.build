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

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Self
from urllib.parse import ParseResult, urlparse

import yaml

from gbcommon.uri.uri import URI
from gbserver.utils.logger import get_logger

GBSPACE_SCHEME = "gb"
SPACE_SCHEME = "space"
STEPS_PREFIX = "steps/"
STEP_FILE_NAME = "step.yaml"

logger = get_logger(__name__)


class SpaceURI(URI):

    _thread_local = threading.local()

    def __new__(self, uri: ParseResult, **kwargs: dict) -> Self:
        if not hasattr(SpaceURI._thread_local, "base_uris"):
            default_base_uris = ["file:"]
            SpaceURI._thread_local.base_uris = default_base_uris
            logger.warning(
                "the space base_uris have not been initialized. Setting it to: %s",
                default_base_uris,
            )
        if not hasattr(SpaceURI._thread_local, "space_secrets"):
            SpaceURI._thread_local.space_secrets = {}
            logger.warning("the space space_secrets have not been initialized.")
        uristr = uri.geturl()
        uri_suffix = uristr
        if uristr.startswith(GBSPACE_SCHEME):
            uri_suffix = uristr.removeprefix(GBSPACE_SCHEME + "://")
        elif uristr.startswith(SPACE_SCHEME):
            uri_suffix = uristr.removeprefix(SPACE_SCHEME + "://")
        # Tier 1: env-co-located step lookup (only for `space://steps/<name>`).
        # The env's own directory carries plain `steps/<name>/` paths — it does
        # not sub-key by step_type — so we check just the bare suffix here, not
        # the step_type-prefixed candidate list.
        if uri_suffix.startswith(STEPS_PREFIX):
            env_dir_uri: Optional[str] = getattr(
                SpaceURI._thread_local, "current_env_dir_uri", None
            )
            if env_dir_uri:
                resolved: URI = URI.get_uri(
                    env_dir_uri,
                    "file",
                    secrets=SpaceURI._thread_local.space_secrets,
                )
                resolved.append_path(uri_suffix)
                if resolved.exists():
                    return resolved  # type: ignore[return-value]
        # Tier 1.5: env-class-match.  Recursively glob all `<base>/**/<name>/step.yaml`
        # files; pick the first (lexicographic) candidate whose `environment_configs`
        # keys contain the active env's class name.  Sub-asset URIs of the form
        # `space://steps/<name>/<rest>` re-use the matched dir.
        match = SpaceURI._try_env_class_match(uri_suffix)
        if match is not None:
            return match  # type: ignore[return-value]
        # Tier 2: step_type chain + env-agnostic fallback against the space's
        # own base_uris.
        candidates = SpaceURI._build_candidates(uri_suffix)
        for candidate in candidates:
            for base_uri in SpaceURI._thread_local.base_uris:
                resolved = URI.get_uri(
                    base_uri, "file", secrets=SpaceURI._thread_local.space_secrets
                )
                resolved.append_path(candidate)
                if resolved.exists():
                    return resolved  # type: ignore[return-value]
        raise ValueError(f"Unresolvable space uri : {uristr}")

    @staticmethod
    def _try_env_class_match(uri_suffix: str) -> Optional[URI]:
        """Resolve `space://steps/<name>[/<rest>]` by env-class metadata match.

        Recursively scans every base_uri (file:// only) for ``<name>/step.yaml``
        files, parses each candidate's ``environment_configs`` keys, and returns
        the first (lexicographically) whose keys contain the active env's class
        name (set via :meth:`with_current_env`).  The directory of the matched
        step.yaml is used as the resolution result; for sub-asset URIs the
        ``<rest>`` portion is appended to that directory.

        Returns ``None`` when:
          * the URI is not a `space://steps/...` lookup;
          * no active env class is set on the thread-local;
          * no candidate step.yaml lists the active env's class.
        Callers fall through to the legacy resolver tiers in that case.
        """
        if not uri_suffix.startswith(STEPS_PREFIX):
            return None
        after = uri_suffix[len(STEPS_PREFIX):]
        if not after:
            return None
        name, _, rest = after.partition("/")
        if not name:
            return None
        env_class: Optional[str] = getattr(
            SpaceURI._thread_local, "current_env_class_name", None
        )
        if not env_class:
            return None
        # Collect (specificity, path-tiebreaker, candidate-path) for every
        # step.yaml whose env_configs lists the active env class.  Specificity
        # is the count of env_configs keys — the smaller, the more env-specific
        # the file is.  We prefer the most specific match so a single-env split
        # file beats a multi-env catch-all that happens to list the same env.
        matches: List = []
        for base_uri in SpaceURI._thread_local.base_uris:
            base_path = SpaceURI._file_uri_to_path(base_uri)
            if base_path is None or not base_path.exists():
                continue
            for cand in base_path.rglob(f"{name}/{STEP_FILE_NAME}"):
                if not cand.is_file():
                    continue
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                except (OSError, yaml.YAMLError):
                    continue
                if not isinstance(data, dict):
                    continue
                env_keys = list((data.get("environment_configs") or {}).keys())
                if env_class in env_keys:
                    matches.append((len(env_keys), str(cand), cand))
        if not matches:
            return None
        # Sort: specificity first (fewer env_configs entries = more specific),
        # then lexicographic path for deterministic tie-break.
        matches.sort(key=lambda m: (m[0], m[1]))
        cand = matches[0][2]
        target = cand.parent if not rest else cand.parent / rest
        if not target.exists():
            return None
        return URI.get_uri(  # type: ignore[return-value]
            f"file://{target}",
            "file",
            secrets=SpaceURI._thread_local.space_secrets,
        )

    @staticmethod
    def _file_uri_to_path(base_uri: str) -> Optional[Path]:
        """Return the local filesystem path for a `file://` base URI, else None.

        Non-file schemes (git://, http://, etc.) and the bare ``file:`` form
        return None — those bases don't support glob.  Strips a single leading
        slash duplicate when both ``file://`` and an absolute path are present.
        """
        parsed = urlparse(base_uri)
        if parsed.scheme not in ("", "file"):
            return None
        # urlparse turns `file:///abs/path` into netloc='', path='/abs/path' and
        # `file:///` into the same.  `file:` (no slashes) yields path=''.
        path_str = (parsed.netloc or "") + (parsed.path or "")
        if not path_str:
            return None
        p = Path(path_str)
        return p if p.is_absolute() else None

    @staticmethod
    def _build_candidates(uri_suffix: str) -> List[str]:
        """Return the ordered list of suffixes to try against each base_uri.

        For ``space://steps/<rest>`` the active env's step_type chain (from the
        thread-local) is prepended so the resolver tries the most-preferred
        step_type-specific path first, falling back through the chain to the
        env-agnostic ``steps/<rest>`` location.  All other URI prefixes pass
        through unchanged so this only affects step lookups.
        """
        if not uri_suffix.startswith(STEPS_PREFIX):
            return [uri_suffix]
        step_types: List[str] = (
            getattr(SpaceURI._thread_local, "current_step_types", None) or []
        )
        rest = uri_suffix[len(STEPS_PREFIX):]
        candidates = [f"{STEPS_PREFIX}{st}/{rest}" for st in step_types]
        candidates.append(uri_suffix)  # env-agnostic fallback
        return candidates

    @classmethod
    def set_baseuris(cls, base_uris: List[str], space_secrets: dict):
        cls._thread_local.space_secrets = space_secrets
        cls._thread_local.base_uris = base_uris

    @classmethod
    @contextmanager
    def with_current_env_class_name(
        cls, env_class_name: Optional[str]
    ) -> Iterator[None]:
        """Scope just the active env's class name on the thread-local for the
        duration of the ``with`` block.

        Used by code paths that resolve ``space://steps/<name>`` URIs without a
        full ``Environment`` instance available — notably the build-creation-time
        validator in :class:`gbserver.build.build.Build`, which knows each
        target's ``environment_uri`` but doesn't instantiate the env.  The
        resolver's env-class-match tier needs ``current_env_class_name`` set to
        find env-keyed step variants under ``<base>/.../<name>/step.yaml``.

        Saves and restores any previous value so nested or sibling validation
        in the same thread doesn't leak.  ``None`` or empty string means "no
        env-class narrowing".

        Args:
            env_class_name: The env's class name (e.g. ``"K8s"``, ``"Skypilot"``,
                ``"Docker"``).  Pass ``None`` or an empty string to opt out of
                narrowing.
        """
        prev = getattr(cls._thread_local, "current_env_class_name", None)
        cls._thread_local.current_env_class_name = env_class_name or None
        try:
            yield
        finally:
            if prev is None:
                if hasattr(cls._thread_local, "current_env_class_name"):
                    del cls._thread_local.current_env_class_name
            else:
                cls._thread_local.current_env_class_name = prev

    @classmethod
    @contextmanager
    def with_current_env_step_types(
        cls, step_types: Optional[List[str]]
    ) -> Iterator[None]:
        """Scope a step_type chain on the thread-local for the duration of the
        ``with`` block.

        Saves and restores any previous value so that nested or sibling target
        processing in the same thread doesn't leak.  ``None`` or an empty list
        means "no step_type-narrowing" — the resolver behaves as if the field
        had never been set.

        Args:
            step_types: Ordered chain of step_type strings, most-preferred first.
                Pass ``None`` or an empty list to opt out of narrowing.
        """
        prev = getattr(cls._thread_local, "current_step_types", None)
        cls._thread_local.current_step_types = list(step_types) if step_types else None
        try:
            yield
        finally:
            if prev is None:
                if hasattr(cls._thread_local, "current_step_types"):
                    del cls._thread_local.current_step_types
            else:
                cls._thread_local.current_step_types = prev

    @classmethod
    @contextmanager
    def with_current_env(cls, environment) -> Iterator[None]:
        """Scope the active env's step-discovery context on the thread-local
        for the duration of the ``with`` block.

        Sets three thread-local fields used by ``SpaceURI.__new__`` when
        resolving ``space://steps/<name>`` URIs:

        * ``current_env_dir_uri``     ← ``environment.environment_dir_uri``
        * ``current_step_types``      ← ``environment.step_type_chain``
        * ``current_env_class_name``  ← ``environment.__class__.__name__``

        Step lookups consult, in order:

        1. ``<environment.environment_dir_uri>/steps/<name>/`` — env-co-located
           steps shipped inside the env's own directory.
        2. Recursive glob ``<base>/**/<name>/step.yaml`` — first candidate whose
           ``environment_configs`` keys contain the active env's class name
           (e.g. ``K8s``, ``Skypilot``).  Subdirectory naming is conventional
           only; the match is by step.yaml content.
        3. ``<base>/steps/<step_type>/<name>/`` for each step_type in the env's
           chain — cross-env-class step_type pools.
        4. ``<base>/steps/<name>/`` — env-agnostic fallback.

        Prior values are saved on enter and restored on exit so nested or
        sibling target processing in the same thread doesn't leak.

        Args:
            environment: The active target's ``Environment`` instance.  Reads
                ``environment.step_type_chain``,
                ``environment.environment_dir_uri``, and the instance class.
        """
        prev_steps = getattr(cls._thread_local, "current_step_types", None)
        prev_dir = getattr(cls._thread_local, "current_env_dir_uri", None)
        prev_class = getattr(cls._thread_local, "current_env_class_name", None)
        chain = getattr(environment, "step_type_chain", None) or None
        env_dir = getattr(environment, "environment_dir_uri", None)
        env_class = environment.__class__.__name__ if environment is not None else None
        cls._thread_local.current_step_types = list(chain) if chain else None
        cls._thread_local.current_env_dir_uri = env_dir
        cls._thread_local.current_env_class_name = env_class
        try:
            yield
        finally:
            if prev_steps is None:
                if hasattr(cls._thread_local, "current_step_types"):
                    del cls._thread_local.current_step_types
            else:
                cls._thread_local.current_step_types = prev_steps
            if prev_dir is None:
                if hasattr(cls._thread_local, "current_env_dir_uri"):
                    del cls._thread_local.current_env_dir_uri
            else:
                cls._thread_local.current_env_dir_uri = prev_dir
            if prev_class is None:
                if hasattr(cls._thread_local, "current_env_class_name"):
                    del cls._thread_local.current_env_class_name
            else:
                cls._thread_local.current_env_class_name = prev_class

    @staticmethod
    def get_supported_schemes() -> List[str]:
        """Return supported uri schemes as list"""
        return [GBSPACE_SCHEME, SPACE_SCHEME]

    def exists(self: Self, force: bool = False) -> bool:
        """# TODO: fix this"""
        return True  # TODO: fix this

    def is_accessible(self) -> bool:
        """# TODO: fix this"""
        return True  # TODO: fix this

    def pull(self: Self, dest: Path, force: bool = False) -> bool:
        """# TODO: fix this"""
        return True  # TODO: fix this

    def delete(self: Self) -> bool:
        raise NotImplementedError("SpaceURI delete is not implemented")
