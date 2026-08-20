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
from typing import Iterator, List, Optional, Self, Tuple
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
        # Tier 1: for `space://steps/<name>` with an active env, first honor the
        # space's own root step (`base_uris[0]/steps/<name>`, highest priority),
        # then the env-co-located ancestor-walk (nearest-wins), bounded by the
        # enclosing base_uri.  Each candidate must be admissible for the active
        # env: a step declaring ``environment_configs`` must list the active env
        # class and satisfy its ``subtypes`` restriction, else it is skipped and
        # resolution continues (a step scoped to other classes never wins here).
        if uri_suffix.startswith(STEPS_PREFIX):
            walked = SpaceURI._walk_colocated_steps(uri_suffix)
            if walked is not None:
                return walked  # type: ignore[return-value]
        # Tier 2: env-class-match.  Recursively glob all `<base>/**/<name>/step.yaml`
        # files; pick the first (lexicographic) candidate whose `environment_configs`
        # keys contain the active env's class name.  Sub-asset URIs of the form
        # `space://steps/<name>/<rest>` re-use the matched dir.
        match = SpaceURI._try_env_class_match(uri_suffix)
        if match is not None:
            return match  # type: ignore[return-value]
        # Tier 3: env-agnostic fallback against the space's own base_uris.
        # Existence is checked via the resolved URI's own scheme-aware
        # ``exists()`` (so ``file://`` and ``git+ssh://`` bases both resolve, and
        # the resolved URI — possibly a git URI pulled later — is returned as-is).
        # For a `steps/<name>[/<rest>]` URI, `_fallback_steps_ok` additionally
        # enforces the `<rest>` containment guard and the `subtypes` restriction
        # against the base's local materialization (the local dir, or the reused
        # git clone), so the restriction is never bypassed by falling through to
        # the fallback.  Bases that can't be materialized locally are admitted
        # (as before sub-types existed).
        parsed = SpaceURI._parse_step_name_rest(uri_suffix)
        for base_uri in SpaceURI._thread_local.base_uris:
            resolved = URI.get_uri(
                base_uri, "file", secrets=SpaceURI._thread_local.space_secrets
            )
            resolved.append_path(uri_suffix)
            if not resolved.exists():
                continue
            if parsed is None or SpaceURI._fallback_steps_ok(base_uri, parsed):
                return resolved  # type: ignore[return-value]
        raise ValueError(f"Unresolvable space uri : {uristr}")

    @staticmethod
    def _parse_step_name_rest(uri_suffix: str) -> Optional[Tuple[str, str]]:
        """Split a ``steps/<name>[/<rest>]`` suffix into ``(name, rest)``.

        Args:
            uri_suffix: The scheme-stripped URI suffix (e.g. ``steps/digit`` or
                ``steps/digit/helm-charts``).

        Returns:
            ``(name, rest)`` where ``rest`` is the empty string for a bare step
            URI, or ``None`` when the suffix is not a ``steps/`` lookup or
            carries no step name.
        """
        if not uri_suffix.startswith(STEPS_PREFIX):
            return None
        after = uri_suffix[len(STEPS_PREFIX) :]
        name, _, rest = after.partition("/")
        if not name:
            return None
        return name, rest

    @staticmethod
    def _step_uri_from_dir(step_dir: Path, rest: str) -> Optional[URI]:
        """Build a resolved file URI for a matched step dir plus ``rest`` suffix.

        The ``rest`` sub-asset path must stay within ``step_dir``: a ``rest`` that
        escapes the step dir (e.g. ``../../secret`` from a
        ``space://steps/<name>/../../secret`` URI) is rejected with ``None``.
        ``space://`` URIs come from trusted config so this is defence-in-depth,
        but the containment check is cheap insurance against a traversal.

        Args:
            step_dir: The directory containing the matched ``step.yaml``.
            rest: Sub-asset path appended to ``step_dir`` (empty for a bare URI).

        Returns:
            The resolved ``URI`` when the target exists and stays under
            ``step_dir``, else ``None``.
        """
        target = step_dir if not rest else step_dir / rest
        if rest:
            base = step_dir.resolve()
            resolved = target.resolve()
            if resolved != base and base not in resolved.parents:
                return None
        if not target.exists():
            return None
        return URI.get_uri(  # type: ignore[return-value]
            f"file://{target}",
            "file",
            secrets=SpaceURI._thread_local.space_secrets,
        )

    @staticmethod
    def _enclosing_base_boundary(env_path: Path) -> Path:
        """Return the deepest base_uri directory enclosing ``env_path``.

        The ancestor-walk never ascends above this boundary, so step lookups stay
        within the base_uri subtree the env lives under.  Falls back to
        ``env_path`` itself when no base_uri encloses it (the walk then
        degenerates to the env's own directory).  Git base_uris are materialized
        to their reused local clone via :meth:`_uri_to_local_path`, so a
        git-backed env whose steps live in the same repo is bounded correctly.

        Args:
            env_path: Absolute, resolved path of the active env's directory.

        Returns:
            The boundary directory (inclusive stop point for the walk).
        """
        boundary = env_path
        best_len = -1
        for base_uri in SpaceURI._thread_local.base_uris:
            base_path = SpaceURI._uri_to_local_path(base_uri)
            if base_path is None:
                continue
            base_path = base_path.resolve()
            if (env_path == base_path or base_path in env_path.parents) and len(
                str(base_path)
            ) > best_len:
                boundary = base_path
                best_len = len(str(base_path))
        return boundary

    @staticmethod
    def _matching_env_key(configs, env_class: str) -> Optional[str]:
        """Return the ``environment_configs`` key that matches ``env_class``.

        Env class names (``K8s``, ``Skypilot``, ...) and the ``environment_configs``
        keys authored in ``step.yaml`` don't always agree on case (e.g. a step
        keyed ``k8s`` for the ``K8s`` env class), so the match is by
        case-insensitive string equality.  An exact match is preferred; a
        case-insensitive match is the fallback.  The lookup is on the **key**
        only and ignores the entry's value, so a present-but-null entry
        (``{Skypilot:}``) still reports its key.

        Args:
            configs: The ``environment_configs`` mapping (any non-dict value,
                including ``None``, yields ``None``).
            env_class: Active env class name.

        Returns:
            The matching key, or ``None`` when no key matches.
        """
        if not isinstance(configs, dict):
            return None
        if env_class in configs:
            return env_class
        lowered = env_class.lower()
        for key in configs:
            if isinstance(key, str) and key.lower() == lowered:
                return key
        return None

    @staticmethod
    def _env_config_entry(data: dict, env_class: str):
        """Look up the value of ``environment_configs[<env_class>]``.

        Uses :meth:`_matching_env_key` for the case-insensitive key match.

        Args:
            data: Parsed ``step.yaml`` mapping.
            env_class: Active env class name.

        Returns:
            The matching ``environment_configs`` value, or ``None`` when no key
            matches **or the matched key's value is null**.  Because a null value
            is indistinguishable from an absent key here, callers testing whether
            the class is *declared* must use :meth:`_env_class_present`, not an
            ``is None`` check on this result.
        """
        configs = data.get("environment_configs")
        if not isinstance(configs, dict):
            return None
        key = SpaceURI._matching_env_key(configs, env_class)
        return configs[key] if key is not None else None

    @staticmethod
    def _env_class_present(data: dict, env_class: str) -> bool:
        """Return whether ``environment_configs`` *declares* ``env_class`` by key.

        Presence is decided by the key alone, so a class that is present but null
        (``environment_configs: {Skypilot:}`` — declared, hence admissible under
        the active ``Skypilot`` env) is distinguished from one that is genuinely
        absent (the step is scoped to other classes only).

        Args:
            data: Parsed ``step.yaml`` mapping.
            env_class: Active env class name.

        Returns:
            ``True`` when a key matches ``env_class`` (case-insensitively).
        """
        return (
            SpaceURI._matching_env_key(data.get("environment_configs"), env_class)
            is not None
        )

    @staticmethod
    def _subtype_ok(
        data: dict, env_class: Optional[str], env_subtype: Optional[str]
    ) -> bool:
        """Return whether a step's sub-type restriction admits the active env.

        Pure ``environment_configs`` logic (no class-presence requirement — that
        is the caller's concern).  For the active env's class:

        * no class entry, or an **empty** ``subtypes`` list → no restriction
          (universal): returns ``True``;
        * a **non-empty** ``subtypes`` list → returns ``True`` only when
          ``env_subtype`` is one of its entries (exact string equality).  An env
          with no sub-type therefore never satisfies a step that lists sub-types.

        A scalar ``subtypes: kubernetes`` is normalized to a single-element list
        so membership stays an exact match; a value that is neither a list nor a
        string is uninterpretable and treated as no restriction.

        When ``env_class`` is unknown there is no class context to filter on, so
        the step is admitted (preserves directory-only ancestor-walk behavior).

        Args:
            data: Parsed ``step.yaml`` mapping.
            env_class: Active env class name (e.g. ``"Skypilot"``), or ``None``.
            env_subtype: Active env sub-type (e.g. ``"kubernetes"``), or ``None``.
        """
        if not env_class:
            return True
        entry = SpaceURI._env_config_entry(data, env_class)
        if not isinstance(entry, dict):
            return True
        subtypes = entry.get("subtypes") or []
        if isinstance(subtypes, str):
            # A scalar `subtypes: kubernetes` is a single sub-type, not an
            # iterable of characters — wrap it so membership stays an exact
            # string match rather than a substring test (``"k" in "kubernetes"``).
            subtypes = [subtypes]
        if not isinstance(subtypes, list) or not subtypes:
            # Empty, or a non-list we can't interpret as sub-types → universal,
            # matching the module's "can't read the restriction → admit"
            # convention (see :meth:`_step_env_ok`).
            return True
        return env_subtype in subtypes

    @staticmethod
    def _env_ok(
        data: dict, env_class: Optional[str], env_subtype: Optional[str]
    ) -> bool:
        """Return whether a step's ``environment_configs`` admit the active env.

        This is the full directory-tier gate: a two-stage check that first
        matches the env **class**, then (via :meth:`_subtype_ok`) the sub-type.

        * ``env_class`` unknown → ``True`` (no class context to filter on, as in
          :meth:`_subtype_ok`).
        * a **non-empty** ``environment_configs`` block whose keys do **not**
          include the active class → ``False``: the step is scoped to other env
          classes and cannot run here, so it is not a match.  This is the
          class-presence requirement the sub-type-only predicate deliberately left
          to callers.
        * otherwise (the class **key** is present, **or** the step declares no
          ``environment_configs`` at all) → delegate to :meth:`_subtype_ok`.  A
          step with no ``environment_configs`` stays env-agnostic/universal,
          preserving the directory-placed ancestor-walk behavior.

        Presence is decided by the class **key**, not its value: a present-but-null
        entry (``environment_configs: {Skypilot:}``) declares the class and so is
        admitted (:meth:`_subtype_ok` then treats the null entry as unrestricted).
        A value-based ``is None`` check would wrongly skip such a step, which is
        scoped to exactly the active class.

        Args:
            data: Parsed ``step.yaml`` mapping.
            env_class: Active env class name (e.g. ``"Skypilot"``), or ``None``.
            env_subtype: Active env sub-type (e.g. ``"kubernetes"``), or ``None``.
        """
        if not env_class:
            return True
        configs = data.get("environment_configs")
        if (
            isinstance(configs, dict)
            and configs
            and not SpaceURI._env_class_present(data, env_class)
        ):
            # Declared for other env classes only → not a match for this env.
            return False
        return SpaceURI._subtype_ok(data, env_class, env_subtype)

    @staticmethod
    def _step_env_ok(
        step_yaml: Path, env_class: Optional[str], env_subtype: Optional[str]
    ) -> bool:
        """Parse ``step_yaml`` and apply :meth:`_env_ok`.

        Applies the class-presence + sub-type gate: a step declaring an
        ``environment_configs`` block must list the active env class (and satisfy
        its sub-type restriction) to be admitted; a step with no
        ``environment_configs`` is env-agnostic and admitted.  A file that can't
        be read/parsed carries no restriction we can evaluate, so it is admitted
        (directory-only match, as before env/sub-type filtering).

        Args:
            step_yaml: Path to the candidate ``step.yaml``.
            env_class: Active env class name, or ``None``.
            env_subtype: Active env sub-type, or ``None``.
        """
        try:
            with open(step_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            return True
        if not isinstance(data, dict):
            return True
        return SpaceURI._env_ok(data, env_class, env_subtype)

    @staticmethod
    def _fallback_steps_ok(base_uri: str, parsed: Tuple[str, str]) -> bool:
        """Vet a Tier 3 ``steps/<name>[/<rest>]`` fallback hit against ``base_uri``.

        The caller has already confirmed the target exists via the resolved
        URI's scheme-aware ``exists()``.  This applies the two guards that need
        the base materialized as a local directory:

        * **containment** — a ``<rest>`` that escapes ``<base>/steps/<name>``
          (e.g. ``../../secret``) is rejected, matching Tiers 1 & 2;
        * **env** — the step's own ``step.yaml`` is read and its
          ``environment_configs`` must admit the active env: a step that
          declares that block must list the active env class (and satisfy its
          ``subtypes`` restriction, if any); a step with no
          ``environment_configs`` is env-agnostic and admitted.

        Reading the step's *own* ``step.yaml`` (resolved from ``<name>``, not
        from the possibly ``..``-displaced ``<rest>``) keeps the restriction
        from being bypassed by a crafted sub-asset path.  The base is resolved
        to a local root via :meth:`_base_uri_to_local_root` (a ``file://`` dir,
        or the reused git clone — no re-clone).  A base that can't be
        materialized locally is admitted, preserving pre-sub-type behavior.

        Args:
            base_uri: The base URI the target resolved against.
            parsed: ``(name, rest)`` from :meth:`_parse_step_name_rest`.

        Returns:
            ``True`` when the hit is admitted; ``False`` when ``rest`` escapes
            the step dir or the ``environment_configs`` restriction (class or
            sub-type) excludes the env.
        """
        name, rest = parsed
        root = SpaceURI._uri_to_local_path(base_uri)
        if root is None:
            return True
        step_dir = root / STEPS_PREFIX.rstrip("/") / name
        if rest:
            base = step_dir.resolve()
            resolved = (step_dir / rest).resolve()
            if resolved != base and base not in resolved.parents:
                return False
        env_class = getattr(SpaceURI._thread_local, "current_env_class_name", None)
        env_subtype = getattr(SpaceURI._thread_local, "current_env_subtype", None)
        return SpaceURI._step_env_ok(step_dir / STEP_FILE_NAME, env_class, env_subtype)

    @staticmethod
    def _space_root_step(
        name: str, rest: str, env_class: Optional[str], env_subtype: Optional[str]
    ) -> Optional[URI]:
        """Resolve ``steps/<name>`` against the space's own root (``base_uris[0]``).

        The space directory (the first base_uri — the space's own ``uristr``) is
        the most authoritative step source: a step it ships at
        ``<space>/steps/<name>`` overrides an env-co-located step or one inherited
        via ``base_uris[1:]`` (e.g. a published assets tree).  This lets a step
        being developed in its own space be exercised (by ``make test``) before it
        is published into an inherited tree.  The same env gate as the
        ancestor-walk is applied (:meth:`_step_env_ok`), so a space step that
        declares ``environment_configs`` without the active env class — or whose
        ``subtypes`` exclude the active env — is skipped and resolution falls
        through (it must never override a valid step with one that cannot run
        under the active environment).

        Args:
            name: Step name.
            rest: Sub-asset suffix appended to the step dir (empty for a bare URI).
            env_class: Active env class name (for the sub-type gate), or ``None``.
            env_subtype: Active env sub-type (for the sub-type gate), or ``None``.

        Returns:
            The resolved step ``URI`` when the space ships an admitting
            ``steps/<name>``, else ``None``.
        """
        base_uris = getattr(SpaceURI._thread_local, "base_uris", None)
        if not base_uris:
            return None
        space_root = SpaceURI._uri_to_local_path(base_uris[0])
        if space_root is None:
            return None
        step_yaml = space_root.resolve() / "steps" / name / STEP_FILE_NAME
        if not step_yaml.is_file() or not SpaceURI._step_env_ok(
            step_yaml, env_class, env_subtype
        ):
            return None
        return SpaceURI._step_uri_from_dir(step_yaml.parent, rest)

    @staticmethod
    def _walk_colocated_steps(uri_suffix: str) -> Optional[URI]:
        """Resolve ``space://steps/<name>`` by walking up from the active env dir.

        Before the walk, the space's own root step (``base_uris[0]/steps/<name>``,
        see :meth:`_space_root_step`) takes priority, so a step the space ships
        overrides an env-co-located or inherited copy of the same name.

        Starting at ``current_env_dir_uri`` the resolver checks
        ``<dir>/steps/<name>/step.yaml`` at each ancestor, nearest-wins, stopping
        at (and including) the deepest ``base_uri`` that encloses the env dir.
        This lets sibling environments under a common family directory share one
        step implementation while an env's own dir still overrides it.  A
        candidate whose ``environment_configs`` don't admit the active env — it
        declares that block without the active env class, or its ``subtypes``
        exclude the active env — is skipped and the walk continues upward
        (see :meth:`_step_env_ok`).

        The env dir URI is materialized to a local path via
        :meth:`_uri_to_local_path`, so a git-backed env resolves against its
        reused clone — the walk then works whenever the co-located steps live in
        the **same repo** (base_uri) as the env (cross-repo steps live in a
        separate clone and are not reachable by the walk).

        Args:
            uri_suffix: The scheme-stripped ``steps/<name>[/<rest>]`` suffix.

        Returns:
            The matched step ``URI`` (with any ``<rest>`` appended), or ``None``
            when there is no active/resolvable env dir or no ancestor within the
            boundary carries a matching step.
        """
        parsed = SpaceURI._parse_step_name_rest(uri_suffix)
        if parsed is None:
            return None
        name, rest = parsed
        env_dir_uri = getattr(SpaceURI._thread_local, "current_env_dir_uri", None)
        if not env_dir_uri:
            return None
        env_path = SpaceURI._uri_to_local_path(env_dir_uri)
        if env_path is None:
            return None
        env_path = env_path.resolve()
        env_class = getattr(SpaceURI._thread_local, "current_env_class_name", None)
        env_subtype = getattr(SpaceURI._thread_local, "current_env_subtype", None)
        # Space-root priority: a step the space itself ships at
        # ``<space>/steps/<name>`` overrides any env-co-located or inherited copy,
        # so a locally-developed step is exercised before it is published.
        space_hit = SpaceURI._space_root_step(name, rest, env_class, env_subtype)
        if space_hit is not None:
            return space_hit
        boundary = SpaceURI._enclosing_base_boundary(env_path)
        cur = env_path
        while True:
            step_yaml = cur / "steps" / name / STEP_FILE_NAME
            if step_yaml.is_file() and SpaceURI._step_env_ok(
                step_yaml, env_class, env_subtype
            ):
                found = SpaceURI._step_uri_from_dir(step_yaml.parent, rest)
                if found is not None:
                    return found
            if cur in (boundary, cur.parent):
                return None
            cur = cur.parent

    @staticmethod
    def _try_env_class_match(uri_suffix: str) -> Optional[URI]:
        """Resolve `space://steps/<name>[/<rest>]` by env-class metadata match.

        Recursively scans every base_uri (``file://`` dirs and git bases, the
        latter via their reused local clone) for ``<name>/step.yaml`` files,
        parses each candidate's ``environment_configs`` keys, and returns the
        first (lexicographically) whose keys contain the active env's class name
        (case-insensitively; set via :meth:`with_current_env`) **and** whose
        per-class ``subtypes`` restriction (if any) admits the active env's
        sub-type.  The
        directory of the matched step.yaml is used as the resolution result; for
        sub-asset URIs the ``<rest>`` portion is appended to that directory.

        Returns ``None`` when:
          * the URI is not a `space://steps/...` lookup;
          * no active env class is set on the thread-local;
          * no candidate step.yaml lists the active env's class (and satisfies
            its sub-type restriction).
        Callers fall through to the legacy resolver tiers in that case.
        """
        parsed = SpaceURI._parse_step_name_rest(uri_suffix)
        if parsed is None:
            return None
        name, rest = parsed
        env_class: Optional[str] = getattr(
            SpaceURI._thread_local, "current_env_class_name", None
        )
        if not env_class:
            return None
        env_subtype: Optional[str] = getattr(
            SpaceURI._thread_local, "current_env_subtype", None
        )
        # Collect (specificity, path-tiebreaker, candidate-path) for every
        # step.yaml whose env_configs lists the active env class and satisfies
        # its sub-type restriction.  Specificity is the count of env_configs keys
        # — the smaller, the more env-specific the file is.  We prefer the most
        # specific match so a single-env split file beats a multi-env catch-all
        # that happens to list the same env.
        matches: List = []
        for base_uri in SpaceURI._thread_local.base_uris:
            base_path = SpaceURI._uri_to_local_path(base_uri)
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
                # Class-presence by key (not value): a present-but-null entry
                # (``{Skypilot:}``) is still declared for the active class and must
                # match here, matching the _env_ok tier.  Testing
                # ``_env_config_entry(...) is not None`` would silently drop it (its
                # value is null), leaving the two tiers disagreeing on such steps.
                if SpaceURI._env_class_present(
                    data, env_class
                ) and SpaceURI._subtype_ok(data, env_class, env_subtype):
                    matches.append((len(env_keys), str(cand), cand))
        if not matches:
            return None
        # Sort: specificity first (fewer env_configs entries = more specific),
        # then lexicographic path for deterministic tie-break.
        matches.sort(key=lambda m: (m[0], m[1]))
        cand = matches[0][2]
        # Route through _step_uri_from_dir so the ``rest`` containment guard
        # (reject sub-asset paths that escape the step dir) applies here too.
        return SpaceURI._step_uri_from_dir(cand.parent, rest)

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
    def _uri_to_local_path(uri_str: str) -> Optional[Path]:
        """Return a local filesystem path for a space URI, materializing git.

        Handles both ``base_uris`` entries and the active env's directory URI:

        * Local (``file://``) → its directory (via :meth:`_file_uri_to_path`).
        * Git (``git+ssh``/``git+https``/...) → the **already-cloned** local path
          via ``GitURI.get_path_in_repo_from_cache`` — the same thread-local
          cache the resolved URI's ``exists()``/env-asset load populates, so no
          second clone happens.  A ``#subdirectory=`` fragment is honored (the
          env dir URI carries one), so this returns the repo root for a bare
          repo base and ``<clone>/<subdir>`` for a sub-path.

        This lets the ancestor-walk (Tier 1), env-class-match glob (Tier 2) and
        the fallback guards (Tier 3) operate on local files for git-backed
        spaces.  Non-local, non-git URIs (or a git URI whose clone fails) return
        ``None`` — callers treat that as "can't inspect" and skip/admit.

        Args:
            uri_str: A ``base_uris`` entry or the active env's directory URI.

        Returns:
            A local ``Path``, or ``None`` when it can't be materialized locally.
        """
        local = SpaceURI._file_uri_to_path(uri_str)
        if local is not None:
            return local
        secrets = getattr(SpaceURI._thread_local, "space_secrets", None)
        resolved = URI.get_uri(uri_str, "file", secrets=secrets)
        get_path_in_repo = getattr(resolved, "get_path_in_repo_from_cache", None)
        if get_path_in_repo is None:
            return None
        try:
            return get_path_in_repo()
        except Exception:  # pylint: disable=broad-except
            # A clone failure (network/auth) is non-fatal here: fall back to
            # "can't inspect" so resolution degrades to path-existence only.
            return None

    @classmethod
    def set_baseuris(cls, base_uris: List[str], space_secrets: dict):
        cls._thread_local.space_secrets = space_secrets
        cls._thread_local.base_uris = base_uris

    @classmethod
    @contextmanager
    def _scope_thread_local(cls, **values) -> Iterator[None]:
        """Set thread-local attrs for the block, restoring prior values on exit.

        Each keyword sets ``cls._thread_local.<key>`` to its value; a falsy value
        clears the attr instead (treated as "unset").  Prior values are saved on
        enter and restored on exit so nested or sibling scopes in the same
        thread don't leak.

        Args:
            **values: Thread-local attribute names mapped to the value to scope.
        """
        saved = {k: getattr(cls._thread_local, k, None) for k in values}
        for key, value in values.items():
            setattr(cls._thread_local, key, value or None)
        try:
            yield
        finally:
            for key, prev in saved.items():
                if prev is None:
                    if hasattr(cls._thread_local, key):
                        delattr(cls._thread_local, key)
                else:
                    setattr(cls._thread_local, key, prev)

    @staticmethod
    def _read_env_subtype(environment) -> Optional[str]:
        """Return an environment's ``subtype``, or ``None``.

        Reads ``environment.config.subtype`` defensively so stand-in
        environments without a full ``EnvironmentConfig`` (e.g. some unit tests)
        simply contribute no sub-type.

        Args:
            environment: The active target's ``Environment`` instance (or None).
        """
        if environment is None:
            return None
        config = getattr(environment, "config", None)
        return getattr(config, "subtype", None)

    @classmethod
    @contextmanager
    def with_current_env_class_name(
        cls,
        env_class_name: Optional[str],
        env_dir_uri: Optional[str] = None,
        env_subtype: Optional[str] = None,
    ) -> Iterator[None]:
        """Scope the active env's step-discovery context from loose values.

        Used by code paths that resolve ``space://steps/<name>`` URIs without a
        full ``Environment`` instance available — notably the build-creation-time
        validator in :class:`gbserver.build.build.Build`, which knows each
        target's ``environment_uri`` (and can read its ``environment.yaml``) but
        doesn't instantiate the env.  Passing ``env_dir_uri`` and ``env_subtype``
        lets the ancestor-walk (Tier 1) and the sub-type filter run during
        validation, not just the bare env-class-match.

        Saves and restores any previous values so nested or sibling validation
        in the same thread doesn't leak.  Falsy values opt out of that field.

        Args:
            env_class_name: The env's class name (e.g. ``"K8s"``, ``"Skypilot"``).
            env_dir_uri: The env's directory URI for the Tier 1 ancestor-walk.
            env_subtype: The env's sub-type for the sub-type filter.
        """
        with cls._scope_thread_local(
            current_env_class_name=env_class_name,
            current_env_dir_uri=env_dir_uri,
            current_env_subtype=env_subtype,
        ):
            yield

    @classmethod
    @contextmanager
    def with_current_env(cls, environment) -> Iterator[None]:
        """Scope the active env's step-discovery context on the thread-local
        for the duration of the ``with`` block.

        Sets three thread-local fields used by ``SpaceURI.__new__`` when
        resolving ``space://steps/<name>`` URIs:

        * ``current_env_dir_uri``     ← ``environment.environment_dir_uri``
        * ``current_env_class_name``  ← ``environment.__class__.__name__``
        * ``current_env_subtype``     ← ``environment.config.subtype``

        Step lookups consult, in order:

        1. ``<env-dir>`` up to the enclosing base_uri — env-co-located steps,
           nearest-wins (a step in the env's own dir overrides an ancestor's);
           a candidate whose ``subtypes`` restriction excludes the active env is
           skipped and the walk continues upward.
        2.  Recursive glob ``<base>/**/<name>/step.yaml`` — first candidate whose
            ``environment_configs`` keys contain the active env's class name and
            whose ``subtypes`` restriction admits the active env's sub-type.
        3.  ``<base>/steps/<name>/`` — env-agnostic fallback.

        Prior values are saved on enter and restored on exit so nested or
        sibling target processing in the same thread doesn't leak.

        Args:
            environment: The active target's ``Environment`` instance.  Reads
                ``environment_dir_uri``, the instance class, and the config's
                ``subtype``.
        """
        env_dir = getattr(environment, "environment_dir_uri", None)
        env_class = environment.__class__.__name__ if environment is not None else None
        subtype = SpaceURI._read_env_subtype(environment)
        with cls._scope_thread_local(
            current_env_dir_uri=env_dir,
            current_env_class_name=env_class,
            current_env_subtype=subtype,
        ):
            yield

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
