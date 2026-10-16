# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Application settings, loaded from the environment via pydantic-settings."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AuthProviderName = Literal["disabled", "api_key", "oidc", "session"]
"""``"session"`` backs the browser backend-for-frontend flow: a session JWT this
service mints and verifies itself, distinct from the bearer tokens "oidc" verifies
directly."""

ADMIN_ROLE = "admin"
"""The one role string that grants unscoped reads.

The counterpart to
:data:`autotunex.core.auth.disabled.STANDALONE_PROVIDER`: a privilege-bearing
string that decides an authorization outcome should exist once, not as a literal
in three modules. ``api.deps.get_principal`` compares ``users.role`` against it,
``core.auth.disabled`` compares ``standalone_role`` against it, and the validator
below accepts it.

It lives here, rather than in ``core/auth/`` beside ``STANDALONE_PROVIDER``,
purely to stay cycle-proof: ``core.auth.disabled`` already imports this module
for :class:`Settings`, so a constant in ``core/auth/`` that this module imported
back would break at import time the moment ``core/auth/__init__.py`` stops being
docstring-only. Do not "tidy" it across.

The comparison against it is case- and whitespace-sensitive, deliberately and
for now: ``users.role`` is a nullable free-text ``String(50)`` written by the
tuning pipeline, so ``"Admin"`` or ``" admin"`` silently resolves to a non-admin
principal. That fails closed — the caller sees only their own jobs — but it fails
silently, and normalizing it is a behaviour change that needs its own decision.
"""

_KNOWN_ROLES = frozenset({ADMIN_ROLE, "user"})

_DEFAULT_LOCAL_OUTPUT_DIR = Path("artifacts/local")
"""Sentinel default for ``Settings.local_output_dir``.

Held as a module constant so ``_default_local_output_dir`` can tell "the caller
left this at its default" (recompute it under ``artifact_dir``) apart from "the
caller pinned it deliberately" (leave it alone) with a single value equality,
rather than duplicating the literal in the field default and the validator.
"""


def _is_unset(value: str | None) -> bool:
    """Return whether *value* counts as unset: ``None``, empty, or whitespace-only.

    ``AUTOTUNEX_OIDC_AUDIENCE=`` in a ``.env`` parses to ``""``, which a bare
    ``if not value`` check already catches. But ``AUTOTUNEX_OIDC_AUDIENCE=" "``
    parses to a single space — truthy, so it slips past that same check — and
    no real ``aud``, ``iss``, or JWKS URI is ever whitespace. Treating it as
    unset closes the gap: a whitespace-only ``oidc_audience`` would otherwise
    reproduce the exact symptom the empty-string gate exists to catch, every
    token rejected with a uniform 401 and nothing in the configuration to
    explain it.
    """
    return value is None or not value.strip()


def _secret_value(value: SecretStr | None) -> str | None:
    """Unwrap a ``SecretStr`` so ``_is_unset`` can inspect it.

    A ``SecretStr`` object is truthy no matter what string it wraps, so
    passing one to ``_is_unset`` directly would never detect an empty or
    whitespace-only secret — the exact gap ``_is_unset`` exists to close,
    reopened for every ``SecretStr`` field.
    """
    return None if value is None else value.get_secret_value()


def _default_auth_providers() -> list[AuthProviderName]:
    """Return a fresh ``["disabled"]`` list.

    A plain ``lambda: ["disabled"]`` infers as ``list[str]`` under mypy
    strict, not ``list[AuthProviderName]`` — this function's explicit return
    annotation is what keeps the field's default well-typed.
    """
    return ["disabled"]


class Settings(BaseSettings):
    """Runtime configuration.

    Every field is overridable by an ``AUTOTUNEX_``-prefixed environment
    variable or an entry in a local ``.env`` file. See ``.env.example``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AUTOTUNEX_",
        extra="ignore",
    )

    # --- App -------------------------------------------------------------
    app_name: str = "AutoTuneX API"
    environment: Literal["dev", "test", "prod"] = "dev"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    api_prefix: str = "/api/v1"

    # --- Frontend (optional) ---------------------------------------------
    frontend_dir: Path | None = None
    """Directory of the built SvelteKit SPA to serve from this service.

    When set to an existing directory, the app mounts it at ``frontend_base_path``
    with SPA-fallback routing (any unmatched sub-path returns ``index.html``, which
    is how ``adapter-static``'s ``fallback: 'index.html'`` client router expects to
    be served). ``None`` (the default) serves API only — the UI is hosted
    separately. For a local checkout this points at ``src/ux/build``.
    """

    frontend_base_path: str = "/autotune"
    """URL prefix the SPA is mounted under; must match SvelteKit's ``kit.paths.base``.

    The build bakes this prefix into every asset URL (``/autotune/_app/...``), so it
    cannot be changed here alone — rebuild the UI with a matching ``base`` if you
    change it.
    """

    # --- Persistence -----------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./autotunex.db"
    database_echo: bool = False
    auto_create_schema: bool = True
    """Create tables on startup. Convenient for dev; use Alembic in prod."""

    database_ssl_ca: str | None = None
    """Path to a CA-certificate PEM enabling verified TLS to the database.

    Required for managed MySQL such as IBM Cloud Databases for MySQL, which
    accepts only TLS connections and presents a certificate signed by a private
    CA — download that CA from the deployment (``ibmcloud cdb
    deployment-cacertificate`` or the console) and point this at it. When set (and
    ``database_ssl_mode`` is left to derive), the async engine and Alembic verify
    the server against this CA (``check_hostname`` on, ``CERT_REQUIRED``); see
    ``db/session.build_ssl_context``. ``None`` (the default) sends no TLS
    parameters, which is correct for local SQLite/Postgres/MySQL. Applied to
    MySQL URLs only.
    """

    database_ssl_mode: Literal["disable", "require", "verify"] | None = None
    """How to negotiate TLS to the database (MySQL URLs only).

    - ``"disable"`` — no TLS at all (local SQLite/Postgres/MySQL).
    - ``"require"`` — encrypt the connection but do **not** verify the server's
      certificate. No ``database_ssl_ca`` needed. This is the "connect without a
      certificate" path for managed MySQL such as IBM Cloud Databases, which
      refuses plaintext auth but accepts unverified TLS. It encrypts traffic but
      cannot detect a man-in-the-middle presenting a different cert — acceptable
      for a dev box against a known endpoint, a downgrade for production.
    - ``"verify"`` — encrypt **and** verify the server against ``database_ssl_ca``
      (``check_hostname`` on, ``CERT_REQUIRED``); requires the CA to be set.

    ``None`` (the default) derives the mode from ``database_ssl_ca``: ``"verify"``
    when a CA is configured, ``"disable"`` otherwise — preserving the historical
    behaviour where a lone CA path meant verified TLS.
    """

    # --- Connection pool (server databases only) -------------------------
    # These knobs matter against managed MySQL (IBM Cloud Databases), where every
    # *new* connection pays a full TLS + caching_sha2_password handshake worth
    # seconds. SQLite's async engine uses a non-queue pool and ignores the sizing
    # knobs (see db/session.build_pool_kwargs), so these are inert for local dev
    # and the test suite. See docs/superpowers/specs — the connection-pool tuning
    # was added after prod GET /configurations and /jobs latencies of 7-8s traced
    # to cold reconnects and pool contention with the reconcile loop.
    database_pool_size: int = Field(default=10, ge=1)
    """Connections kept open and reused per worker process (``QueuePool`` base size).

    Each pooled connection is *warm*: a request that reuses one skips the multi-second
    TLS + ``caching_sha2_password`` handshake a cold connection to managed MySQL costs.
    Size this to cover concurrent request handlers **plus** the background reconcile
    loop (:attr:`job_reconcile_concurrency`), which draws from this same pool when
    ``job_backend="llmb"``. Undersized, a reconcile sweep checks out every warm
    connection and pushes request handlers onto cold :attr:`database_max_overflow`
    connections — the exact contention behind the observed latency spikes. The
    default of 10 covers the reconcile loop's 5 with headroom to spare; raise it for
    higher request concurrency, but stay within the database plan's connection limit.
    Ignored for SQLite.
    """

    database_max_overflow: int = Field(default=5, ge=0)
    """Extra connections opened on demand beyond :attr:`database_pool_size` (``QueuePool``).

    The hard ceiling is ``database_pool_size + database_max_overflow``. Overflow
    connections are opened cold — each paying the full handshake — and discarded when
    returned rather than kept warm, so they are the expensive path. Keep
    :attr:`database_pool_size` large enough that steady-state traffic rarely reaches
    overflow; this is headroom for bursts, not the normal operating point. Ignored for
    SQLite.
    """

    database_pool_timeout_seconds: float = Field(default=30.0, gt=0)
    """Seconds a request waits for a free connection before failing (``QueuePool``).

    When all ``database_pool_size + database_max_overflow`` connections are checked
    out, a further request blocks up to this long and then raises ``TimeoutError``
    rather than hanging indefinitely — bounding the worst case when the pool is
    saturated (e.g. a slow reconcile sweep holding connections). Ignored for SQLite.
    """

    database_pool_recycle_seconds: int = Field(default=1800, ge=-1)
    """Recycle a pooled connection older than this many seconds; ``-1`` disables it.

    Managed MySQL closes idle server-side connections after its ``wait_timeout``
    (IBM Cloud Databases defaults to 3600s). A connection the server has already
    closed but the pool still holds yields a "MySQL server has gone away" error — or,
    caught by :attr:`database_pool_pre_ping`, a silent full reconnect — on its next
    use. Recycling on our own schedule, comfortably under the server's
    ``wait_timeout`` (30 min by default), refreshes connections before the server
    kills them. Applied to server databases only.
    """

    database_pool_pre_ping: bool = True
    """Liveness-check a pooled connection on checkout, reconnecting if it is dead.

    Costs one extra round trip per checkout but prevents handing a request a
    connection the server has silently closed. Left on by default: against a managed
    DB that recycles idle connections, the ping's round trip is far cheaper than the
    failed query plus reconnect it prevents. :attr:`database_pool_recycle_seconds`
    reduces how often the ping finds a dead connection but does not replace it — a
    connection can still die before reaching its recycle age. Applies to every pool,
    SQLite included (where it is a cheap local no-op).
    """

    database_pool_use_lifo: bool = True
    """Hand out the most-recently-used connection first (LIFO) instead of round-robin.

    Under bursty, sporadic traffic — this service's real pattern — LIFO keeps a small
    set of connections genuinely hot while letting the rest age out to be recycled or
    pre-ping-refreshed, rather than round-robin spreading every request across all
    connections so each drifts toward staleness. Directly reduces how often a request
    lands on a cold or server-closed connection. Ignored for SQLite.
    """

    # --- Storage ---------------------------------------------------------
    artifact_dir: Path = Path("./artifacts")
    """Where trained adapters and checkpoints are written."""

    # --- Datasets & storage ------------------------------------------------
    dataset_storage_dir: Path = Path("artifacts/datasets")
    """Root for persisted local dataset files; staging lives under ``.staging``.

    Its relationship to ``artifact_dir`` is deliberately loose (see the design
    spec's open item): today a sibling default under ``artifacts/``.
    """

    dataset_upload_max_bytes: int = Field(default=5 * 1024**3, ge=1)
    """Hard cap on a single uploaded file, enforced while streaming (413)."""

    dataset_upload_max_concurrent: int = Field(default=2, ge=1)
    """Max dataset uploads processed concurrently in-process (WS2 backpressure).

    The upload runner holds one slot for the whole of a dataset's off-request
    processing — count/split/remap AND the storage ``persist`` (the ``llmb``
    HuggingFace push, when active). Over-limit uploads queue on the slot rather
    than erroring; the ``202`` has already been returned. Bounds peak memory and
    concurrent pushes in the single API process. Not multi-replica-safe — a
    resource heuristic, not a correctness lock (a real queue would supersede it).
    """

    dataset_processing_timeout_seconds: float = Field(default=3600.0, gt=0)
    """Backstop wall-clock timeout for one dataset's off-request processing.

    Wraps the runner's count/split/remap/persist. On expiry the dataset is driven
    to ``error`` rather than left ``uploading`` forever. This cannot kill a
    blocking ``llmb`` subprocess mid-push (bounded separately by
    ``dataset_push_timeout_seconds``); it is the backstop for the CPU/IO phases.
    """

    dataset_push_timeout_seconds: float = Field(default=1800.0, gt=0)
    """Timeout for each ``llmb`` subprocess (auth + artifact push) in the HF backend.

    Passed to ``subprocess.run(timeout=...)`` so a hung push is actually killed —
    unlike ``dataset_processing_timeout_seconds``, whose ``asyncio`` cancellation
    cannot stop a blocking subprocess running in a worker thread.
    """

    dataset_client_gzip_enabled: bool = True
    """Whether the frontend should gzip-compress compressible dataset uploads.

    Surfaced to the browser via ``GET /api/v1/app-config`` — the frontend has
    no other channel to learn this. Only affects text formats (jsonl/json/csv);
    the frontend skips parquet regardless, since it is already internally
    compressed and gzip would only burn CPU for near-zero size reduction.
    """

    dataset_client_gzip_min_bytes: int = Field(default=1024**2, ge=0)
    """Skip client-side gzip below this size — not worth the CPU. See above."""

    dataset_client_parquet_preview_max_bytes: int = Field(default=100 * 1024**2, ge=1)
    """Byte threshold above which the frontend skips a local Parquet preview.

    Surfaced the same way as the two settings above.
    """

    dataset_storage_backend: Literal["auto", "local", "huggingface"] = "auto"
    """``"auto"`` resolves to ``local`` whenever ``gb_environment`` is
    ``standalone`` (where ``llmb artifact push`` is unavailable), regardless of
    tokens; otherwise it resolves to ``huggingface`` when the ``llmb`` tooling
    and **both** token env vars (``gb_token_env`` + ``hf_token_env``) are
    present, and ``local`` if not. ``"local"``/``"huggingface"`` force it — a
    forced ``huggingface`` missing either token is refused at startup."""

    llmb_command: str = "llmb"
    """The ``llmb build`` CLI binary name or path (HuggingFace push)."""

    hf_token_env: str = "HF_TOKEN"
    """Name of the env var holding the HF token that ``llmb build`` reads.

    Only the env var's *presence* is checked here and at backend selection; the
    token value is never loaded into settings — it is passed to the ``llmb``
    subprocess environment untouched.
    """

    hf_namespace: str | None = None
    """Optional HF org/namespace prefix for the derived repo name."""

    hf_preview_enabled: bool = True
    """Enable dataset preview via the HuggingFace dataset viewer.

    When off, HuggingFace-stored datasets return an empty preview. Local-stored
    datasets are unaffected — they read rows from disk regardless."""

    hf_viewer_base_url: str = "https://datasets-server.huggingface.co"
    """Base URL of the HuggingFace dataset-viewer service; the client appends
    ``/splits`` (to discover the repo's config name and readiness) and ``/rows``.
    Override to point at a mirror or an enterprise viewer."""

    hf_hub_base_url: str = "https://huggingface.co"
    """Base URL of the HuggingFace Hub API, used to list a job's output-model
    files for the result-report endpoint. Public default; overridable for a
    mirror or a test double."""

    hf_viewer_timeout_seconds: float = Field(default=2.5, gt=0)
    """Per-call HTTP timeout for one viewer request. A preview makes a ``/splits``
    call to discover the config, then fetches the two splits' ``/rows``
    concurrently, so the worst-case latency preview adds to a
    ``GET /datasets/{id}?preview=true`` read is roughly two of these timeouts."""

    # --- LLM intelligence (Phase 2; optional) ----------------------------
    llm_base_url: str | None = None
    """OpenAI-compatible gateway base URL. The adapter appends ``/chat/completions``."""

    llm_api_key: SecretStr | None = None
    """Bearer token for the gateway. Unwrapped only at call time, never logged."""

    llm_model: str | None = None
    """Model name passed through to the gateway. No provider-model default,
    deliberately — unlike the 2025 hardcoded model names."""

    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    """Per-call HTTP timeout for a single chat completion."""

    llm_max_retries: int = Field(default=2, ge=0)
    """Bounded parse-strategy self-correction retries (total attempts = this + 1)."""

    llm_max_sample_bytes: int = Field(default=8000, ge=1)
    """Cap on sample bytes sent to the LLM (cost + injection-surface bound)."""

    llm_structured_output: Literal["json_schema", "json_object", "none"] = "json_object"
    """How to ask the gateway for JSON, since gateways diverge on ``response_format``.

    - ``json_object`` (default): send ``response_format={"type": "json_object"}`` —
      JSON mode without a schema. Widely supported, including OpenAI-compatible
      gateways *and* Bedrock/Anthropic behind litellm.
    - ``json_schema``: send the full OpenAI structured-output schema. Stricter, but
      Bedrock's Claude rejects common JSON-Schema keywords (``minimum``/``maximum``,
      ``anyOf``, ...) with a 400, so only use it against a gateway known to accept them.
    - ``none``: send no ``response_format`` at all. Guaranteed to be accepted by any
      gateway; relies purely on the prompt.

    Any mode is safe because ``DatasetIntelligenceService`` already extracts the
    JSON, validates it with Pydantic, and retries — the ``response_format`` is only
    a hint, never the contract."""

    # --- Reward-function sandbox (online-RL reward step) ---
    reward_timeout_seconds: int = Field(default=5, ge=1)
    """Hard wall-clock (and CPU rlimit) budget for one sandboxed reward run.

    The subprocess executor SIGKILLs the child's process group past this — the
    concrete fix for the 2025 daemon-thread timeout that could not kill runaway
    code."""

    reward_memory_bytes: int = Field(default=512 * 1024 * 1024, ge=1)
    """Address-space rlimit for the reward sandbox child. Enforced on Linux
    (production/Docker); on macOS dev ``RLIMIT_AS`` may be a no-op, but the
    wall-clock kill always applies."""

    # --- Chat assistant + MCP server ---
    enable_mcp: bool = False
    """Mount the standalone MCP server at ``/mcp``. Requires the ``[mcp]`` extra
    (``fastmcp``) and, for external callers, ``"api_key"`` in ``auth_providers``.
    The in-app chat does NOT depend on this."""

    chat_max_tool_iterations: int = Field(default=8, ge=1)
    """Upper bound on ReAct tool-call rounds per chat turn (a runaway guard)."""

    chat_thread_ttl_seconds: float = Field(default=3600.0, gt=0)
    """Idle lifetime of a ``thread_id`` conversation held in the in-process memory."""

    chat_max_threads: int = Field(default=500, ge=1)
    """LRU cap on stored conversations, bounding chat memory growth."""

    # --- Execution (not implemented yet; see CLAUDE.md open decisions) ---
    queue_url: str | None = None
    """Broker URL for the future JobRunner implementation, e.g. redis://...."""

    # --- Execution: job launch ---
    # See docs/superpowers/specs/2026-08-06-job-launch-runner-design.md.
    job_backend: Literal["none", "llmb", "local"] = "none"
    """Which JobRunner to build. ``none`` → NoOpJobRunner (jobs stay pending, the
    default). ``llmb`` → InProcessJobRunner submitting granite.build builds via the
    llmb CLI (custom_code, or the local-bash spec when ``gb_environment="standalone"``).
    ``local`` → LocalJobRunner running the ``autotune`` HPO in-process (no granite.build)."""

    job_runtime_image: str | None = None
    """Container image the cluster runs the tuning build in. Required when ``llmb``."""

    job_trainer_repo: str | None = None
    """Trainer source repository the build checks out. Required when ``llmb``."""

    job_trainer_ref: str = "main"
    """Branch/tag/commit of ``job_trainer_repo`` to check out."""

    job_output_uri_root: str | None = None
    """Root URI for run artifacts; the run's output is a subpath. Required when ``llmb``."""

    job_callback_url: str | None = None
    """The api-bridge / callback base URL the build reports to.

    The ``custom_code`` and LSF start commands emit it as
    ``--autotunex_server_url`` only when it is set; the standalone local-bash
    variant ALWAYS emits it as ``AUTOTUNEX_SERVER_URL``, defaulting to
    ``http://localhost:8001`` (the api-bridge) when this is unset."""

    gb_token_env: str = "GB_TOKEN"
    """Name of the env var holding the llmb/gb auth token.

    Mirrors ``hf_token_env``: before ``llmb build start`` the launcher runs
    ``llmb auth login --token <value>`` reading the token from this variable
    (``GB_TOKEN`` by default, matching the 2025 ``gb_service.login_gb``). Only the
    var's name lives in ``Settings`` — the token value is read at the subprocess
    boundary, never loaded into settings and never logged. When
    ``job_backend="llmb"`` this env var is REQUIRED at startup: there is no
    ambient-credential fallback, so startup fails if it is unset."""

    gb_tags: str = "autotunex"
    """Tag(s) passed to ``llmb build start`` and ``llmb artifact push`` via ``--tag``.

    A single tag or a comma-separated list (``"autotunex,exp-42"``) — the value is
    forwarded verbatim as one ``--tag`` argument, and ``llmb`` itself splits a
    comma-separated list, so no parsing happens here. Defaults to ``"autotunex"``,
    which both tags every build with a shared marker and preserves the tag the
    dataset push already carried. An empty or whitespace-only value disables
    tagging entirely: the ``--tag`` flag is then omitted from both commands."""

    job_spec_dir: Path = Path("tmp")
    """Directory the generated ``build.yaml`` is written to, one per job at
    ``<job_spec_dir>/<job_id>/build.yaml``.

    Relative paths resolve against the process working directory (the repo root
    under ``uvicorn autotunex.main:app``). Unlike a scratch temp file, the spec is
    *kept* after submission — including when the launch fails — so it can be
    inspected or replayed by hand; the default ``tmp/`` is git-ignored. Never
    required (it has a working default), so ``_validate_job_backend`` does not
    demand it when ``job_backend="llmb"``."""

    gb_server_url: str | None = None
    """Base URL of the gbserver REST API the reconcile loop polls, e.g.
    ``https://gbserver.example.com``. Required when ``job_backend="llmb"``: a
    launcher without reconcile parks every job at ``pending`` forever, which is
    the defect this exists to fix, so startup fails rather than shipping it."""

    job_reconcile_interval_seconds: int = Field(default=30, ge=1)
    """How often the reconcile loop sweeps non-terminal jobs for status changes.
    30 s is imperceptible against tuning runs of minutes to hours."""

    job_reconcile_concurrency: int = Field(default=5, ge=1)
    """Upper bound on simultaneous status reads against gbserver per sweep, so N
    in-flight jobs never open N simultaneous connections."""

    # --- Execution: local & bash runners ---
    # See docs/superpowers/specs/2026-08-11-local-and-bash-runners-design.md.
    gb_environment: str | None = Field(
        default=None,
        validation_alias=AliasChoices("gb_environment", "GB_ENVIRONMENT"),
    )
    """granite.build's own (unprefixed) environment name.

    When ``"standalone"`` and ``job_backend="llmb"``, the launcher emits the
    local-bash spec (``space://steps/autotune``) instead of custom_code. Read via
    a ``validation_alias`` so the ``AUTOTUNEX_`` prefix does **not** apply — the
    exact ``GB_ENVIRONMENT`` variable granite.build already sets is read, and the
    prefixed ``AUTOTUNEX_GB_ENVIRONMENT`` is deliberately ignored. The alias is an
    :class:`~pydantic.AliasChoices` including the bare field name rather than a plain
    ``"GB_ENVIRONMENT"`` string so the field can still be populated by name in a
    ``Settings(gb_environment=...)`` call; a plain alias silently drops that init
    kwarg (init kwargs match aliases, not field names, when an alias is set)."""

    bash_fm_tune_root: str | None = None
    """``FM_TUNE_ROOT`` injected into the bash spec's ``bash.env`` (the fm-tune checkout/repo)."""

    bash_fm_tune_ref: str | None = None
    """``FM_TUNE_REF`` injected into the bash spec's ``bash.env`` (the fm-tune branch/tag/commit
    to check out; ``None`` leaves the repo's default branch)."""

    bash_fm_tune_extra: str = "full,mlx"
    """``FM_TUNE_EXTRA`` for the bash spec: the extras to install (default ``full,mlx``)."""

    bash_backend: str = "mlx"
    """``BACKEND`` for the bash spec: ``mlx`` (Apple Silicon) or ``torch``."""

    # --- Execution: LSF / SkyPilot standalone runner ---
    # See docs/superpowers/specs/2026-08-13-lsf-runner-design.md.
    lsf_cluster: str | None = None
    """SkyPilot/LSF cluster name (``launcher_config.resources.cluster``).

    **Discriminator.** When set together with ``job_backend="llmb"`` and
    ``gb_environment="standalone"``, the launcher emits the LSF/SkyPilot build
    spec instead of the local-bash spec. Unset keeps the bash default."""

    lsf_environment_uri: str | None = None
    """granite.build space environment for the LSF build (``environment_uri``),
    e.g. ``space://environments/skypilot/lsf/<cluster>``. Required in LSF mode.
    No default so no cluster-specific identifier is baked into ``src/``."""

    lsf_image: str | None = None
    """Runtime container image the LSF step runs in (``skypilot.image``). Required in LSF mode."""

    lsf_accelerators: str | None = None
    """SkyPilot accelerators string (``resources.accelerators``), e.g. ``"H100:2"``.
    Unset omits the key entirely, producing a 0-GPU build."""

    lsf_queue: str | None = None
    """LSF queue mapped to ``resources.zone``. Optional; omitted when unset."""

    lsf_memory: str | None = None
    """Requested memory for ``resources.memory``. Optional; omitted when unset."""

    lsf_venv_path: str = "/step_venv"
    """``skypilot.venv_path``. The runtime image installs torch/ray here and only
    symlinks it off PATH under enroot, so it must be set explicitly."""

    lsf_cuda_home: str = "/opt/share/cuda-12.9"
    """CUDA toolkit path exported in the LSF start command (the toolkit present on
    the runtime image; the custom_code build uses ``/usr/local/cuda-13.0``)."""

    lsf_num_cpus_per_node: int = Field(default=32, ge=1)
    """``compute_config.num_cpus_per_node`` for the LSF build."""

    lsf_total_memory_per_node: str = "256Gi"
    """``compute_config.total_memory_per_node`` for the LSF build."""

    lsf_poll_interval_seconds: int = Field(default=30, ge=1)
    """Both ``poll_interval_seconds`` and ``log_retrieval_interval_seconds`` on the LSF step."""

    local_ray_address: str | None = None
    """Ray cluster address for the ``local`` runner. ``None`` → ``ray.init()`` (local)."""

    local_output_dir: Path = _DEFAULT_LOCAL_OUTPUT_DIR
    """Root for a local run's output; the per-job subdir is ``<local_output_dir>/<job_id>/``.

    Defaults to ``artifact_dir / "local"``. The default is anchored under
    ``artifact_dir`` by ``_default_local_output_dir`` rather than being a fixed
    literal, so overriding ``artifact_dir`` moves this with it unless it is set
    explicitly."""

    local_cancel_timeout_seconds: float = 30.0
    """Seconds ``LocalJobRunner.cancel`` waits for an in-process run to stop before
    returning ``JobCancellationInProgressError``. The cancel is latched regardless;
    this only bounds the wait. Read as ``AUTOTUNEX_LOCAL_CANCEL_TIMEOUT_SECONDS``."""

    # --- Auth --------------------------------------------------------------
    auth_providers: list[AuthProviderName] = Field(
        default_factory=_default_auth_providers, min_length=1
    )
    """Which credential kinds are accepted. ``["disabled"]`` is standalone mode.

    Non-empty by construction: an empty list passes every rule below yet leaves
    a service that 401s every request with nothing in the configuration to
    explain why.
    """

    standalone_email: str | None = None
    """The email attributed to the standalone principal's writes.

    When unset, standalone mode attributes writes to the default
    ``SYSTEM_STANDALONE_EMAIL`` system owner; when set, that email is the
    owner instead. Either way the owner's ``users`` row is provisioned
    lazily, on first request, not at startup.
    """

    standalone_role: str = ADMIN_ROLE
    """The role the standalone principal carries, whether or not ``standalone_email`` is set."""

    api_keys: dict[str, str] = Field(default_factory=dict)
    """SHA-256 hex digest of a key -> the owner's email. Never raw keys."""

    auto_provision_users: bool = False
    """Just-in-time provision a ``users`` row on a caller's first request.

    Off by default: provisioning is an authorization *policy*. When on, a caller
    with a resolvable, already-verified email (a real provider — OIDC/session
    token or the ``api_keys`` mapping) but no matching ``users`` row gets one
    created on the spot, so they own what they create instead of being refused
    a 403. This flag governs only those real providers: standalone mode always
    resolves to a concrete owner email (the default ``SYSTEM_STANDALONE_EMAIL``
    or a configured ``standalone_email``) and is always provisioned, regardless
    of this flag.

    Two properties are load-bearing and enforced in ``api/deps.get_principal`` /
    ``SqlAlchemyUserRepository.provision``: a provisioned row is always
    ``role='user'`` (never admin — see the design spec's no-privilege-escalation
    rule), and creation is race-safe (a concurrent first request loses the
    ``UNIQUE(email)`` insert and re-reads the winner rather than erroring). Note
    this makes even a ``GET`` write to the database on a caller's first request —
    the deliberate cost of JIT, which is why it is opt-in. See CLAUDE.md open
    decision 4.
    """

    login_activity_throttle_minutes: int = Field(default=15, ge=0)
    """How stale ``users.last_login_at`` must be before a request refreshes it.

    Backs the Users table's "Last login on" column. A completed browser login
    (``/auth/callback``) always records itself; this governs the *other* half —
    the refresh on any authenticated request in ``api/deps.get_principal``, which
    is what gives API-key and OIDC-bearer callers a value at all, since neither
    ever passes through the login flow.

    A window rather than a write per request: unthrottled, every ``GET`` from
    every caller would carry an ``UPDATE`` and a commit, and the value is only
    ever read at minute granularity by a human looking at a table. 15 minutes
    bounds the cost to at most one write per caller per quarter hour while
    keeping the column accurate enough to answer "was this person here today".
    Raise it to trade precision for fewer writes.

    ``0`` disables throttling and records on every request — meant for tests and
    for diagnosing whether the touch fires at all, not for production.
    """

    allow_insecure_no_auth: bool = False
    """Permit ``auth_providers=["disabled"]`` while ``environment="prod"``.

    Off by default, so a production deployment that forgets to configure a real
    provider fails fast at startup rather than silently serving every caller as
    the standalone system owner (see the design spec). Set it only for a
    deliberate single-tenant deployment that genuinely wants no authentication;
    ``create_app`` logs a loud warning when it takes effect.
    """

    oidc_issuer: str | None = None
    """The issuer (``iss``) a bearer token must carry. Required if "oidc" is enabled."""

    oidc_jwks_uri: str | None = None
    """Where to fetch the issuer's signing keys. Required if "oidc" is enabled."""

    oidc_audience: str | None = None
    """The audience (``aud``) a bearer token must carry. Required if "oidc" is enabled.

    Checked unconditionally, not gated on whether a client id happens to be
    configured — see the comment above the audience check in
    ``_validate_auth`` for why that distinction is a real vulnerability, not
    a style preference.
    """

    oidc_algorithms: list[str] = Field(default_factory=lambda: ["RS256"])
    """Signature algorithms accepted when verifying a bearer token."""

    oidc_email_claims: list[str] = Field(default_factory=lambda: ["email", "emailAddress"])
    """Claim names checked, in order, to resolve a token to an email."""

    oidc_leeway_seconds: int = Field(default=30, ge=0, le=300)
    """Clock-skew tolerance for exp/nbf checks."""

    # --- Backend-for-frontend (browser sessions) --------------------------
    oidc_client_id: str | None = None
    """The BFF's own OIDC client id, distinct from bearer-token verification above.

    Required if "session" is enabled.
    """

    oidc_client_secret: SecretStr | None = None
    """The BFF's OIDC client secret, exchanged at the token endpoint.

    Required if "session" is enabled.
    """

    oidc_authorization_endpoint: str | None = None
    """Where ``/auth/login`` redirects the browser. Required if "session" is enabled."""

    oidc_token_endpoint: str | None = None
    """Where ``/auth/callback`` exchanges the authorization code.

    Required if "session" is enabled.
    """

    oidc_end_session_endpoint: str | None = None
    """Where ``/auth/logout`` redirects to end the upstream session, if the issuer
    supports RP-initiated logout. Optional: no rule below requires it.
    """

    public_base_url: str | None = None
    """Builds redirect_uri. Never taken from a request header."""

    session_secret: SecretStr | None = Field(default=None, min_length=32)
    """Signs the session JWT. No random fallback — required whenever "session" is enabled.

    ``min_length=32`` is RFC 7518 §3.2's own floor for an HS256 key, not a
    stylistic choice: this is the entire strength of the session cookie, and
    a short secret is exactly what makes forging one tractable. It is also
    what was silently missing before — a short secret still passed the
    non-empty/non-whitespace check below, and every mint and every verify
    then logged PyJWT's own ``InsecureKeyLengthWarning``, which in production
    means every request. ``Field(min_length=...)`` applies to ``SecretStr``
    the same as to ``str`` (verified against the installed pydantic 2.13),
    and — like every other field here — is skipped when the value is
    ``None``, so "unset" still falls through to the rule below rather than
    failing with a length error that would misname the actual problem.
    Deliberately not applied to ``oidc_client_secret``: that value is issued
    by the IdP, not chosen by this service, so we are in no position to
    constrain its shape.
    """

    session_ttl_hours: int = Field(default=8, ge=1, le=24)
    """How long a minted session cookie remains valid."""

    session_cookie_same_site: Literal["lax", "none"] = "lax"
    """"none" is only legal alongside a non-empty ``cors_allow_origins`` (see the
    validator below): cross-site cookies need an explicit allowlist.
    """

    cors_allow_origins: list[str] = Field(default_factory=list)
    """Origins ``CORSMiddleware`` allows once a browser UI is wired in.

    Must never contain ``"*"``: paired with the credentialed CORS the
    backend-for-frontend requires, a wildcard origin would let
    ``CORSMiddleware`` echo back any request's ``Origin`` header with the
    session cookie attached — see the validator below for the mechanism.
    """

    @property
    def dataset_staging_dir(self) -> Path:
        """Where in-flight uploads are streamed before the runner processes them."""
        return self.dataset_storage_dir / ".staging"

    @property
    def llm_configured(self) -> bool:
        """Whether all three required LLM settings are set.

        ``_validate_llm`` guarantees they are all-set-or-all-unset, so this is a
        clean feature toggle for ``api/deps.get_dataset_intelligence_service``.
        """
        return (
            not _is_unset(self.llm_base_url)
            and not _is_unset(_secret_value(self.llm_api_key))
            and not _is_unset(self.llm_model)
        )

    @field_validator("gb_environment", mode="after")
    @classmethod
    def _normalize_gb_environment(cls, value: str | None) -> str | None:
        """Fold ``gb_environment`` to lowercase (trimmed) for case-insensitive matching.

        granite.build's own uppercase ``GB_ENVIRONMENT=STANDALONE`` must match the lowercase
        ``"standalone"`` the launcher registry and ``_validate_job_backend`` compare against.
        Runs during field validation — before every ``model_validator`` — so
        ``_validate_job_backend`` sees the canonical value. Only AutoTuneX's own branch
        choice reads this setting; the ``llmb`` subprocess reads the real ``GB_ENVIRONMENT``
        from the process environment, not this normalized copy, so granite.build still sees
        whatever the operator set. An empty/whitespace-only value folds to ``None`` (unset).
        """
        if value is None:
            return None
        return value.strip().lower() or None

    @model_validator(mode="after")
    def _validate_auth(self) -> Settings:
        """Fail startup rather than at request time on misconfiguration."""
        if self.auth_providers.count("disabled") > 0 and len(self.auth_providers) > 1:
            raise ValueError('"disabled" cannot combine with another auth provider.')
        if (
            self.environment == "prod"
            and self.auth_providers == ["disabled"]
            and not self.allow_insecure_no_auth
        ):
            raise ValueError(
                "Refusing to start in production with authentication disabled. "
                "Set allow_insecure_no_auth=True to override this deliberately."
            )
        if "api_key" in self.auth_providers:
            if not self.api_keys:
                raise ValueError('"api_key" is enabled but settings.api_keys is empty.')
            for key, email in self.api_keys.items():
                if not re.fullmatch(r"[0-9a-f]{64}", key):
                    raise ValueError(
                        f"api_keys entry for {email!r} is not a SHA-256 hex digest "
                        "(64 lowercase hex chars); the offending key is withheld "
                        "from this message — store the digest, not the key itself."
                    )
        # oidc_audience is required unconditionally, never gated on whether a
        # client id happens to be configured: granite.build made audience
        # checking conditional on that, so a caller with no client id set
        # silently accepted tokens minted for a different application on the
        # same issuer. Keep this check unconditional.
        if "oidc" in self.auth_providers:
            # Falsy, not `is None`: `AUTOTUNEX_OIDC_AUDIENCE=` with nothing
            # after the `=` — the single most likely way to write this in a
            # `.env` — parses to `""`, not `None`. An `is None` test let that
            # through the very validator built to catch it, and the service
            # then rejected every token, since no real `aud` equals `""`.
            # Whitespace-only (`" "`, `"\t"`) is the same gap one layer down:
            # truthy, so `not value` alone misses it, and no real `aud`,
            # `iss`, or JWKS URI is ever whitespace either. `_is_unset` covers
            # both shapes. Empty or whitespace-only means unset here.
            missing = [
                name
                for name, value in (
                    ("oidc_issuer", self.oidc_issuer),
                    ("oidc_jwks_uri", self.oidc_jwks_uri),
                    ("oidc_audience", self.oidc_audience),
                )
                if _is_unset(value)
            ]
            if missing:
                raise ValueError(
                    f'"oidc" is enabled but {", ".join(missing)} is not set '
                    "(an empty or whitespace-only value counts as unset)."
                )
        if "session" in self.auth_providers:
            # `_is_unset`, not `is None` and not a bare `not value`: the same
            # empty-string / whitespace-only gap the "oidc" block above closes
            # applies here too, and matters most for `session_secret` — it is
            # the HS256 key the session cookie is signed with, so an unset
            # value slipping through would let anyone mint a session cookie
            # for any email address. The two `SecretStr` fields are unwrapped
            # first: a `SecretStr` object is truthy no matter what it wraps.
            missing = [
                name
                for name, value in (
                    ("oidc_issuer", self.oidc_issuer),
                    ("oidc_jwks_uri", self.oidc_jwks_uri),
                    ("oidc_audience", self.oidc_audience),
                    ("oidc_client_id", self.oidc_client_id),
                    ("oidc_client_secret", _secret_value(self.oidc_client_secret)),
                    ("oidc_authorization_endpoint", self.oidc_authorization_endpoint),
                    ("oidc_token_endpoint", self.oidc_token_endpoint),
                    ("public_base_url", self.public_base_url),
                    ("session_secret", _secret_value(self.session_secret)),
                )
                if _is_unset(value)
            ]
            if missing:
                raise ValueError(
                    f'"session" is enabled but {", ".join(missing)} is not set '
                    "(an empty or whitespace-only value counts as unset)."
                )
        if "*" in self.cors_allow_origins:
            # Not gated on "session" being enabled: this is a property of
            # `cors_allow_origins` itself, since a later task attaches
            # `CORSMiddleware(..., allow_credentials=True)` using this exact
            # list — see `test_cors_allow_origins_rejects_a_wildcard` and this
            # task's report for the Starlette source lines that make a
            # wildcard origin plus credentialed CORS dangerous, not merely
            # unusual.
            raise ValueError(
                'cors_allow_origins must not contain "*": combined with the '
                "credentialed CORS (allow_credentials=True) the backend-for-frontend "
                "requires, Starlette's CORSMiddleware echoes back any request's "
                "Origin header with the session cookie attached, which defeats "
                "the allowlist entirely."
            )
        if self.session_cookie_same_site == "none" and not self.cors_allow_origins:
            raise ValueError(
                'session_cookie_same_site="none" requires a non-empty cors_allow_origins.'
            )
        if self.standalone_role not in _KNOWN_ROLES:
            raise ValueError(
                f"standalone_role must be one of {sorted(_KNOWN_ROLES)}, "
                f"got {self.standalone_role!r}."
            )
        return self

    @model_validator(mode="after")
    def _validate_debug(self) -> Settings:
        """Debug mode leaks tracebacks; never allow it in production."""
        if self.environment == "prod" and self.debug:
            raise ValueError("debug=True is not permitted when environment='prod'.")
        return self

    @model_validator(mode="after")
    def _validate_datasets(self) -> Settings:
        """Fail startup on an unusable dataset-storage configuration.

        Two fail-fast checks, mirroring ``_validate_auth``'s stance (a
        misconfiguration should fail at startup, not on the first upload):

        - A forced ``huggingface`` backend is impossible for the same-host
          *bash* standalone runner (``gb_environment="standalone"`` with
          ``lsf_cluster`` unset): the HF push runs ``llmb artifact push``,
          which the CLI disables under ``GB_ENVIRONMENT=STANDALONE``. Checked
          first, so it fires regardless of whether the tokens happen to be
          set. The remote LSF/SkyPilot standalone variant (``lsf_cluster``
          set) is exempt: it runs on a cluster a local ``file://`` locator
          cannot reach, so ``huggingface`` is the intended storage there —
          its own ``llmb artifact push`` limitation is a separate, deferred
          non-goal, not something this check tries to solve.
        - A forced ``huggingface`` backend with either token absent would
          otherwise fail on the first upload, far from the misconfiguration.

        ``"auto"`` never fails here — it degrades to ``local`` (and, in
        bash-standalone, emits a ``file://`` locator for the bash build; see
        ``services/storage/registry.get_storage_backend``).
        """
        if self.dataset_storage_backend == "huggingface":
            if self.gb_environment == "standalone" and not self.lsf_cluster:
                raise ValueError(
                    'dataset_storage_backend="huggingface" is not supported by '
                    "the same-host bash standalone runner (gb_environment="
                    '"standalone" without lsf_cluster): the HuggingFace push '
                    "runs `llmb artifact push`, which granite.build disables in "
                    'standalone mode. Use "auto" (stores locally with a file:// '
                    'locator in standalone) or "local". The remote LSF/SkyPilot '
                    "standalone variant (lsf_cluster set) still uses huggingface."
                )
            missing = [
                env for env in (self.gb_token_env, self.hf_token_env) if not os.environ.get(env)
            ]
            if missing:
                raise ValueError(
                    f'dataset_storage_backend="huggingface" requires the '
                    f"{', '.join(repr(env) for env in missing)} environment "
                    "variable(s) to be set (the GB token authenticates the llmb CLI; "
                    "the HF token is the HuggingFace push destination)."
                )
        return self

    @model_validator(mode="after")
    def _validate_llm(self) -> Settings:
        """Fail startup on a *partial* LLM configuration.

        Mirrors ``_validate_auth`` / ``_validate_datasets``: all of
        ``{llm_base_url, llm_api_key, llm_model}`` must be set together or none.
        All unset means the feature is disabled and the endpoints return 503 at
        request time. ``_is_unset`` treats empty/whitespace as unset; the
        ``SecretStr`` is unwrapped for the check via ``_secret_value``.
        """
        provided = {
            "llm_base_url": not _is_unset(self.llm_base_url),
            "llm_api_key": not _is_unset(_secret_value(self.llm_api_key)),
            "llm_model": not _is_unset(self.llm_model),
        }
        if any(provided.values()) and not all(provided.values()):
            missing = sorted(name for name, ok in provided.items() if not ok)
            raise ValueError(
                "LLM intelligence is partially configured; set all of "
                "{llm_base_url, llm_api_key, llm_model} together or none "
                f"(missing: {', '.join(missing)})."
            )
        return self

    @model_validator(mode="after")
    def _default_local_output_dir(self) -> Settings:
        """Anchor ``local_output_dir`` under ``artifact_dir`` unless set explicitly.

        The field default is the sentinel ``_DEFAULT_LOCAL_OUTPUT_DIR``; when it is
        still that value we recompute it as ``artifact_dir / "local"`` so the two
        paths stay related even when ``artifact_dir`` is overridden (the reason this
        is a validator, not a plain literal default). A caller who pins
        ``local_output_dir`` to anything else keeps it. The one ambiguous case — a
        caller pinning it to exactly the sentinel while also overriding
        ``artifact_dir`` — is treated as "left at default" and re-anchored, which is
        the safe reading for a value whose whole purpose is to sit under
        ``artifact_dir``.
        """
        if self.local_output_dir == _DEFAULT_LOCAL_OUTPUT_DIR:
            self.local_output_dir = self.artifact_dir / "local"
        return self

    @model_validator(mode="after")
    def _validate_job_backend(self) -> Settings:
        """Fail startup if ``job_backend="llmb"`` is missing a required build input.

        Mirrors ``_validate_llm``'s fail-fast stance: a launcher with no image,
        trainer repo, output root, ``gb_server_url``, or the ``gb_token_env``
        environment variable would otherwise fail on the first submitted job (or,
        for the token, on the reconcile loop's first sweep), far from the
        misconfiguration.

        Three shapes, keyed on ``job_backend`` (and, for ``llmb``, on
        ``gb_environment``):

        - ``none`` and ``local`` require nothing. ``local`` runs the ``autotune``
          HPO in-process; whether that package is installed is a *runtime* concern
          (a clear error when it is selected and absent), not a startup gate, so a
          ``local``-configured deployment without ``autotune`` still starts.
        - ``llmb`` with ``gb_environment="standalone"`` (the local-bash variant)
          drops the custom_code-only inputs (``job_runtime_image``,
          ``job_trainer_repo``, ``job_output_uri_root``); none are *required*. The
          bash spec anchors its run output under ``artifact_dir`` (as an absolute
          ``file://`` URI), not ``job_output_uri_root`` — the bash environment only
          writes ``file://``. It still requires ``gb_server_url`` (the reconcile loop
          polls the local gbserver) and still honours the ``gb_token_env`` presence check.
        - ``llmb`` with ``gb_environment="standalone"`` **and** ``lsf_cluster`` set
          (the LSF/SkyPilot variant) additionally requires ``lsf_environment_uri``,
          ``lsf_image`` and ``job_trainer_repo``.
        - ``llmb`` otherwise (custom_code) keeps the full required set, unchanged.
        """
        if self.job_backend == "llmb":
            required: tuple[tuple[str, str | None], ...]
            if self.gb_environment == "standalone":
                if self.lsf_cluster:
                    # LSF (SkyPilot) variant: the bash inputs plus the LSF build inputs.
                    required = (
                        ("gb_server_url", self.gb_server_url),
                        ("lsf_environment_uri", self.lsf_environment_uri),
                        ("lsf_image", self.lsf_image),
                        ("job_trainer_repo", self.job_trainer_repo),
                    )
                else:
                    # local-bash variant: no cluster inputs.
                    required = (("gb_server_url", self.gb_server_url),)
            else:
                required = (
                    ("job_runtime_image", self.job_runtime_image),
                    ("job_trainer_repo", self.job_trainer_repo),
                    ("job_output_uri_root", self.job_output_uri_root),
                    ("gb_server_url", self.gb_server_url),
                )
            missing = [name for name, value in required if _is_unset(value)]
            if missing:
                raise ValueError(
                    f'job_backend="llmb" requires {", ".join(missing)} to be set '
                    "(an empty or whitespace-only value counts as unset)."
                )
            # Over HTTP there is no ambient CLI-credential fallback, so the token
            # the reconcile loop authenticates to gbserver with is genuinely
            # required. Its *value* is read at the request boundary and never
            # loaded into Settings — this only checks the env var is present, the
            # same shape as ``_validate_datasets``.
            if not os.environ.get(self.gb_token_env):
                raise ValueError(
                    f'job_backend="llmb" requires the {self.gb_token_env!r} environment '
                    "variable to be set (the reconcile loop authenticates to gbserver "
                    "over HTTP with it; there is no ambient credential fallback)."
                )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
