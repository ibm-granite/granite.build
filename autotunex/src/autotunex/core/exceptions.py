# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Domain exceptions.

Services raise these. They are translated into RFC 9457 problem-detail HTTP
responses by the handlers registered in :mod:`autotunex.api.errors`.

Never raise ``fastapi.HTTPException`` outside the ``api`` package.
"""

from __future__ import annotations

from http import HTTPStatus


class AutoTuneXError(Exception):
    """Base class for every AutoTuneX domain error."""

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    title: str = "Internal Server Error"
    headers: dict[str, str] | None = None

    def __init__(self, detail: str, *, headers: dict[str, str] | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.headers = headers


class NotFoundError(AutoTuneXError):
    """A requested resource does not exist."""

    status_code = HTTPStatus.NOT_FOUND
    title = "Not Found"


class JobNotFoundError(NotFoundError):
    """No job exists with the requested identifier."""

    def __init__(self, job_id: object) -> None:
        super().__init__(f"Job {job_id} not found.")


class BuildNotFoundError(NotFoundError):
    """No job is associated with the requested granite.build build id.

    Raised by the by-build-id lookup for both "no ``gb_task`` carries this build
    id" and "the resolved job belongs to another owner" — a scoped caller must
    not tell the two apart, exactly as :class:`JobNotFoundError` hides another
    user's job on the read path. ``JobNotFoundError`` is not reused because at the
    point of failure there is no job id to name; the message speaks of the build
    the caller actually supplied.
    """

    def __init__(self, build_id: object) -> None:
        super().__init__(f"No job found for build {build_id}.")


class TrialNotFoundError(NotFoundError):
    """No trial exists with the requested identifier."""

    def __init__(self, trial_id: object) -> None:
        super().__init__(f"Trial {trial_id} not found.")


class ConfigurationNotFoundError(NotFoundError):
    """No configuration exists with the requested identifier.

    Also raised when the configuration exists but belongs to another user — a
    scoped caller must not be able to tell "yours, gone" apart from "someone
    else's", exactly as :class:`JobNotFoundError` does on the read path.
    """

    def __init__(self, configuration_id: object) -> None:
        super().__init__(f"Configuration {configuration_id} not found.")


class UserNotFoundError(NotFoundError):
    """No user exists with the requested identifier."""

    def __init__(self, user_id: object) -> None:
        super().__init__(f"User {user_id} not found.")


class ForbiddenError(AutoTuneXError):
    """An authenticated caller is not permitted to perform this action.

    Distinct from the 401 authentication errors: the credential verified, but the
    resolved principal may not do this. The read path never emits a 403 — an
    unprovisioned caller sees an empty page instead, so existence never leaks. A
    *write* has no empty-page analogue, so this is where a caller with no
    resolvable identity is turned away (see :class:`CallerNotProvisionedError`).
    """

    status_code = HTTPStatus.FORBIDDEN
    title = "Forbidden"


class CallerNotProvisionedError(ForbiddenError):
    """The caller has no ``users`` row, so a created resource has no owner.

    ``configurations.user_id`` is ``NOT NULL`` and references ``users`` — a row
    needs an owner. Standalone mode is never the cause: ``get_principal`` always
    provisions its resolved owner (the default ``SYSTEM_STANDALONE_EMAIL`` or a
    configured ``standalone_email``), independent of ``auto_provision_users``, so
    a standalone caller always has an identity to attach. This fires only for a
    real provider (OIDC, session, or API key) whose verified email has no
    matching ``users`` row while ``auto_provision_users`` is off — an
    authenticated-but-unprovisioned caller (``user_id`` is ``None``, not an
    admin) is refused here rather than allowed to write a row that would
    violate the constraint. The fix is to turn on ``auto_provision_users`` or
    provision the caller's ``users`` row out of band.
    """

    def __init__(self) -> None:
        super().__init__("Your account is not provisioned to create resources.")


class ScopeNotPermittedError(ForbiddenError):
    """A non-admin requested the cross-user view (``?scope=all``).

    Unlike the rest of the read path — which returns an empty page / 404 so a
    resource's existence never leaks — this 403 is safe to emit on a read: it
    is a verdict on the *privilege to widen scope*, decided before any row is
    consulted, so it reveals nothing about what data exists. Only an admin may
    ask for ``DataScope.ALL``; everyone else is turned away here.
    """

    def __init__(self) -> None:
        super().__init__("Only an administrator may request the cross-user (scope=all) view.")


class AdminRequiredError(ForbiddenError):
    """The action requires an administrator, and the caller is not one.

    A coarse authorization gate applied to the user-management endpoints via the
    ``require_admin`` dependency (``api/deps.py``). Distinct from
    :class:`ScopeNotPermittedError`, which refuses only the cross-user
    *widening* of an owned-resource read; here the whole endpoint is admin-only,
    because a user is an identity, not an owned row with an "own" view to fall
    back to.
    """

    def __init__(self) -> None:
        super().__init__("This action requires administrator privileges.")


class SystemResourceProtectedError(ForbiddenError):
    """A row owned by the reserved system user may not be deleted.

    The shared tier (:data:`~autotunex.core.constants.SYSTEM_USER_ID`) is curated
    starter content that *every* caller reads and launches from, so a delete is
    not a per-tenant loss — it removes the row for everybody, irreversibly. Read
    access being deliberately wide is exactly what makes the delete narrow.

    This is a standing invariant on the row's owner, not a consequence of the
    ownership filter: ``include_shared=False`` on the mutation paths only protects
    a *provisioned non-admin*, and left three principals through — an admin
    widening with ``?scope=all`` (no owner filter at all), and any caller whose own
    resolved ``user_id`` is the system user's, which is the single-owner standalone
    deployment where "every caller" is that one owner.

    The one sanctioned way to curate shared content is to assume the system user
    (``POST /auth/assume/{id}``): that requires a real admin, is already audited by
    ``api/routers/auth.py``, and makes the destructive act explicit rather than a
    side effect of who the ambient principal happens to be. See
    :func:`autotunex.services.scoping.is_delete_protected`.

    A 403 rather than a 404: the row is already readable by this caller, so naming
    it leaks nothing, and the UI needs to tell "protected" apart from "gone" to
    explain itself in the delete dialog.
    """

    title = "System Resource Protected"

    def __init__(self, resource: str, resource_id: object) -> None:
        super().__init__(
            f"{resource} {resource_id} is provided by the system and cannot be deleted."
        )


class ConflictError(AutoTuneXError):
    """A request conflicts with the current state of a resource."""

    status_code = HTTPStatus.CONFLICT
    title = "Conflict"


class ConfigurationNameConflictError(ConflictError):
    """A configuration with this name already exists for the same owner.

    ``configurations`` has a ``UNIQUE (user_id, name)`` constraint. Surfaced as a
    clean 409 rather than a raw ``IntegrityError``, on both create and rename.
    """

    def __init__(self, name: object) -> None:
        super().__init__(f"A configuration named {name!r} already exists.")


class ConfigurationInUseError(ConflictError):
    """A configuration cannot be deleted while a job still references it.

    ``jobs.config_id`` references ``configurations.id`` with ``ON DELETE
    RESTRICT``. Surfaced as a 409 rather than a raw ``IntegrityError``.
    """

    def __init__(self, configuration_id: object) -> None:
        super().__init__(
            f"Configuration {configuration_id} is referenced by one or more jobs "
            "and cannot be deleted."
        )


class PayloadTooLargeError(AutoTuneXError):
    """A request body exceeds a server-enforced size limit."""

    status_code = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    title = "Payload Too Large"


class UnsupportedMediaTypeError(AutoTuneXError):
    """A request carries content in a media type the endpoint does not accept."""

    status_code = HTTPStatus.UNSUPPORTED_MEDIA_TYPE
    title = "Unsupported Media Type"


class BadGatewayError(AutoTuneXError):
    """An upstream dependency returned a failed or invalid response."""

    status_code = HTTPStatus.BAD_GATEWAY
    title = "Bad Gateway"


class ServiceUnavailableError(AutoTuneXError):
    """A required upstream dependency is not currently available."""

    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    title = "Service Unavailable"


class LlmNotConfiguredError(ServiceUnavailableError):
    """An intelligence endpoint was called but no LLM provider is configured.

    The intelligence layer is optional to a deployment, so an unset ``llm_*``
    configuration is a request-time 503 rather than a startup failure.
    """

    def __init__(self) -> None:
        super().__init__("The LLM intelligence feature is not configured on this server.")


class RewardExecutionUnavailableError(ServiceUnavailableError):
    """The reward-function execution sandbox could not be started.

    Reserved for a genuine sandbox-spawn failure. The current subprocess
    executor returns a structured ``executed=False`` result for user-code
    failures instead of raising, so this may be unused at first — it is kept so
    the reward router can declare a 503 and a future stricter executor can raise
    it without a new error shape.
    """

    def __init__(self) -> None:
        super().__init__("The reward-function execution sandbox is unavailable.")


class DatabaseUnavailableError(ServiceUnavailableError):
    """The database is unreachable — raised by the readiness probe.

    A subclass of :class:`ServiceUnavailableError` so it maps to 503, but a
    distinct type so a readiness failure is legible in logs and separable from
    the LLM/reward 503s above. Turns an uncaught ``SQLAlchemyError`` — which the
    global handler would render as a generic 500 — into the 503 an orchestrator
    reads to keep traffic away until the database is back.
    """

    def __init__(self) -> None:
        super().__init__("The database is not reachable.")


class CannotImpersonateSelfError(AutoTuneXError):
    """An admin tried to assume their own identity — a confusing no-op."""

    status_code = HTTPStatus.BAD_REQUEST
    title = "Bad Request"

    def __init__(self) -> None:
        super().__init__("You cannot assume your own identity.")


class ImpersonationUnavailableError(ServiceUnavailableError):
    """Impersonation needs a session secret to sign the overlay cookie, and none is set."""

    def __init__(self) -> None:
        super().__init__("Impersonation is unavailable: no session secret is configured.")


class AutotuneCoreUnavailableError(ServiceUnavailableError):
    """The optional ``autotune`` training core is not importable on this server.

    Its UI config template and dataset-type catalog both come from ``autotune``,
    an optional dependency omitted from a lean install because it is heavy
    (Ray, torch, verl, numpy). Surfacing a request-time 503 (rather than
    failing at startup) keeps the rest of the API usable where the package is
    not installed.
    """

    def __init__(self) -> None:
        super().__init__("The autotune training core is not installed on this server.")


class LlmUnavailableError(BadGatewayError):
    """The upstream LLM call failed or returned unusable output.

    Carries only a fixed, safe message: the raw upstream body or exception text
    is logged at ``warning``, never returned — closing the 2025
    ``HTTPException(500, detail=f"...{e}")`` leak.
    """

    def __init__(
        self,
        detail: str = "The LLM provider could not be reached or returned an invalid response.",
    ) -> None:
        super().__init__(detail)


class DatasetNotFoundError(NotFoundError):
    """No dataset exists with the requested identifier.

    Also raised when the dataset exists but belongs to another user — a scoped
    caller must not distinguish "yours, gone" from "someone else's", exactly as
    :class:`ConfigurationNotFoundError` does.
    """

    def __init__(self, dataset_id: object) -> None:
        super().__init__(f"Dataset {dataset_id} not found.")


class DatasetNameConflictError(ConflictError):
    """A dataset with this name already exists for the same owner.

    ``datasets`` has a ``UNIQUE (user_id, name)`` constraint, surfaced as a clean
    409 rather than a raw ``IntegrityError`` on both create and rename.
    """

    def __init__(self, name: object) -> None:
        super().__init__(f"A dataset named {name!r} already exists.")


class DatasetInUseError(ConflictError):
    """A dataset cannot be deleted while a job still references it.

    ``jobs.dataset_id`` references ``datasets.id`` with ``ON DELETE RESTRICT``.
    """

    def __init__(self, dataset_id: object) -> None:
        super().__init__(
            f"Dataset {dataset_id} is referenced by one or more jobs and cannot be deleted."
        )


class JobNotCancellableError(ConflictError):
    """A job cannot be cancelled because it has already finished.

    ``completed`` and ``error`` are terminal with no work left to stop. An
    already-``terminated`` job is not an error — cancel is idempotent there and
    handled by the service before this is raised.
    """

    def __init__(self, job_id: object, status: object) -> None:
        super().__init__(f"Job {job_id} is already {status} and cannot be cancelled.")


class JobCancellationInProgressError(ConflictError):
    """A local in-process run was asked to stop but had not stopped in time.

    The cancellation is latched (the run will still stop), so the request — a
    cancel or a delete — should be retried shortly. A 409, beside the other
    job-state conflicts.
    """

    def __init__(self, job_id: object) -> None:
        super().__init__(f"Job {job_id} is still stopping; retry shortly.")


class DatasetNotReadyError(ConflictError):
    """An upload was requested while the dataset is already uploading.

    The ``status='uploading'`` row is the coordination point, so this guard is
    durable across replicas rather than an in-process lock.
    """

    def __init__(self, dataset_id: object) -> None:
        super().__init__(f"Dataset {dataset_id} is already uploading; wait for it to finish.")


class JobReferenceConflictError(ConflictError):
    """A job's referenced configuration or dataset changed during submission.

    The service validates both references before inserting, so this signals a
    rare race: one was deleted between the check and the insert. Retry.
    """

    def __init__(self) -> None:
        super().__init__(
            "A referenced configuration or dataset was modified concurrently; please retry."
        )


class AmbiguousIdentityError(AutoTuneXError):
    """Several ``users`` rows match one already-verified email.

    A deployment data bug, not a caller error, so it keeps the base class's 500
    and ``"Internal Server Error"``. ``users.email UNIQUE`` is case-*sensitive*
    on SQLite and Postgres, so ``Alice@example.com`` and ``alice@example.com`` coexist
    there and the case-insensitive principal lookup matches both. Resolving that
    by picking a row would settle admin-ness by row order — the matched rows can
    carry different ``role`` values — so the lookup fails closed instead. The
    root fix is a ``UNIQUE INDEX ON users (lower(email))``, tracked under
    CLAUDE.md open decision 6's schema work.

    The detail is a fixed string, distinct from the generic 500 handler's, so an
    operator reading a client-side trace can tell the two apart. It deliberately
    does not mention the duplication or either email: that goes to the WARNING
    log, which only the operator can read.
    """

    def __init__(self) -> None:
        super().__init__("The account could not be resolved.")


class DomainValidationError(AutoTuneXError):
    """A request is well-formed but violates a domain rule."""

    status_code = HTTPStatus.UNPROCESSABLE_ENTITY
    title = "Unprocessable Entity"


class InvalidConfigDataError(DomainValidationError):
    """A configuration's ``config_data`` is not a non-empty JSON object.

    ``configurations.config_data`` is a schema-less ``JSON`` column, so the API
    does not impose the internal shape the tuning pipeline uses (``tune_config``,
    ``tuners_config``, ``training_config`` and friends) — that shape is neither
    stable nor owned here. The one rule kept is that a configuration must carry
    *some* settings: an empty object, or a non-object body, is rejected.
    """


class InvalidDatasetFormatError(DomainValidationError):
    """``DatasetCreate.data_format`` is not one of ``{jsonl, csv, parquet}``."""


class InvalidSampleError(DomainValidationError):
    """A sample sent to an intelligence endpoint is unusable.

    Raised when the sample is empty, oversized, mis-shaped, or its
    ``data_format`` is outside the accepted set.
    """


class UnknownTrainingFormatError(DomainValidationError):
    """A requested ``target_format`` is not a key in the training-format catalog."""


class EmptyDatasetError(DomainValidationError):
    """An uploaded file contained no bytes."""


class EmptySplitError(DomainValidationError):
    """A resulting train or validation split is empty.

    Surfaced asynchronously via ``status='error'`` (the split runs in the upload
    runner, not the request path), so the client discovers it by polling.
    """


class UploadProcessingError(AutoTuneXError):
    """Base for off-request dataset-processing failures with a safe, authored detail.

    Raised inside the upload runner's background processing (parse/split/remap/
    persist), NOT into the HTTP layer — the request already returned ``202``. The
    runner catches these and records ``status='error'`` with the detail surfaced
    verbatim (like :class:`EmptySplitError`), so the message must be safe: fixed,
    authored here, never raw subprocess or exception text. ``status_code`` is
    inherited but unused (these never reach an HTTP handler).
    """


class DatasetProcessingTimeoutError(UploadProcessingError):
    """Off-request processing exceeded ``dataset_processing_timeout_seconds``."""

    def __init__(self, timeout_seconds: float) -> None:
        super().__init__(
            f"Upload processing exceeded the {timeout_seconds:g}-second limit and was stopped; "
            "re-upload, or split the dataset into smaller files."
        )


class DatasetPushTimeoutError(UploadProcessingError):
    """The ``llmb`` push to storage exceeded ``dataset_push_timeout_seconds``."""

    def __init__(self, timeout_seconds: float) -> None:
        super().__init__(
            f"Publishing the dataset to storage exceeded the {timeout_seconds:g}-second limit "
            "and was stopped; please retry."
        )


class DatasetPushFailedError(UploadProcessingError):
    """The ``llmb`` push to storage failed (bad exit, missing CLI, or unparseable output).

    Carries only a fixed, safe message; the underlying subprocess stderr/exit is
    logged by the caller, never surfaced (it can embed environment detail).
    """

    def __init__(self) -> None:
        super().__init__(
            "Publishing the dataset to storage failed; please retry or contact an operator."
        )


class InsufficientStorageError(UploadProcessingError):
    """Not enough free disk to finalize the upload (staging + the storage copy)."""

    def __init__(self) -> None:
        super().__init__(
            "Not enough disk space to finalize the upload; free up space and re-upload."
        )


class DatasetTooLargeError(PayloadTooLargeError):
    """An uploaded file exceeds ``settings.dataset_upload_max_bytes``."""


class UnsupportedDatasetFormatError(UnsupportedMediaTypeError):
    """An uploaded file's extension is not one of ``{jsonl, csv, parquet}``."""


class MissingRewardFunctionError(DomainValidationError):
    """An online-RL configuration was submitted without a reward function.

    Configurations whose ``rl_tuner_type`` is online-RL (``ppo``/``grpo``/
    ``dapo``) score rollouts with a reward function, so a job referencing one
    must carry ``reward_function_code``.
    """

    def __init__(self, rl_tuner_type: str) -> None:
        super().__init__(
            f"Configuration uses online-RL tuner '{rl_tuner_type}'; a reward function is required."
        )


class DatasetNotReadyForJobError(ConflictError):
    """A job referenced a dataset whose upload has not finished.

    A run may only start once the dataset's file is uploaded and processed
    (``status='ready'``). An ``empty``, ``uploading`` or ``error`` dataset is
    refused so a run never starts against absent or partial data.
    """

    def __init__(self, dataset_id: object, status: object) -> None:
        super().__init__(
            f"Dataset {dataset_id} is not ready to start a job (status: {status}); "
            "its upload must finish first."
        )


class CannotChangeOwnRoleError(ConflictError):
    """An admin may not change their own role.

    Prevents accidental self-demotion and makes the last-admin case harder to
    reach — an admin manages *other* admins, not themselves. A 409 rather than a
    403: the caller *has* the privilege, but the request conflicts with this
    invariant.
    """

    def __init__(self) -> None:
        super().__init__("You cannot change your own role.")


class LastAdminError(ConflictError):
    """A role change would leave the system with no administrators.

    ``UserService`` counts admins before demoting one and refuses when the
    result would be zero. The check-then-write is not fully atomic (design spec,
    decision 6); the residual race is accepted. In practice this fires for a
    caller who is an admin *without* a counted ``users`` row — an unrestricted
    standalone admin (``user_id=None``) demoting the sole real admin — since a
    row-backed admin demoting the last admin is themselves and trips
    :class:`CannotChangeOwnRoleError` first.
    """

    def __init__(self) -> None:
        super().__init__("Cannot remove the last administrator.")


class InvalidStateTransitionError(AutoTuneXError):
    """An illegal job or trial status transition was attempted."""

    status_code = HTTPStatus.CONFLICT
    title = "Conflict"

    def __init__(self, current: object, requested: object) -> None:
        super().__init__(f"Cannot transition from {current} to {requested}.")


class JobNotReconcilableError(ConflictError):
    """A job has no gb build to reconcile against.

    Raised when the job has no ``TUNING`` task, or that task carries no
    ``build_id`` — there is nothing to poll granite.build for.
    """

    def __init__(self, job_id: object) -> None:
        super().__init__(f"Job {job_id} has no build to reconcile.")


class BuildReconcileUnavailableError(ServiceUnavailableError):
    """This deployment cannot reconcile builds on demand.

    There is no gbserver build-status reader because ``job_backend`` is not
    ``"llmb"`` (or its shared client is absent). Mirrors
    :class:`LlmNotConfiguredError`: a 503 that says "not configured here", never
    a 500.
    """

    def __init__(self) -> None:
        super().__init__(
            "On-demand reconcile is unavailable: this deployment has no gbserver "
            "reader (job_backend is not 'llmb')."
        )


class BuildReconcileUpstreamError(BadGatewayError):
    """granite.build could not be read while reconciling a job.

    Wraps a reader-level ``BuildStatusError`` (timeout, 404, 401/403, malformed
    body) as a 502 upstream failure, mirroring :class:`LlmUnavailableError`.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(f"Could not reconcile with granite.build: {detail}")


class BuildCancelUpstreamError(BadGatewayError):
    """granite.build refused or failed a build-cancel request.

    Wraps the CLI failure as a 502, mirroring :class:`BuildReconcileUpstreamError`.
    The job is left unchanged so the cancel can be retried.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(f"Could not cancel the build: {detail}")


class JobArtifactsNotReadyError(ConflictError):
    """A job's output artifacts are not available yet.

    The TUNING build has not reached ``success`` (or its task carries no
    ``artifact_uri``), or a local run has not ``completed``. Mirrors the 2025
    API's "assets are not available because the job status is X".
    """

    def __init__(self, job_id: object, status: object) -> None:
        super().__init__(f"Artifacts for job {job_id} are not available yet (status: {status}).")


class JobArtifactsNotFoundError(NotFoundError):
    """No downloadable output artifacts could be located for the job.

    Covers a job with no TUNING task and no local results directory, a
    ``file://`` directory that does not exist, an HF repo whose file tree is
    404, or an ``artifact_uri`` that yields no repository id / path. The detail
    is fixed and non-leaky; the specific location is logged, never returned.
    """

    def __init__(self, detail: str = "No output artifacts were found for this job.") -> None:
        super().__init__(detail)


class ArtifactSourceUnavailableError(BadGatewayError):
    """A job's artifact source exists but could not be read.

    A HuggingFace Hub request failed (transport error, or a non-200/non-404
    status/body) or a local artifact directory could not be read. The raw
    upstream error is logged, never returned — mirroring
    :class:`GbLogsUpstreamError`.
    """

    def __init__(self, detail: str = "The artifact source could not be read.") -> None:
        super().__init__(detail)


_REALM = 'Bearer realm="autotunex"'


def _invalid_token_challenge(description: str) -> str:
    """Build an RFC 6750 §3.1 challenge.

    ``description`` is always a fixed string chosen here, never library text and
    never anything read out of the credential — see the design spec's
    non-disclosure rules.
    """
    return f'{_REALM}, error="invalid_token", error_description="{description}"'


class AuthenticationError(AutoTuneXError):
    """Base class for a rejected or missing credential.

    Catch this to mean "authentication failed for any reason." The three
    concrete errors below are siblings, not a chain: expiry must not be
    catchable as invalidity, or the distinction RFC 6750 needs is lost.
    """

    status_code = HTTPStatus.UNAUTHORIZED
    title = "Unauthorized"


class MissingCredentialsError(AuthenticationError):
    """No credential was presented at all."""

    def __init__(self) -> None:
        super().__init__("Authentication is required.", headers={"WWW-Authenticate": _REALM})


class InvalidCredentialsError(AuthenticationError):
    """A credential was presented but does not verify.

    Also raised for a credential routed to a provider that is not enabled —
    deliberately the same body as a genuinely invalid one, so naming which
    schemes are configured never leaks to a caller.
    """

    def __init__(self) -> None:
        super().__init__(
            "The credential is not valid.",
            headers={"WWW-Authenticate": _invalid_token_challenge("The credential is not valid")},
        )


class ExpiredCredentialsError(AuthenticationError):
    """A credential was well-formed and signed, but is past its expiry.

    A sibling of :class:`InvalidCredentialsError`, not a subclass, so a
    spec-compliant client can tell from ``error_description`` that it should
    refresh rather than re-prompt — and so a test asserting "invalid" can never
    be satisfied by an expired token instead.
    """

    def __init__(self) -> None:
        super().__init__(
            "The access token has expired.",
            headers={"WWW-Authenticate": _invalid_token_challenge("The access token has expired")},
        )


class GbLogsUnavailableError(ServiceUnavailableError):
    """The gb log integration is not configured, or the job has no build to query.

    gb log reading is optional to a deployment (it needs the ``gbcli`` package and
    a cluster token), so an unconfigured integration — or a job not yet launched to
    a build — is a request-time 503, not a startup failure.
    """

    def __init__(self) -> None:
        super().__init__("Live build logs are not available for this job.")


class GbLogsUpstreamError(BadGatewayError):
    """The gb log-query server was unreachable or returned an error.

    Carries only a fixed, safe message; the raw upstream error is logged, never
    returned — mirroring ``LlmUnavailableError``.
    """

    def __init__(self) -> None:
        super().__init__("The build log server could not be reached.")


class ConflictingCredentialsError(AutoTuneXError):
    """Two explicit credentials were presented in the same request."""

    status_code = HTTPStatus.BAD_REQUEST
    title = "Bad Request"

    def __init__(self) -> None:
        super().__init__("Provide exactly one credential.")
