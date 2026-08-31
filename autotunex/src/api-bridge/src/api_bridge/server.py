# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

# Standard library
import hmac
import logging
import os
import secrets
from contextlib import asynccontextmanager
from typing import Any

# Third-party
import jwt
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from starlette.middleware.cors import CORSMiddleware

# Local
from api_bridge import database, dependencies, log_service
from api_bridge import model as bridge_models
from api_bridge import models as base_models
from api_bridge.logging_config import setup_logging
from api_bridge.middleware import RequestLoggingMiddleware
from api_bridge.services import (
    config_service,
    dataset_service,
    github_service,
    job_service,
    user_service,
)

# Configure logging before anything else logs, so every getLogger(__name__)
# in the codebase inherits our stdout handler and format.
setup_logging()

db: database.Database = database.Database()
log: log_service.LogService = log_service.LogService(db)
logger = logging.getLogger("api-bridge-server")


@asynccontextmanager
async def startup_event(app: FastAPI):
    async def startup():
        logger.info("Testing DB connection")
        db.test_db_connection_and_structure()
        logger.info("Startup complete")

    async def shutdown():
        logger.info("Shutting down")

    await startup()
    yield
    await shutdown()


app = FastAPI(
    title="AutotuneX Logging Server",
    description="REST API for logging",
    version="0.0.1",
    terms_of_service="IBM Research",
    contact={
        "name": "IBM Research",
        "url": "https://github.com/ibm-granite/granite.build/tree/main/autotunex",
    },
    license_info={
        "name": "IBM Research",
        "url": "https://github.com/ibm-granite/granite.build/tree/main/autotunex",
    },
    servers=[
        {
            "url": os.getenv("AUTOTUNE_SERVER_URL", "http://localhost:8000"),
            "description": "AutoTune Server URL",
        }
    ],
    openapi_url="/fmtune/openapi.json",
    redoc_url="/fmtune/docs",
    docs_url="/fmtune/try",
    lifespan=startup_event,
)

prefix_router = APIRouter(prefix="/fmtune")


def _parse_cors_origins(raw: str | None) -> list[str]:
    """Parse a comma-separated CORS allowlist, defaulting to none.

    This service is called machine-to-machine from a remote tuning cluster, never
    from a browser, so the safe default is an empty allowlist. The previous
    ``allow_origins=["*"]`` combined with ``allow_credentials=True`` is a CORS
    misconfiguration: Starlette echoes the request's ``Origin`` back verbatim
    when the origin list is a wildcard, which — together with credentials —
    would let any cross-origin page make credentialed requests and read the
    response. Set ``API_BRIDGE_CORS_ORIGINS`` explicitly if a browser caller is
    ever added; never restore the wildcard alongside credentials.
    """
    if not raw:
        return []
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


# Request logging runs outermost (added last) so it observes the final status
# code, including responses shaped by CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(os.getenv("API_BRIDGE_CORS_ORIGINS")),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
app.add_middleware(RequestLoggingMiddleware)

SESSION_COOKIE = "autotunex_session"
SESSION_SECRET = os.getenv("SESSION_SECRET", "")
OIDC_ENABLED = bool(
    os.getenv("OIDC_CLIENT_ID")
    and os.getenv("OIDC_CLIENT_SECRET")
    and os.getenv("OIDC_SECURITY_ENDPOINT")
)

if not SESSION_SECRET:
    SESSION_SECRET = secrets.token_urlsafe(32)
    logger.warning("SESSION_SECRET not set — using random value")


def decode_session_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SESSION_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None


def _email_from_request(request: Request) -> str | None:
    """Pull the caller's email off the request.

    fm-tune's AutoTuneXAPI client sends it on every request, both as the
    ``X-User-Email`` header and as the ``email`` cookie. Header wins; cookie is
    the fallback. Returns None when neither is present.
    """
    return request.headers.get("X-User-Email") or request.cookies.get("email")


async def get_current_user(request: Request) -> base_models.AuthUser:
    if not OIDC_ENABLED:
        # Trust the email supplied by the (trusted, internal) caller. Fall back
        # to DEV_USER_EMAIL only when the request carries no identity.
        dev_role = os.getenv("DEV_USER_ROLE", "admin")
        email = _email_from_request(request) or os.getenv("DEV_USER_EMAIL", "dev@example.com")
        return base_models.AuthUser(email=email, role=base_models.Roles(dev_role))

    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    claims = decode_session_token(token)
    if not claims:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired or invalid",
        )

    return base_models.AuthUser(
        email=claims.get("impersonating", claims["email"]),
        role=base_models.Roles(claims["role"]),
    )


# API_BRIDGE_TOKEN gates the 4 write routes below (record_logs, insert_trials,
# update_status, insert_trial_result). Unlike get_current_user, this is checked
# once at import time (not per request) so the log line below fires exactly once
# at startup rather than flooding the log on every unauthenticated request.
API_BRIDGE_TOKEN = os.getenv("API_BRIDGE_TOKEN")

if not API_BRIDGE_TOKEN:
    # This bridge is the tuning pipeline's write-path to production MySQL, reachable
    # over the network from a remote cluster with no browser/CORS boundary to lean
    # on. Without a token, any network caller that can reach this service can write
    # arbitrary logs/trials/status/results. Warn loudly once at startup — mirroring
    # the main service's allow_insecure_no_auth warning — rather than failing
    # closed, so the existing pipeline keeps working until a token is configured.
    logger.warning(
        "API_BRIDGE_TOKEN is not set: write endpoints are UNAUTHENTICATED. "
        "Set it (and configure the caller) to require auth."
    )


def require_write_token(request: Request) -> None:
    """Require ``Authorization: Bearer <API_BRIDGE_TOKEN>`` on a write route.

    Backward-compatible by design: when ``API_BRIDGE_TOKEN`` is unset, every
    request is allowed through (the startup warning above already flags that
    state). When it is set, the header must be present and match via
    ``hmac.compare_digest`` — a constant-time comparison, so a mismatched token
    can't be brute-forced by timing how quickly ``==`` bails out on the first
    differing byte.
    """
    if not API_BRIDGE_TOKEN:
        return

    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token, API_BRIDGE_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )


@prefix_router.post("/api/record_logs", tags=["Utils"], dependencies=[Depends(require_write_token)])
async def record_logs(logs: list[bridge_models.LogEntry]):
    """
    Record Log Entries
    """
    return await log.record_logs(logs)


@prefix_router.post(
    "/api/record_trial", tags=["Utils"], dependencies=[Depends(require_write_token)]
)
async def insert_trials(config: bridge_models.Trial):
    """
    Insert job trials
    """
    return await log.insert_trial(data=config)


@prefix_router.post(
    "/api/update_status", tags=["Utils"], dependencies=[Depends(require_write_token)]
)
async def update_status(data: bridge_models.UpdateStatus):
    """
    Update job and trial status
    """
    result = await log.status_updates(data=data)
    return result


@prefix_router.post(
    "/api/insert_trial_result", tags=["Utils"], dependencies=[Depends(require_write_token)]
)
async def insert_trial_result(data: dict[str, Any]):
    """
    insert trial result
    """

    result = await log.insert_trial_results(id=data["id"], result=data)
    return result


@prefix_router.get(
    "/api/configs",
    tags=["Configurations"],
    summary="List all configurations",
    response_description="Array of configuration objects for the current user",
)
def get_configs(
    auth_user: base_models.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    configuration: config_service.Config = Depends(dependencies.get_config_service),
) -> list[base_models.SimpleConfiguration]:
    """
    Retrieve all configurations created by the authenticated user.

    Returns a list of saved configurations that can be used to start
    new fine-tuning jobs. Each configuration includes:
    - Configuration ID and name
    - Tuner type and settings
    - Hyperparameter search space
    - Creation timestamp

    **Returns:**
    - Array of configuration objects

    **Use Case:**
    - View existing configurations before starting a job
    - Reuse proven configurations for new experiments
    """
    db_user = user.get_user(auth_user.email)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    user_id = db_user["id"]
    logger.debug("Listing configs for user_id=%s", user_id)
    return configuration.get_configs(user_id=user_id)


@prefix_router.get(
    "/api/config/{config_name}",
    tags=["Configurations"],
    summary="Get a configuration by name",
    response_description="Configuration object matching the name for the current user",
)
def get_config_by_name(
    config_name: str,
    auth_user: base_models.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    configuration: config_service.Config = Depends(dependencies.get_config_service),
):
    """
    Retrieve a configuration by name for the authenticated user.
    """
    db_user = user.get_user(auth_user.email)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    user_id = db_user["id"]
    config = configuration.get_config_by_name(user_id=user_id, config_name=config_name)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Configuration '{config_name}' not found")
    return config


@prefix_router.post(
    "/api/config",
    tags=["Configurations"],
    summary="Create or update a configuration",
    response_description="Response with configuration ID and status",
)
async def create_config(
    config: base_models.Configuration,
    auth_user: base_models.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    configuration: config_service.Config = Depends(dependencies.get_config_service),
) -> base_models.Response:
    """
    Create a new configuration or update an existing one.

    Configurations define the hyperparameter search space and tuning strategy
    for fine-tuning jobs. They include:
    - **name**: Unique configuration name
    - **tuner_type**: Type of HPO algorithm (Bayesian, Grid Search, etc.)
    - **config_data**: Hyperparameter definitions and ranges

    **Request Body Example:**
    ```json
    {
      "name": "my-lora-config",
      "tuner_type": "bayesian",
      "config_data": {
        "learning_rate": {"min": 1e-5, "max": 1e-3},
        "lora_rank": {"values": [8, 16, 32]},
        "batch_size": {"values": [4, 8, 16]}
      }
    }
    ```

    **Returns:**
    - Configuration ID and creation status

    **Note:** If config with same name exists, it will be updated
    """
    db_user = user.get_user(auth_user.email)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    user_id = db_user["id"]
    config.user_id = user_id
    return await configuration.push_config(config)


@prefix_router.get(
    "/api/datasets",
    tags=["Data sets"],
    summary="List all datasets",
    response_description="Array of all datasets for the current user",
)
async def get_datasets(
    auth_user: base_models.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    dataset: dataset_service.Dataset = Depends(dependencies.get_dataset_service),
) -> list[base_models.DatasetResponse]:
    """
    Retrieve all datasets created by the authenticated user.
    """
    db_user = user.get_user(auth_user.email)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    user_id = db_user["id"]
    return dataset.get_datasets(user_id)


@prefix_router.post(
    "/api/dataset/register",
    tags=["Data sets"],
    summary="Register a dataset with artifact URL",
    response_description="Registered dataset details",
)
async def register_dataset(
    request: dict[str, Any],
    auth_user: base_models.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    dataset: dataset_service.Dataset = Depends(dependencies.get_dataset_service),
):
    """
    Register a new dataset with optional artifact URL.

    **Request Body:**
    - **name**: Dataset name (required)
    - **description**: Dataset description
    - **artifact_url**: URL to pre-existing dataset artifact
    - **artifact_id**: Artifact identifier
    - **train_records**: Number of training records
    - **train_file_size**: Training file size in bytes
    - **validation_records**: Number of validation records
    - **validation_file_size**: Validation file size in bytes
    """
    db_user = user.get_user(auth_user.email)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    user_id = db_user["id"]
    dataset_info = base_models.DatasetInfo(
        user_id=user_id,
        name=request.get("name"),
        description=request.get("description", ""),
    )
    result = dataset.push_dataset(dataset_info)
    artifact_url = request.get("artifact_url")
    if artifact_url:
        metadata = {
            "artifact_id": request.get("artifact_id", ""),
            "artifact_url": artifact_url,
            "train_records": request.get("train_records", 0),
            "train_file_size": request.get("train_file_size", 0),
            "validation_records": request.get("validation_records", 0),
            "validation_file_size": request.get("validation_file_size", 0),
        }
        db_svc = dependencies.get_database()
        result = db_svc.update_dataset_metadata(
            id=str(result.id), user_id=user_id, metadata=metadata
        )
    return result


@prefix_router.post(
    "/api/job/register",
    tags=["Tunings"],
    summary="Register a job record without starting training",
    response_description="Registered job ID and status",
)
async def register_job(
    config: base_models.TuningConfig,
    auth_user: base_models.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    job: job_service.Job = Depends(dependencies.get_job_service),
):
    """
    Register a new fine-tuning job in the database without launching training.

    Accepts a full TuningConfig and creates a DB record only.
    """
    db_user = user.get_user(auth_user.email)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    user_id = db_user["id"]
    config.user_id = user_id
    build_id = config.build_id
    job_id = job.db.insert_job(config)
    if build_id:
        db_instance = dependencies.get_database()
        db_instance.insert_gb_task(job_id=job_id, build_id=build_id)
    return {"id": str(job_id), "build_id": build_id, "status": "registered"}


@prefix_router.post(
    "/api/job/bootstrap",
    tags=["Tunings"],
    summary="Idempotently register config + dataset + job for a run",
    response_description="Resolved config_id, dataset_id, job_id",
)
async def bootstrap(
    request: bridge_models.BootstrapRequest,
    auth_user: base_models.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    configuration: config_service.Config = Depends(dependencies.get_config_service),
    dataset: dataset_service.Dataset = Depends(dependencies.get_dataset_service),
    job: job_service.Job = Depends(dependencies.get_job_service),
):
    """Register a run idempotently.

    job_id-first: if a job with this id already exists (UX path), return its
    existing config/dataset/job ids. Otherwise (template path) find-or-create
    config, dataset, and the job, then return the resolved ids.
    """
    db_user = user.get_user(auth_user.email)
    if db_user is None:
        # Non-strict: the run owner may not have a row yet (e.g. first remote
        # build for this user). Create it on the fly rather than 404-ing.
        logger.info("bootstrap: user %s not found, creating", auth_user.email)
        user.push_user(auth_user.email)
        db_user = user.get_user(auth_user.email)
        if db_user is None:
            logger.error("bootstrap: failed to create user %s", auth_user.email)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to bootstrap user '{auth_user.email}'",
            )
    user_id = db_user["id"]

    # 1. job_id-first short-circuit (UX path)
    existing_job = job.get_by_id(request.job_id)
    if existing_job is not None:
        return {
            "config_id": str(existing_job["config_id"]),
            "dataset_id": str(existing_job["dataset_id"]),
            "job_id": str(existing_job["id"]),
            "created": False,
        }

    # 2. fresh (template path): find-or-create everything
    config_id = configuration.find_or_create(
        name=request.config.name,
        tuner_type=request.config.tuner_type,
        config_data=request.config.config_data,
        user_id=user_id,
        rl_tuner_type=request.config.rl_tuner_type,
    )
    dataset_id = dataset.find_or_create(
        name=request.dataset.name,
        artifact_uri=request.dataset.artifact_uri,
        user_id=user_id,
    )
    tuning_cfg = base_models.TuningConfig(
        id=request.job_id,
        user_id=user_id,
        config_id=config_id,
        dataset_id=dataset_id,
        model=request.job.model,
        experiment_name=request.job.experiment_name,
        tuning_type=request.job.tuning_type,
        seed=request.job.seed,
        build_id=request.build_id,
    )
    job_id = job.create(tuning_cfg)
    # Link the Granite Build task (template path: build_id == job_id). Mirrors
    # register_job; the UX path returns above and manages gb_tasks via gb_runner.
    if request.build_id:
        db_instance = dependencies.get_database()
        db_instance.insert_gb_task(job_id=job_id, build_id=request.build_id)
    return {
        "config_id": config_id,
        "dataset_id": dataset_id,
        "job_id": job_id,
        "created": True,
    }


@prefix_router.get(
    "/api/user/{build_id}",
    tags=["GB"],
    summary="Get User details for a build",
    response_description="User details JSON",
)
async def get_user_details(
    build_id: str,
    gb: github_service.GitHubService = Depends(dependencies.get_github_service),
):
    """
    Get the user details for a build by build ID.
    """
    return await gb.get_gb_user_details(build_id)


@prefix_router.get("/", tags=["Utils"], deprecated=True, include_in_schema=False)
def root():
    """Redirects to the documentation."""
    return RedirectResponse(url="/fmtune/docs")


app.include_router(prefix_router)

if __name__ == "__main__":
    # log_config=None keeps uvicorn from clobbering our root logging setup with
    # its own dictConfig. reload is a real bool — env vars are strings, so the
    # old `os.getenv("DEV_MODE", False)` was truthy for any value including "".
    uvicorn.run(
        "api_bridge.server:app",
        host="0.0.0.0",
        port=int(os.getenv("API_BRIDGE_SERVER_PORT", 8000)),
        reload=os.getenv("DEV_MODE", "").lower() in ("1", "true", "yes"),
        log_config=None,
    )
