"""LlmbTuningLauncher: auth, command assembly, spec persistence, stdout parsing.

``subprocess.run`` is faked throughout, so no real ``llmb`` binary is needed; the
spec file, however, is written for real into pytest's ``tmp_path`` so the
kept-on-disk behaviour can be asserted. The auth token env var is a test-only
name, always cleared or set explicitly per test so an ambient ``GB_TOKEN`` in the
developer's shell cannot change the outcome.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from autotunex.services.launch.llmb import LlmbBuildCanceller, LlmbTuningLauncher
from autotunex.services.launch.protocols import LaunchContext
from autotunex.services.launch.spec import build_spec

TOKEN_ENV = "AUTOTUNEX_LLMB_TEST_TOKEN"

BUILD_ID = UUID("22222222-2222-2222-2222-222222222222")

_OK_STDOUT = json.dumps({"build_url": "https://example.com/granite/pull/42", "uuid": str(BUILD_ID)})

CTX = LaunchContext(
    job_id=UUID("11111111-1111-1111-1111-111111111111"),
    model="ibm/granite",
    model_source="huggingface",
    experiment_name="exp1",
    tuning_type="lora",
    rl_tuner_type=None,
    config_name="my-config",
    config_data={},
    dataset_name="alpaca",
    dataset_uri="s3://data/alpaca",
    data_format="jsonl",
    autotune=True,
    seed=42,
    reward_function_code=None,
    reward_function_name=None,
)


def _spec_builder(ctx: LaunchContext) -> str:
    """A real custom_code builder, so the persisted spec file is non-empty yaml."""
    return build_spec(
        ctx,
        runtime_image="registry/tuner:1",
        trainer_repo="https://example/trainer.git",
        trainer_ref="stage",
        output_uri_root="s3://bucket/runs",
        callback_url=None,
    )


def _launcher(
    spec_dir: Path, *, token_env: str = TOKEN_ENV, tags: str = "autotunex"
) -> LlmbTuningLauncher:
    return LlmbTuningLauncher(
        llmb_command="llmb",
        spec_dir=spec_dir,
        token_env=token_env,
        tags=tags,
        spec_builder=_spec_builder,
    )


async def test_launch_submits_the_builder_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, str] = {}

    def fake_builder(ctx: LaunchContext) -> str:
        return "SPEC-TEXT"

    launcher = LlmbTuningLauncher(
        llmb_command="llmb",
        spec_dir=tmp_path,
        token_env=TOKEN_ENV,
        tags="autotunex",
        spec_builder=fake_builder,
    )

    def fake_run(spec: str, job_id: UUID) -> subprocess.CompletedProcess[str]:
        captured["spec"] = spec
        return subprocess.CompletedProcess([], 0, stdout=_OK_STDOUT, stderr="")

    monkeypatch.setattr(launcher, "_run_llmb", fake_run)

    handle = await launcher.launch(CTX)

    assert captured["spec"] == "SPEC-TEXT"
    assert handle.build_id == BUILD_ID


async def test_launch_invokes_llmb_build_start_and_parses_handle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=_OK_STDOUT, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    handle = await _launcher(tmp_path).launch(CTX)

    assert calls[0][0] == "llmb"
    assert calls[0][1:6] == ["build", "start", "--quiet", "--format", "json"]
    assert calls[0][6] == "-f"
    assert handle.build_id == BUILD_ID
    assert handle.pr_url == "https://example.com/granite/pull/42"


async def test_launch_passes_the_configured_tag_to_build_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=_OK_STDOUT, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    # A comma-separated value is forwarded verbatim; llmb splits it, not us.
    await _launcher(tmp_path, tags="autotunex,exp-42").launch(CTX)

    build_start = calls[0]
    assert build_start[build_start.index("--tag") + 1] == "autotunex,exp-42"


async def test_launch_omits_tag_when_none_is_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=_OK_STDOUT, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    await _launcher(tmp_path, tags="   ").launch(CTX)  # whitespace-only == unset

    assert "--tag" not in calls[0]


async def test_launch_authenticates_before_build_start_when_token_is_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(TOKEN_ENV, "secret-token")
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=_OK_STDOUT, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    await _launcher(tmp_path).launch(CTX)

    assert calls[0] == ["llmb", "auth", "login", "--token", "secret-token"]
    assert calls[1][1:6] == ["build", "start", "--quiet", "--format", "json"]


async def test_launch_auth_failure_raises_without_leaking_the_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(TOKEN_ENV, "secret-token")

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[1:3] == ["auth", "login"]:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="bad token")
        return subprocess.CompletedProcess(args, 0, stdout="queued", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as exc_info:
        await _launcher(tmp_path).launch(CTX)

    message = str(exc_info.value)
    assert "bad token" in message  # llmb's stderr is surfaced
    assert "secret-token" not in message  # the token is never in the error


async def test_launch_skips_auth_when_token_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=_OK_STDOUT, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    await _launcher(tmp_path).launch(CTX)

    assert all(call[1] != "auth" for call in calls)  # no `llmb auth login`
    assert calls[0][1:6] == ["build", "start", "--quiet", "--format", "json"]


async def test_launch_writes_a_persistent_spec_file_keyed_by_job_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    passed_paths: list[str] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        passed_paths.append(args[args.index("-f") + 1])
        return subprocess.CompletedProcess(args, 0, stdout=_OK_STDOUT, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    await _launcher(tmp_path).launch(CTX)

    expected = tmp_path / str(CTX.job_id) / "build.yaml"
    assert passed_paths == [str(expected)]
    assert expected.is_file()  # kept after the run, not a deleted temp file
    assert expected.read_text().strip()  # holds the generated spec


async def test_launch_keeps_the_spec_file_when_the_cli_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError):
        await _launcher(tmp_path).launch(CTX)

    # The spec survives a failed launch — that is precisely when you need it.
    assert (tmp_path / str(CTX.job_id) / "build.yaml").is_file()


async def test_launch_raises_when_the_build_id_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout='{"build_url": "https://x"}', stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="build id"):
        await _launcher(tmp_path).launch(CTX)


async def test_launch_surfaces_llmb_stderr_when_build_start_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        # A non-zero exit from `llmb build start` — subprocess.run RETURNS this
        # (the launcher no longer passes check=True), it is not a raised exception.
        return subprocess.CompletedProcess(
            args, 1, stdout="", stderr="ERROR: unknown step_uri space://steps/autotunex-tune"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as exc_info:
        await _launcher(tmp_path).launch(CTX)

    message = str(exc_info.value)
    assert "unknown step_uri" in message  # llmb's stderr is surfaced, not swallowed
    assert "exit 1" in message


async def test_launch_failure_falls_back_to_stdout_when_stderr_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 2, stdout="detail printed to stdout", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as exc_info:
        await _launcher(tmp_path).launch(CTX)

    assert "detail printed to stdout" in str(exc_info.value)


CANCEL_TOKEN_ENV = "AUTOTUNEX_LLMB_TEST_CANCEL_TOKEN"


async def test_canceller_runs_llmb_build_cancel_with_the_build_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, list[str]] = {}

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        seen["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    monkeypatch.delenv(CANCEL_TOKEN_ENV, raising=False)  # skip auth login
    monkeypatch.setattr(subprocess, "run", fake_run)
    build_id = uuid4()

    await LlmbBuildCanceller(llmb_command="llmb", token_env=CANCEL_TOKEN_ENV).cancel(build_id)

    # --skip-version-check is required: some granite.build CLI builds abort nonzero on
    # their own outdated-version check before doing any work, which would otherwise make
    # every cancel a spurious BuildCancelUpstreamError.
    assert seen["args"] == [
        "llmb",
        "build",
        "cancel",
        "--quiet",
        "--format",
        "json",
        "--skip-version-check",
        str(build_id),
    ]


async def test_canceller_raises_runtimeerror_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="no such build")

    monkeypatch.delenv(CANCEL_TOKEN_ENV, raising=False)
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="no such build"):
        await LlmbBuildCanceller(llmb_command="llmb", token_env=CANCEL_TOKEN_ENV).cancel(uuid4())
