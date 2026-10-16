"""Guards the job-level status ownership invariant.

AutoTune runs as one step of a multi-step granite.build build. A later build
step can fail *after* training succeeds, so AutoTune must NOT report a
job-level terminal status: the job's terminal status is owned by the build
outcome (AutoTuneX's reconcile loop, which maps the build's final state). A
premature COMPLETED here masks a later build failure.

main.py builds all its logic inside ``if __name__ == "__main__":`` and its run
path spins up Ray and reads real files, so it cannot be imported and called in a
unit test. A source-level assertion is the practical guard — the same approach
``test_main_cli.py::test_help_omits_removed_flags`` already uses for removed
CLI flags.

Trial-level status reporting (``tuner_callback.py``, keyed on ``trial_id``) is a
separate, legitimate concern and must remain — the guard below asserts it was
not removed by accident along with the job-level write.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_PY = REPO_ROOT / "main.py"
TUNER_CALLBACK = REPO_ROOT / "autotune" / "callbacks" / "tuner_callback.py"


def test_main_does_not_report_job_level_status():
    source = MAIN_PY.read_text()

    assert "UPDATE_STATUS" not in source, (
        "main.py must not post a job-level status update. AutoTune is one step "
        "of a multi-step build; the job's terminal status is owned by the build "
        "outcome (AutoTuneX's reconcile loop), not by AutoTune."
    )


def test_trial_level_status_reporting_is_preserved():
    source = TUNER_CALLBACK.read_text()

    assert "UPDATE_STATUS" in source, (
        "Trial-level status reporting was removed from tuner_callback.py. Only "
        "the job-level status write in main.py should have been removed."
    )
