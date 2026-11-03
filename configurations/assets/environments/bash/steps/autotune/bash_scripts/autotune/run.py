#!/usr/bin/env python3
"""autotune step driver: build the main.py argv from the environment, run
fm-tune's main.py in $FM_TUNE_ROOT, then emit the column-0 artifact marker.

Runs in the gbserver Bash and Docker environments. command.sh materializes the
tuning config to $AUTOTUNE_CONFIG_FILE and chooses the interpreter; this script
resolves inputs/params from env and shells out to main.py.

NO Jinja tokens (double-brace, percent-brace, hash-brace) may appear in this file:
the whole step dir is rendered with strict=True at run time and this file must pass
through untouched.
"""

import glob
import os
import subprocess
import sys

ARTIFACT_ID = "custom"  # must match outputs.custom in build.yaml


def _bool(env, name, default=False):
    v = env.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def resolve_split(dataset_dir, override, pattern):
    """Explicit override wins (abs or relative to dataset_dir); else the sole glob match.

    A pattern matching several files is an error rather than a silent pick.
    """
    if override:
        path = (
            override if os.path.isabs(override) else os.path.join(dataset_dir, override)
        )
        if not os.path.isfile(path):
            sys.exit("autotune: split override does not exist: " + repr(path))
        return path
    matches = sorted(glob.glob(os.path.join(dataset_dir, pattern)))
    if not matches:
        sys.exit(
            "autotune: no file matching "
            + repr(pattern)
            + " under "
            + repr(dataset_dir)
        )
    if len(matches) > 1:
        sys.exit(
            "autotune: "
            + repr(pattern)
            + " is ambiguous under "
            + repr(dataset_dir)
            + " ("
            + ", ".join(os.path.basename(m) for m in matches)
            + "); set TRAIN_FILE/VAL_FILE to choose"
        )
    return matches[0]


def _algo_defaults(config_file):
    """Read training_config.tuning_algorithm/rl_algorithm defaults from the tuning config.

    Expects the AutoTuneX generator shape, where each key is a spec dict carrying a
    `default` (see AutoTuneX's CreateConfigForm). A raw fm-tune config carries neither
    key -- fm-tune takes both from the CLI -- so this correctly falls through to
    lora/none there, and TUNING_ALGO/RL_ALGO are the way to override.
    """
    try:
        import yaml
    except ImportError:
        print("autotune: pyyaml unavailable; defaulting to lora/none", file=sys.stderr)
        return "lora", "none"
    try:
        with open(config_file) as fh:
            cfg = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        print(
            "autotune: could not read " + repr(config_file) + ": " + str(exc),
            file=sys.stderr,
        )
        return "lora", "none"
    tc = cfg.get("training_config") or {}

    def val(key):
        node = tc.get(key)
        return node.get("default") if isinstance(node, dict) else node

    return (val("tuning_algorithm") or "lora"), (val("rl_algorithm") or "none")


def build_argv(env):
    """Build the main.py argument vector from the environment (pure/testable)."""
    model = env.get("LLMB_BASH_INPUT_MODEL")
    dataset_dir = env.get("LLMB_BASH_INPUT_DATASET_FILES")
    config_file = env.get("AUTOTUNE_CONFIG_FILE")
    output_dir = env.get("LLMB_BASH_OUTPUT_DIR")
    for label, value in (
        ("model input", model),
        ("dataset_files input", dataset_dir),
        ("AUTOTUNE_CONFIG_FILE", config_file),
        ("LLMB_BASH_OUTPUT_DIR", output_dir),
    ):
        if not value:
            sys.exit("autotune: required " + label + " is unset")

    train = resolve_split(dataset_dir, env.get("TRAIN_FILE"), "*_train.jsonl")
    validation = resolve_split(dataset_dir, env.get("VAL_FILE"), "*_validation.jsonl")
    ta_default, rl_default = _algo_defaults(config_file)
    run_name = env.get("RUN_NAME") or env.get("JOB_ID") or "autotune-run"

    argv = [
        "main.py",
        "--config_file",
        config_file,
        "--train_file",
        train,
        "--validation_file",
        validation,
        "--model_name_or_path",
        model,
        "--tuning_algo",
        env.get("TUNING_ALGO") or ta_default,
        "--rl_algo",
        env.get("RL_ALGO") or rl_default,
        "--run_name",
        run_name,
        "--output_dir",
        output_dir,
        "--output_model_name",
        env.get("OUTPUT_MODEL_NAME") or run_name,
        "--backend",
        env.get("BACKEND") or "torch",
    ]
    if env.get("JOB_ID"):
        argv += ["--job_id", env["JOB_ID"]]
    if env.get("AUTOTUNEX_SERVER_URL"):
        argv += ["--autotunex_server_url", env["AUTOTUNEX_SERVER_URL"]]
    if _bool(env, "NO_AUTOTUNE"):
        argv.append("--no_autotune")
    if _bool(env, "CLEANUP"):
        argv.append("--cleanup")
    if _bool(env, "SAVE_HISTORY"):
        argv.append("--save_history")
    return argv


def artifact_marker(output_dir):
    """The column-0 line the bash/docker monitor scrapes to bind the `custom` output."""
    return "GB_ARTIFACT_ID:" + ARTIFACT_ID + " GB_ARTIFACT_PATH:" + output_dir


def main():
    fm_tune_root = os.environ.get("FM_TUNE_ROOT")
    if not fm_tune_root or not os.path.isdir(fm_tune_root):
        sys.exit(
            "autotune: FM_TUNE_ROOT unset or not a directory: " + repr(fm_tune_root)
        )
    argv = build_argv(os.environ)
    proc = subprocess.run([sys.executable] + argv, cwd=fm_tune_root)
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    # Leading newline: the bash monitor's artifact regex is ^-anchored, and main.py
    # (ray/tqdm) may leave a partial line, which would push the marker off column 0
    # and silently lose the output -- skip_finding_output_artifacts removes the
    # fallback scan, so there is no second chance.
    print("\n" + artifact_marker(os.environ["LLMB_BASH_OUTPUT_DIR"]), flush=True)


if __name__ == "__main__":
    main()
