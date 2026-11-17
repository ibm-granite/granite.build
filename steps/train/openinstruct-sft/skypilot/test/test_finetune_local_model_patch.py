# SPDX-License-Identifier: Apache-2.0

"""The build-time finetune.py patch skips the Hub pre-fetch for a local model dir."""

import importlib.util
import pathlib

import pytest

_PATCH = (
    pathlib.Path(__file__).resolve().parents[1] / "patches" / "finetune_local_model.py"
)

_UPSTREAM = """import os
from huggingface_hub import snapshot_download


def main(args, accelerator):
    model_path = args.config_name or args.model_name_or_path
    if model_path and accelerator.is_main_process:
        snapshot_download(model_path, revision=args.model_revision)
    accelerator.wait_for_everyone()
"""


def _load():
    spec = importlib.util.spec_from_file_location("finetune_local_model", _PATCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_patch_skips_local_dirs(tmp_path):
    f = tmp_path / "finetune.py"
    f.write_text(_UPSTREAM)
    _load().patch(f)
    out = f.read_text()
    assert (
        "if model_path and not os.path.isdir(model_path) and accelerator.is_main_process:"
        in out
    )
    compile(out, str(f), "exec")


def test_patch_fails_loudly_when_upstream_changes(tmp_path):
    f = tmp_path / "finetune.py"
    f.write_text(
        _UPSTREAM.replace(
            "accelerator.is_main_process", "accelerator.is_local_main_process"
        )
    )
    with pytest.raises(SystemExit):
        _load().patch(f)


def test_patch_is_not_applied_twice(tmp_path):
    f = tmp_path / "finetune.py"
    f.write_text(_UPSTREAM)
    mod = _load()
    mod.patch(f)
    with pytest.raises(SystemExit):
        mod.patch(f)
