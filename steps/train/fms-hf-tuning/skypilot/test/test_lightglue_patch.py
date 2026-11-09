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

"""patches/lightglue_no_remote_code.py: the CVE-2026-5241 backport.

The string tests need nothing. The exploit tests run against a COPY of the real
transformers 4.55.4 package (CPU venv only; skipped without transformers): a
crafted LightGlue config.json makes AutoModelForCausalLM.from_pretrained run a
model repo's Python without trust_remote_code. Unpatched, the payload runs;
patched, it does not.
"""

import importlib.util
import json
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

_STEP = pathlib.Path(__file__).resolve().parents[1]
_PATCH = _STEP / "patches" / "lightglue_no_remote_code.py"


def _load():
    spec = importlib.util.spec_from_file_location("lightglue_patch", _PATCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PASS_THROUGH = re.compile(r"trust_remote_code=(self|config)\.trust_remote_code")


def _fake_package(tmp_path, patch):
    """A lightglue dir holding exactly the patch's OLD text, for the string tests."""
    d = tmp_path / "lightglue"
    d.mkdir()
    for name, edits in patch.EDITS.items():
        olds = "# middle\n".join(old for old, _new in edits)
        (d / name).write_text("# header\n" + olds + "# footer\n")
    return d


class TestStrings:
    def test_covers_the_files_upstream_fixed(self):
        # huggingface/transformers 676559d5 edits all three; the modular file is
        # never imported, but ships in the wheel with the same two pass-throughs.
        patch = _load()
        assert sorted(patch.EDITS) == [
            "configuration_lightglue.py",
            "modeling_lightglue.py",
            "modular_lightglue.py",
        ]
        modular, config, modeling = (
            patch.EDITS[name]
            for name in (
                "modular_lightglue.py",
                "configuration_lightglue.py",
                "modeling_lightglue.py",
            )
        )
        assert modular == config + modeling

    def test_replaces_every_target(self, tmp_path):
        patch = _load()
        d = _fake_package(tmp_path, patch)
        patch.patch(d)
        for name, edits in patch.EDITS.items():
            text = (d / name).read_text()
            for old, new in edits:
                assert old not in text and new in text, name
                assert "trust_remote_code=" not in new, name
            assert not _PASS_THROUGH.search(text), name

    def test_refuses_a_missing_target(self, tmp_path):
        patch = _load()
        d = _fake_package(tmp_path, patch)
        (d / "modular_lightglue.py").write_text(
            patch.EDITS["modular_lightglue.py"][0][0]
        )
        with pytest.raises(SystemExit, match="not found exactly once"):
            patch.patch(d)

    def test_refuses_a_duplicated_target(self, tmp_path):
        patch = _load()
        d = _fake_package(tmp_path, patch)
        first = next(iter(patch.EDITS))
        old = patch.EDITS[first][0][0]
        (d / first).write_text(old + old)
        with pytest.raises(SystemExit, match="not found exactly once"):
            patch.patch(d)

    def test_writes_nothing_when_any_target_is_missing(self, tmp_path):
        patch = _load()
        d = _fake_package(tmp_path, patch)
        before = {p.name: p.read_text() for p in d.iterdir()}
        (d / "modular_lightglue.py").write_text("# upstream changed\n")
        before["modular_lightglue.py"] = "# upstream changed\n"
        with pytest.raises(SystemExit):
            patch.patch(d)
        assert {p.name: p.read_text() for p in d.iterdir()} == before

    def test_dockerfile_applies_it_after_the_install(self):
        dockerfile = (_STEP / "Dockerfile").read_text()
        install = dockerfile.index("--require-hashes")
        copy = dockerfile.index("COPY patches/lightglue_no_remote_code.py")
        assert install < copy
        assert "lightglue_no_remote_code.py" in dockerfile[copy:]
        # The image ships no bytecode (#616): a .pth imports _distutils_hack at
        # startup, so the patch run must not write any.
        assert (
            "/opt/fms-hf-tuning/.venv/bin/python -B /tmp/lightglue_no_remote_code.py"
            in dockerfile
        )


def _exploit(site_packages: pathlib.Path, work: pathlib.Path) -> bool:
    """Load a crafted LightGlue model with the given transformers; True if its code ran."""
    inner, outer = work / "inner", work / "outer"
    inner.mkdir()
    outer.mkdir()
    marker = work / "MARKER"
    (inner / "config.json").write_text(
        json.dumps(
            {"model_type": "custom_kp", "auto_map": {"AutoConfig": "kp.KPConfig"}}
        )
    )
    (inner / "kp.py").write_text(
        "import pathlib\n"
        f"pathlib.Path({str(marker)!r}).write_text('ran')\n"
        "from transformers import PretrainedConfig\n"
        "class KPConfig(PretrainedConfig):\n"
        "    model_type = 'custom_kp'\n"
    )
    (outer / "config.json").write_text(
        json.dumps(
            {
                "model_type": "lightglue",
                "trust_remote_code": True,
                "keypoint_detector_config": {
                    "model_type": "custom_kp",
                    "_name_or_path": str(inner),
                },
            }
        )
    )
    # As fms-hf-tuning loads a base model: no trust_remote_code argument.
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(site_packages)!r})\n"
        "from transformers import AutoModelForCausalLM\n"
        "try:\n"
        f"    AutoModelForCausalLM.from_pretrained({str(outer)!r})\n"
        "except Exception:\n"
        "    pass\n"
    )
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(work),
        "HF_HUB_OFFLINE": "1",
        "HF_MODULES_CACHE": str(work / "modules"),
    }
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=work,
        env=env,
        capture_output=True,
        check=False,
        timeout=300,
    )
    return marker.exists()


@pytest.fixture(name="transformers_copy")
def _transformers_copy(tmp_path):
    """A private copy of the installed transformers package, safe to patch."""
    pytest.importorskip("transformers.models.lightglue")
    import transformers  # pylint: disable=import-outside-toplevel

    site = tmp_path / "site"
    shutil.copytree(pathlib.Path(transformers.__file__).parent, site / "transformers")
    return site


class TestExploit:
    def test_unpatched_copy_runs_the_payload(self, transformers_copy, tmp_path):
        work = tmp_path / "w1"
        work.mkdir()
        assert _exploit(transformers_copy, work)

    def test_patched_copy_does_not(self, transformers_copy, tmp_path):
        _load().patch(transformers_copy / "transformers" / "models" / "lightglue")
        work = tmp_path / "w2"
        work.mkdir()
        assert not _exploit(transformers_copy, work)

    def test_patched_copy_keeps_no_pass_through(self, transformers_copy):
        # The image check: no lightglue source file passes trust_remote_code on.
        lightglue = transformers_copy / "transformers" / "models" / "lightglue"
        _load().patch(lightglue)
        left = [
            p.name
            for p in lightglue.glob("*.py")
            if _PASS_THROUGH.search(p.read_text())
        ]
        assert not left
