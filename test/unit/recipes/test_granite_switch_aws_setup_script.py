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

"""recipes/granite-switch/skypilot/aws/setup-local-space.sh, run against a
throwaway GB_HOME_DIR.

The script turns the shipped environment template into a private copy with the
user's EFS id, writes the standalone space that serves it, and records its
ENVIRONMENT_URI in a params file the recipes read. The copy must validate as an
environment, and the recipes must render with that URI.
"""

import pathlib
import stat
import subprocess

import pytest
import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters
from gbserver.types.environmentconfig import EnvironmentConfig

_REPO = pathlib.Path(__file__).resolve().parents[3]
_AWS = _REPO / "recipes" / "granite-switch" / "skypilot" / "aws"
_SCRIPT = _AWS / "setup-local-space.sh"
_FS = "fs-0123456789abcdef0"


def _home(tmp_path, region, fs=_FS, name=None):
    env = tmp_path / "gbhome" / "local-env"
    env.mkdir(parents=True, exist_ok=True)
    ids = env / (name or f"gb-shared-{region}.ids")
    ids.write_text(f"REGION={region}\nVPC=vpc-x\nSG=sg-x\nFS={fs}\n")
    return tmp_path / "gbhome"


def _run(home, region):
    return subprocess.run(
        ["bash", str(_SCRIPT), region],
        env={"GB_HOME_DIR": str(home), "PATH": "/usr/bin:/bin", "HOME": str(home)},
        capture_output=True,
        text=True,
        check=False,
    )


def _params(home):
    lines = (home / "local-env" / "granite-switch.params").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines if line)


def _env(home, name):
    path = home / "local-space/overlay/environments/skypilot-local" / name
    return path / "environment.yaml"


@pytest.mark.parametrize(
    "region,name",
    [("us-east-1", "aws-gb-shared"), ("us-west-2", "aws-gb-shared-usw2")],
)
def test_writes_the_environment_space_and_params(tmp_path, region, name):
    home = _home(tmp_path, region)
    proc = _run(home, region)
    assert proc.returncode == 0, proc.stderr

    env_file = _env(home, name)
    data = yaml.safe_load(env_file.read_text())
    EnvironmentConfig.model_validate(data)
    config = data["config"]
    assert data["name"] == f"skypilot-local-{name}"
    assert config["shared_filesystem"]["efs"]["file_system_id"] == _FS
    assert config["shared_filesystem"]["efs"]["region"] == region
    for step in config["steps"].values():
        assert step["launcher_config"]["resources"]["infra"] == f"aws/{region}"
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600

    space = yaml.safe_load((home / "local-space/space.yaml").read_text())
    assert space["name"] == "public"
    assert space["base_uris"] == [
        f"file://{home}/local-space/overlay",
        f"file://{_REPO}/configurations/assets",
    ]

    uri = f"space://environments/skypilot-local/{name}"
    assert _params(home) == {"ENVIRONMENT_URI": uri}
    assert f"ENVIRONMENT_URI={uri}" in proc.stdout
    assert "gbserver standalone --space-dir" in proc.stdout


def test_another_region_starts_from_the_us_east_1_template(tmp_path):
    home = _home(tmp_path, "eu-west-1")
    assert _run(home, "eu-west-1").returncode == 0
    text = _env(home, "aws-gb-shared-eu-west-1").read_text()
    assert "us-east-1" not in text
    config = yaml.safe_load(text)["config"]
    assert config["shared_filesystem"]["efs"]["region"] == "eu-west-1"
    assert {
        s["launcher_config"]["resources"]["infra"] for s in config["steps"].values()
    } == {"aws/eu-west-1"}


def test_finds_an_ids_file_by_its_region_line(tmp_path):
    home = _home(tmp_path, "us-east-1", name="gb-shared-use1.ids")
    proc = _run(home, "us-east-1")
    assert proc.returncode == 0, proc.stderr
    assert "gb-shared-use1.ids" in proc.stdout


def test_a_rerun_keeps_other_params_and_moves_the_uri(tmp_path):
    home = _home(tmp_path, "us-east-1")
    _home(tmp_path, "us-west-2")
    assert _run(home, "us-east-1").returncode == 0
    params = home / "local-env" / "granite-switch.params"
    params.write_text(params.read_text() + "DATA_URI=hf:///datasets/o/r\n")
    assert _run(home, "us-west-2").returncode == 0
    assert _params(home) == {
        "DATA_URI": "hf:///datasets/o/r",
        "ENVIRONMENT_URI": "space://environments/skypilot-local/aws-gb-shared-usw2",
    }
    assert _env(home, "aws-gb-shared").is_file()


def test_an_existing_space_without_the_overlay_is_left_alone(tmp_path):
    home = _home(tmp_path, "us-east-1")
    space = home / "local-space" / "space.yaml"
    space.parent.mkdir(parents=True)
    space.write_text("name: mine\n")
    proc = _run(home, "us-east-1")
    assert proc.returncode == 0
    assert space.read_text() == "name: mine\n"
    assert "does not list the overlay" in proc.stderr


@pytest.mark.parametrize("fs", ["fs-REPLACE-ME", "", "fs-xyz", "sg-0123456789abcdef0"])
def test_refuses_an_id_that_is_not_an_efs_id(tmp_path, fs):
    home = _home(tmp_path, "us-east-1", fs=fs)
    proc = _run(home, "us-east-1")
    assert proc.returncode != 0
    assert "is not an EFS file system id" in proc.stderr
    assert not (home / "local-space").exists()


def test_refuses_without_an_ids_file(tmp_path):
    home = tmp_path / "gbhome"
    proc = _run(home, "us-east-1")
    assert proc.returncode != 0
    assert "README.md, step 2" in proc.stderr


@pytest.mark.parametrize("recipe", ["answerability-lora", "answerability-sr"])
def test_the_recipes_render_with_the_params_it_writes(tmp_path, recipe):
    home = _home(tmp_path, "us-west-2")
    assert _run(home, "us-west-2").returncode == 0
    params = [f"{k}={v}" for k, v in _params(home).items()]
    params += [
        "SOURCE_DATA_URI=hf:///datasets/o/r/main/d",
        "DATA_URI=hf:///datasets/o/r",
    ]
    text = apply_parameters(
        (_AWS / recipe / "build.yaml").read_text(),
        params,
        get_params_from_file(str(_AWS / recipe / "parameters.yaml")),
        str(tmp_path),
    )
    targets = yaml.safe_load(text)["granite.build"]["targets"].values()
    assert {t["environment_uri"] for t in targets} == {
        "space://environments/skypilot-local/aws-gb-shared-usw2"
    }
