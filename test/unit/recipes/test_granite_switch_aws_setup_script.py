#!/usr/bin/env python3

# Copyright Granite.Build Authors
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

The script turns environment-template.yaml (next to it) into a private copy
with the user's EFS id, region and GPU host AMI, writes the standalone space that
serves it, and records its ENVIRONMENT_URI in a params file the recipes read. The
copy must validate as an environment, and the recipes must render with that URI.

Each run gets its own PATH: symlinks to the tools the script uses, plus a stub
`aws` (or none). A real AWS CLI on the host is never reached.
"""

import os
import pathlib
import shutil
import signal
import stat
import subprocess
import time

import pytest
import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters
from gbserver.types.environmentconfig import EnvironmentConfig

_REPO = pathlib.Path(__file__).resolve().parents[3]
_AWS = _REPO / "recipes" / "granite-switch" / "skypilot" / "aws"
_SCRIPT = _AWS / "setup-local-space.sh"
_FS = "fs-0123456789abcdef0"
_AMI = "ami-0123456789abcdef0"
_SSM = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"


def _home(tmp_path, region, fs=_FS, name=None, quote="", ami=None):
    """A GB_HOME_DIR holding one ids file for region, as README.md step 2 writes."""
    env = tmp_path / "gbhome" / "local-env"
    env.mkdir(parents=True, exist_ok=True)
    ids = env / (name or f"gb-shared-{region}.ids")
    pin = f"AMI={ami}\n" if ami is not None else ""
    ids.write_text(f"REGION={quote}{region}{quote}\nVPC=vpc-x\nSG=sg-x\nFS={fs}\n{pin}")
    return tmp_path / "gbhome"


# Every external command the script runs. The test PATH holds only these (and the
# stub `aws`), so a host with a real CLI in /usr/bin cannot leak into a test.
_TOOLS = "basename cat chmod dirname grep head mkdir mktemp mv rm sed tail tr".split()


def _bin(tmp_path, name):
    """A PATH directory with the script's tools linked in and no `aws`."""
    bin_dir = tmp_path / name
    bin_dir.mkdir(exist_ok=True)
    for tool in _TOOLS:
        link = bin_dir / tool
        if not link.exists():
            found = shutil.which(tool, path="/usr/bin:/bin")
            if found is None:
                pytest.skip(f"setup-script tests need {tool} in /usr/bin or /bin")
            link.symlink_to(found)
    return bin_dir


def test_a_tool_missing_from_usr_bin_skips_and_names_it(tmp_path, monkeypatch):
    # On a system without the tools in /usr/bin or /bin (NixOS), the script tests
    # skip with the tool's name rather than fail on symlink_to(None).
    monkeypatch.setattr(shutil, "which", lambda tool, path=None: None)
    with pytest.raises(pytest.skip.Exception, match="basename"):
        _bin(tmp_path, "bin")


def _aws_stub(tmp_path, out=_AMI, rc=0, err=""):
    """A PATH directory with a fake `aws`: it logs its argv, and for
    `ssm get-parameter` prints out (and err on stderr) and exits rc."""
    bin_dir = _bin(tmp_path, "bin")
    stub = bin_dir / "aws"
    stub.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" >> "{tmp_path}/aws.log"\n'
        'case "$1 $2" in "ssm get-parameter")\n'
        f'  printf "%s\\n" "{out}"; [ -n "{err}" ] && printf "%s\\n" "{err}" >&2\n'
        f"  exit {rc} ;;\n"
        "esac\n"
        "exit 9\n"
    )
    stub.chmod(0o755)
    return bin_dir


def _no_aws(tmp_path):
    """A PATH directory with the script's tools and no `aws` at all."""
    return _bin(tmp_path, "noaws")


def _aws_calls(tmp_path):
    """The aws invocations the stub saw, one per line."""
    log = tmp_path / "aws.log"
    return log.read_text().splitlines() if log.exists() else []


def _run(home, region, aws=None, env=None):
    """Run the script for region with only home as GB_HOME_DIR and HOME.

    aws: the PATH directory (default: a stub that returns _AMI next to home).
    env: extra environment variables.
    """
    aws = aws or _aws_stub(home.parent)
    return subprocess.run(
        ["/bin/bash", str(_SCRIPT), region],
        env={"GB_HOME_DIR": str(home), "PATH": str(aws), "HOME": str(home)}
        | (env or {}),
        capture_output=True,
        text=True,
        check=False,
    )


def _params(home):
    """The params file the script wrote, as a dict."""
    lines = (home / "local-env" / "granite-switch.params").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines if line)


def _env(home, name):
    """The environment copy the script writes for environment name."""
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
    assert "REPLACE-ME" not in env_file.read_text()
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


def _host_images(home, name="aws-gb-shared-usw2"):
    """The docker.host_image_id each step type gets, or None."""
    steps = yaml.safe_load(_env(home, name).read_text())["config"]["steps"]
    return {
        slug: (step["launcher_config"].get("docker") or {}).get("host_image_id")
        for slug, step in steps.items()
    }


def test_the_gpu_steps_run_on_the_dlami_host(tmp_path):
    home = _home(tmp_path, "us-west-2")
    proc = _run(home, "us-west-2")
    assert proc.returncode == 0, proc.stderr
    # hf-peft-sft (a container) and byoc (a bare host, or a container with an
    # IMAGE) get the DLAMI as their host; the CPU-only steps keep SkyPilot's
    # default AMI. No step gets an image_id, which would replace its container.
    assert _host_images(home) == {
        "hfpull": None,
        "hfpush": None,
        "command": None,
        "hf-peft-sft": _AMI,
        "byoc": _AMI,
    }
    text = _env(home, "aws-gb-shared-usw2").read_text()
    for slug, step in yaml.safe_load(text)["config"]["steps"].items():
        assert "image_id" not in step["launcher_config"], slug
    assert f"# Host AMI: {_AMI}" in text


def test_looks_the_ami_up_in_the_region_from_aws_ssm(tmp_path):
    home = _home(tmp_path, "eu-west-1")
    proc = _run(home, "eu-west-1")
    assert proc.returncode == 0, proc.stderr
    assert _aws_calls(tmp_path) == [
        f"ssm get-parameter --region eu-west-1 --name {_SSM}"
        " --query Parameter.Value --output text"
        " --cli-connect-timeout 10 --cli-read-timeout 20"
    ]
    assert _host_images(home, "aws-gb-shared-eu-west-1")["byoc"] == _AMI


def test_the_lookup_uses_the_aws_profile_given(tmp_path):
    home = _home(tmp_path, "us-west-2")
    proc = _run(home, "us-west-2", env={"AWS_PROFILE": "gb-skypilot"})
    assert proc.returncode == 0, proc.stderr
    (call,) = _aws_calls(tmp_path)
    assert " --profile gb-skypilot" in call
    assert "profile gb-skypilot" in proc.stdout


def test_an_ami_in_the_ids_file_wins_and_skips_the_lookup(tmp_path):
    pinned = "ami-0f9b1a75c4247001e"
    home = _home(tmp_path, "us-west-2", ami=f'"{pinned}"')
    proc = _run(home, "us-west-2")
    assert proc.returncode == 0, proc.stderr
    assert _aws_calls(tmp_path) == []
    images = _host_images(home)
    assert images["hf-peft-sft"] == images["byoc"] == pinned


@pytest.mark.parametrize(
    "out,rc", [(_AMI, 255), ("", 0), ("None", 0), ("ami-xyz", 0), ("sg-0123abcd", 0)]
)
def test_refuses_when_the_lookup_gives_no_ami(tmp_path, out, rc):
    home = _home(tmp_path, "us-west-2")
    proc = _run(home, "us-west-2", aws=_aws_stub(tmp_path, out=out, rc=rc))
    assert proc.returncode != 0
    assert "AMI=" in proc.stderr
    assert not (home / "local-space").exists()
    assert not (home / "local-env" / "granite-switch.params").exists()


def test_reads_ids_written_with_windows_line_endings(tmp_path):
    # FS= and AMI= go through the same reader, so both take quoting and CRLF.
    home = _home(tmp_path, "us-west-2", ami='"ami-0f9b1a75c4247001e"')
    ids = home / "local-env" / "gb-shared-us-west-2.ids"
    ids.write_bytes(ids.read_bytes().replace(b"\n", b"\r\n"))
    proc = _run(home, "us-west-2")
    assert proc.returncode == 0, proc.stderr
    text = _env(home, "aws-gb-shared-usw2").read_text()
    assert f"file_system_id: {_FS}\n" in text
    assert _host_images(home)["byoc"] == "ami-0f9b1a75c4247001e"


def test_refuses_a_pinned_ami_that_is_not_an_ami_id(tmp_path):
    home = _home(tmp_path, "us-west-2", ami="ami-REPLACE-ME")
    proc = _run(home, "us-west-2")
    assert proc.returncode != 0
    assert "is not an AMI id" in proc.stderr
    assert not (home / "local-space").exists()


def test_a_failed_lookup_shows_the_aws_cli_error_and_profile(tmp_path):
    # An expired login, a wrong profile or a missing ssm:GetParameter permission
    # all fail the lookup; the CLI's own message says which.
    err = "An error occurred (ExpiredTokenException): The security token included in the request is expired"
    home = _home(tmp_path, "us-west-2")
    proc = _run(home, "us-west-2", aws=_aws_stub(tmp_path, out="", rc=254, err=err))
    assert proc.returncode != 0
    assert "ExpiredTokenException" in proc.stderr
    assert "profile default" in proc.stderr
    assert not (home / "local-space").exists()


def test_an_interrupted_lookup_leaves_no_temp_file(tmp_path):
    # Ctrl-C while the AWS CLI hangs (a network that drops AWS traffic) must not
    # leave the captured-stderr temp file behind.
    home = _home(tmp_path, "us-west-2")
    tmp = tmp_path / "tmpdir"
    tmp.mkdir()
    bin_dir = _bin(tmp_path, "hang")
    touch = shutil.which("touch", path="/usr/bin:/bin")
    sleep = shutil.which("sleep", path="/usr/bin:/bin")
    (bin_dir / "aws").write_text(
        f'#!/bin/sh\n{touch} "{tmp_path}/aws.started"\nexec {sleep} 30\n'
    )
    (bin_dir / "aws").chmod(0o755)
    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        ["/bin/bash", str(_SCRIPT), "us-west-2"],
        env={
            "GB_HOME_DIR": str(home),
            "PATH": str(bin_dir),
            "HOME": str(home),
            "TMPDIR": str(tmp),
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    started = tmp_path / "aws.started"
    for _ in range(100):
        if started.exists():
            break
        time.sleep(0.05)
    assert started.exists()
    assert list(tmp.iterdir())  # the stderr temp file exists while aws runs
    os.killpg(proc.pid, signal.SIGINT)  # what a terminal Ctrl-C sends
    assert proc.wait(timeout=20) != 0
    assert not list(tmp.iterdir())


def test_refuses_without_the_aws_cli_unless_an_ami_is_pinned(tmp_path):
    home = _home(tmp_path, "us-west-2")
    proc = _run(home, "us-west-2", aws=_no_aws(tmp_path))
    assert proc.returncode != 0
    assert "no aws CLI" in proc.stderr
    assert "AMI=" in proc.stderr
    assert not (home / "local-space").exists()

    pinned = _home(tmp_path / "pinned", "us-west-2", ami="ami-0f9b1a75c4247001e")
    proc = _run(pinned, "us-west-2", aws=_no_aws(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert _host_images(pinned)["byoc"] == "ami-0f9b1a75c4247001e"


def test_warns_about_environments_written_before_the_host_ami(tmp_path):
    # An environment from an older script has no host AMI: its region still
    # runs GPU steps on SkyPilot's default AMI, where 8-GPU H200s see no GPUs.
    home = _home(tmp_path, "us-west-2")
    old = _env(home, "aws-gb-shared")
    old.parent.mkdir(parents=True)
    old.write_text("name: skypilot-local-aws-gb-shared\nconfig: {}\n")
    proc = _run(home, "us-west-2")
    assert proc.returncode == 0, proc.stderr
    assert "aws-gb-shared has no GPU host AMI" in proc.stderr
    assert "setup-local-space.sh us-east-1" in proc.stderr
    assert "aws-gb-shared-usw2 has no" not in proc.stderr


def test_the_template_names_the_ami_placeholder_only_where_it_is_filled():
    # The template's comments must not carry the placeholder: sed fills every
    # occurrence, so a comment would end up naming the AMI as a placeholder.
    text = (_AWS / "environment-template.yaml").read_text()
    assert text.count("AMI-REPLACE-ME") == 2  # hf-peft-sft and byoc


def test_another_region_gets_its_own_name_from_the_same_template(tmp_path):
    home = _home(tmp_path, "eu-west-1")
    assert _run(home, "eu-west-1").returncode == 0
    text = _env(home, "aws-gb-shared-eu-west-1").read_text()
    assert "REPLACE-ME" not in text
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


@pytest.mark.parametrize("quote", ['"', "'"])
def test_finds_an_ids_file_whose_region_is_quoted(tmp_path, quote):
    home = _home(tmp_path, "us-west-2", name="mine.ids", quote=quote)
    proc = _run(home, "us-west-2")
    assert proc.returncode == 0, proc.stderr
    config = yaml.safe_load(_env(home, "aws-gb-shared-usw2").read_text())["config"]
    assert config["shared_filesystem"]["efs"]["region"] == "us-west-2"


def test_the_template_has_only_placeholders_for_the_ids():
    text = (_AWS / "environment-template.yaml").read_text()
    config = yaml.safe_load(text)["config"]
    assert config["shared_filesystem"]["efs"] == {
        "file_system_id": "fs-REPLACE-ME",
        "region": "REGION-REPLACE-ME",
        "tls": True,
    }
    assert {
        s["launcher_config"]["resources"]["infra"] for s in config["steps"].values()
    } == {"aws/REGION-REPLACE-ME"}
    for slug in ("hf-peft-sft", "byoc"):
        assert config["steps"][slug]["launcher_config"]["docker"] == {
            "host_image_id": "AMI-REPLACE-ME"
        }


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
