# SPDX-License-Identifier: Apache-2.0

"""recipes/granite-switch/skypilot/slurm/: the example site file.

The recipe runs in an environment rendered from the shared SLURM template
(recipes/environments/skypilot-slurm.template.yaml) and takes its site values
from the same site file. These tests check that the example renders the
template, that it and parameters.yaml together give the recipe every key it
reads, that no key is in both, and that nothing site-specific is committed
under recipes/granite-switch/skypilot/slurm/. test_distill_site.py covers the
template itself.
"""

import os
import pathlib
import re
import subprocess
import sys

import pytest
import yaml
from libgbtest.site_template import (
    LEAK,
    REPO_ROOT,
    SLURM_TEMPLATE,
    leak_hits,
    render,
    undeclared_keys,
    value_keys,
)

from gbcli.services.service_build import get_params_from_file, load_layered_params
from gbcli.utils.buildutil import apply_parameters

_SLURM = REPO_ROOT / "recipes" / "granite-switch" / "skypilot" / "slurm"
_RECIPE = _SLURM / "answerability-lora"
_BUILD = _RECIPE / "build.yaml"
_PARAMS = _RECIPE / "parameters.yaml"
_SITE = _RECIPE / "site.example.yaml"
_SETUP = REPO_ROOT / "scripts" / "setup-local-space.sh"

# The keys parameters.yaml used to default to one cluster's values, and the
# CPU partition prepare runs on.
_RECIPE_SITE_KEYS = {
    "ENVIRONMENT_URI",
    "SOURCE_DATA_URI",
    "DATA_ROOT",
    "ADAPTER_ROOT",
    "ACCELERATORS",
    "CPU_PARTITION",
}


def _site():
    """The example site file, as a dict."""
    with open(_SITE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_example_site_is_a_flat_mapping():
    site = _site()
    assert isinstance(site, dict)
    for key in site:
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*", key), key


def test_template_renders_from_example(tmp_path):
    out = tmp_path / "rendered.yaml"
    proc = render(_SITE, out)
    assert proc.returncode == 0, proc.stderr
    env = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert env["name"] == _site()["ENV_NAME"]
    assert env["config"]["steps"]["hfpull"]["zone"] == _site()["CPU_PARTITION"]


def test_every_template_key_is_in_example_site():
    site = set(_site())
    assert value_keys() <= site, sorted(value_keys() - site)
    assert undeclared_keys() <= site, sorted(undeclared_keys() - site)


def test_site_keys_are_in_the_site_file_only():
    site = set(_site())
    params = set(get_params_from_file(str(_PARAMS)))
    assert _RECIPE_SITE_KEYS <= site, sorted(_RECIPE_SITE_KEYS - site)
    assert not site & params, sorted(site & params)


def test_site_and_parameters_give_the_recipe_every_key():
    reads = undeclared_keys(_BUILD)
    given = set(load_layered_params(str(_BUILD), [str(_SITE)]))
    assert _RECIPE_SITE_KEYS <= reads
    assert reads <= given, sorted(reads - given)


def test_example_site_uris_are_full_uris():
    site = _site()
    for key, value in site.items():
        if key.endswith("_URI"):
            assert re.match(r"[a-z][a-z0-9+]*://", str(value)), (key, value)
    # Plain paths: build.yaml prefixes them with env://.
    for key in ("DATA_ROOT", "ADAPTER_ROOT"):
        assert site[key].startswith("/"), (key, site[key])


def test_output_roots_are_visible_in_the_container():
    """prepare and train run in containers, which see only the mounts and the
    cluster workdir: the example's roots are under one of them."""
    site = _site()
    roots = [m["container_path"] for m in site["CONTAINER_MOUNTS"]]
    roots.append(site["CLUSTER_WORKDIR"])
    for key in ("DATA_ROOT", "ADAPTER_ROOT"):
        path = pathlib.PurePosixPath(site[key])
        assert any(path.is_relative_to(r) for r in roots), (key, site[key])


def test_setup_script_points_the_recipe_at_the_rendered_environment(tmp_path):
    """scripts/setup-local-space.sh on a copy of the example: the recipe then
    renders with ENVIRONMENT_URI naming the environment it wrote."""
    site = tmp_path / "site.yaml"
    site.write_text(_SITE.read_text(encoding="utf-8"), encoding="utf-8")
    space = tmp_path / "space"
    proc = subprocess.run(
        [
            "bash",
            str(_SETUP),
            "--template",
            str(SLURM_TEMPLATE),
            "--site",
            str(site),
            "--name",
            "gs-test",
            "--space-dir",
            str(space),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "GB_PYTHON": sys.executable,
            "GB_HOME_DIR": str(tmp_path / "gbhome"),
            "GB_ENVIRONMENT": "STANDALONE",
        },
    )
    assert proc.returncode == 0, proc.stderr
    env = space / "overlay/environments/skypilot-local/gs-test/environment.yaml"
    assert yaml.safe_load(env.read_text(encoding="utf-8"))["type"] == "Skypilot"

    text = apply_parameters(
        _BUILD.read_text(encoding="utf-8"),
        [],
        load_layered_params(str(_BUILD), [str(site)]),
        str(tmp_path),
    )
    targets = yaml.safe_load(text)["granite.build"]["targets"]
    assert {t["environment_uri"] for t in targets.values()} == {
        "space://environments/skypilot-local/gs-test"
    }


def test_no_site_specific_tokens_under_the_slurm_recipes():
    files = [p for p in _SLURM.rglob("*") if p.is_file()]
    assert _SITE in files
    hits = leak_hits(files)
    assert not hits, "\n".join(hits)


@pytest.mark.parametrize(
    "line",
    [
        'DATA_ROOT: "/proj/x/data"',
        'ENVIRONMENT_URI: "space://environments/skypilot/slurm/bluevela"',
        "User: granitebuild",
    ],
)
def test_leak_guard_catches_the_old_defaults(line):
    """The guard is not vacuous: it catches the kinds of value the recipe and
    its environment used to name (a shared path, the environment, the user)."""
    assert LEAK.search(line), line


def test_leak_guard_allows_the_public_image_org():
    assert "quay.io/granitebuild/" in _PARAMS.read_text(encoding="utf-8")
    assert not leak_hits([_PARAMS])
