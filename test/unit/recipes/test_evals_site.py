# SPDX-License-Identifier: Apache-2.0

"""recipes/evals/: every eval recipe takes where it runs from a site file.

The site keys (the environment, the cluster, the partitions and the GPU model)
live only in a private site file, whose keys and fake values
recipes/evals/site.example.yaml lists; no recipe's parameters.yaml may define
one. The recipes are rendered the way `gb build start -f <build> --param-file <site>` renders them: the recipe's
parameters.yaml, then the site file, through gbcli's own loader and renderer.

The keys share their names with recipes/distill/site.example.yaml and the SLURM
environment template, so the distill site file must serve these recipes too.
"""

import pathlib
import re

import jinja2
import pytest
import yaml
from jinja2 import meta
from libgbtest.site_template import (
    REPO_ROOT,
    leak_hits,
    render,
    undeclared_keys,
    value_keys,
)

from gbcli.services.service_build import get_params_from_file, load_layered_params
from gbcli.utils.buildutil import apply_parameters

_EVALS = REPO_ROOT / "recipes" / "evals"
_SITE = _EVALS / "site.example.yaml"
_DISTILL_SITE = REPO_ROOT / "recipes" / "distill" / "site.example.yaml"

# The keys the eval recipes read from the site file: where the targets run and
# the GPU model they ask for. SCORE_REMOTE_HOST is optional: build-local.yaml
# falls back to CLUSTER without it.
_SITE_KEYS = {
    "ENVIRONMENT_URI",
    "CLUSTER",
    "EVAL_PARTITION",
    "CPU_PARTITION",
    "SCORE_REMOTE_HOST",
    "EVAL_ACCELERATOR",
}
_OPTIONAL_SITE_KEYS = {"SCORE_REMOTE_HOST"}
# The names these keys had before the site file, which must not come back.
_OLD_KEYS = {"QUEUE", "CPU_QUEUE"}

_RECIPE_DIRS = sorted(_EVALS.glob("*/*/skypilot"))
_BUILDS = sorted(_EVALS.glob("*/*/skypilot/build*.yaml"))
_LOCAL_ONLY = [_EVALS / "judged" / "multi-challenge" / "skypilot" / "score-local.yaml"]
_SITE_BUILDS = [b for b in _BUILDS if b not in _LOCAL_ONLY]
_MULTI_CHALLENGE_LOCAL = (
    _EVALS / "judged" / "multi-challenge" / "skypilot" / "build-local.yaml"
)


def _rel(path):
    return str(path.relative_to(_EVALS))


def _load(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_keys(build):
    """Every parameter the build file reads, in a value or a <% %> block."""
    jenv = jinja2.Environment(
        variable_start_string="$${",
        variable_end_string="}",
        block_start_string="<%",
        block_end_string="%>",
    )
    return meta.find_undeclared_variables(jenv.parse(build.read_text(encoding="utf-8")))


def _render(build, tmp_path, sites=(), params=()):
    """`gb build start -f build [--param-file site ...] [--param k=v ...]`'s render."""
    layered = load_layered_params(str(build), [str(s) for s in sites])
    return apply_parameters(
        build.read_text(encoding="utf-8"), list(params), layered, str(tmp_path)
    )


def test_the_globs_see_every_recipe():
    assert len(_RECIPE_DIRS) >= 57
    assert len(_SITE_BUILDS) > len(_RECIPE_DIRS)
    assert all(p.is_file() for p in _LOCAL_ONLY + [_MULTI_CHALLENGE_LOCAL])


def test_example_site_is_a_flat_mapping():
    site = _load(_SITE)
    assert isinstance(site, dict)
    for key in site:
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*", key), key


def test_example_site_has_every_recipe_and_template_key():
    site = set(_load(_SITE))
    needed = _SITE_KEYS | value_keys() | undeclared_keys()
    assert needed <= site, sorted(needed - site)


def test_distill_site_has_every_required_recipe_key():
    site = set(_load(_DISTILL_SITE))
    required = _SITE_KEYS - _OPTIONAL_SITE_KEYS
    assert required <= site, sorted(required - site)


def test_template_renders_from_example(tmp_path):
    out = tmp_path / "rendered.yaml"
    proc = render(_SITE, out)
    assert proc.returncode == 0, proc.stderr
    text = out.read_text(encoding="utf-8")
    assert "$${" not in text and "<%" not in text
    assert yaml.safe_load(text)["config"]["cluster"] == _load(_SITE)["CLUSTER"]


@pytest.mark.parametrize("recipe", _RECIPE_DIRS, ids=_rel)
def test_parameters_yaml_defines_no_site_key(recipe):
    params = set(get_params_from_file(str(recipe / "parameters.yaml")))
    assert not params & (_SITE_KEYS | _OLD_KEYS), sorted(
        params & (_SITE_KEYS | _OLD_KEYS)
    )


@pytest.mark.parametrize("site", [_SITE, _DISTILL_SITE], ids=["evals", "distill"])
@pytest.mark.parametrize("build", _BUILDS, ids=_rel)
def test_recipe_renders_with_the_site_file(build, site, tmp_path):
    text = _render(build, tmp_path, sites=[site])
    assert "$${" not in text and "<%" not in text
    values = _load(site)
    targets = yaml.safe_load(text)["granite.build"]["targets"]
    assert targets
    for name, target in targets.items():
        if target["environment_uri"] in (
            "space://environments/bash",
            "space://environments/docker",
        ):
            continue
        assert target["environment_uri"] == values["ENVIRONMENT_URI"], name
        for step in target["steps"]:
            res = step["config"]["launcher_config"]["resources"]
            assert res["cluster"] == values["CLUSTER"], name
            assert res["zone"] in (
                values["EVAL_PARTITION"],
                values["CPU_PARTITION"],
            ), name
            # A GPU target runs on the eval lane, a CPU-only one on the CPU partition.
            gpu = "accelerators" in res
            assert (
                res["zone"] == values["EVAL_PARTITION" if gpu else "CPU_PARTITION"]
            ), name
            if gpu:
                model, _, count = res["accelerators"].partition(":")
                assert model == values["EVAL_ACCELERATOR"], name
                assert count.isdigit(), name


@pytest.mark.parametrize("build", _SITE_BUILDS, ids=_rel)
def test_recipe_needs_a_site_file(build, tmp_path):
    """Without --param-file the render fails, and only for want of site keys."""
    params = set(get_params_from_file(str(build.parent / "parameters.yaml")))
    missing = _read_keys(build) - params
    assert missing - _OPTIONAL_SITE_KEYS, "reads no site key"
    assert missing <= _SITE_KEYS, sorted(missing - _SITE_KEYS)
    with pytest.raises(jinja2.UndefinedError) as e:
        _render(build, tmp_path)
    assert re.search(rf"'({'|'.join(sorted(_SITE_KEYS))})' is undefined", str(e.value))


@pytest.mark.parametrize("build", _LOCAL_ONLY, ids=_rel)
def test_local_only_recipe_reads_no_site_key(build, tmp_path):
    """score-local.yaml runs only on the gbserver host, so it needs no site file."""
    assert not _read_keys(build) & (_SITE_KEYS | _OLD_KEYS)
    assert "$${" not in _render(build, tmp_path)


def _fetch_host(tmp_path, sites, params=()):
    rendered = yaml.safe_load(
        _render(_MULTI_CHALLENGE_LOCAL, tmp_path, sites=sites, params=params)
    )
    command = rendered["granite.build"]["targets"]["multi-challenge-fetch"]["steps"][0][
        "config"
    ]["command_config"]["command"]
    (host,) = re.findall(r'"([^"\s]+):\$SRC_DIR"', command)
    return host


@pytest.mark.parametrize("site", [_SITE, _DISTILL_SITE], ids=["evals", "distill"])
def test_score_remote_host_defaults_to_cluster(site, tmp_path):
    assert _fetch_host(tmp_path, [site]) == _load(site)["CLUSTER"]


def test_score_remote_host_overrides_cluster(tmp_path):
    host = _fetch_host(tmp_path, [_SITE], ["SCORE_REMOTE_HOST=login.example.org"])
    assert host == "login.example.org"


def test_no_site_specific_tokens_under_recipes_evals():
    files = [p for p in _EVALS.rglob("*") if p.is_file()]
    assert _SITE in files
    hits = leak_hits(files)
    assert not hits, "\n".join(hits)
