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

"""Print the `gb build start` command that re-runs a finished build's failed evals.

Given the id of a finished recipes/distill/distill or distill-eval build, read its
targets, find every benchmark eval that never succeeded, and print the
recipes/distill/distill-eval command that re-runs exactly those:

    failed eval-<name>-ck<N>  EVALS "<N>:<name>,...", on rung N's exported model
    failed eval-bfcl-ck<N>    EVALS "<N>:bfcl"
    lost export-<N>           reported, not planned: there is no model to evaluate

The re-run writes under a NEW experiment name, <EXPERIMENT>-<RERUN_TAG>, never into
the original one: gbserver refuses an artifact URI another build registered (and a
failed eval may have registered its URI), the target still reports SUCCESS, and
everything downstream of it never dispatches. So the re-run's CSVs and combined
table hold the re-run cells only; --whole-rungs re-runs every eval of a rung that
had a failure, so its column is complete on its own.

A per-rung CSV or the combined table that failed while every eval it gates on
succeeded is reported, not planned: distill-eval re-runs evals, not CSVs alone, and
its results are on disk (pass --whole-rungs to re-measure such rungs instead).

"Failed" means no attempt of the target succeeded, which covers targets that never
dispatched. It is the gbserver status only: a sage eval that reported SUCCESS with
another eval's results file is not caught here. The sage-eval step anchors its
results search at the experiment directory, but check a rung's CSV for MISSING
cells, and the eval's own results file, before trusting a SUCCESS.

The build's own parameters come from its archive (parameters-applied.yaml). The
command layers distill-eval's parameters.yaml and the SITE file (--site, never
copied: no site value is put on the command line), and adds a --param for every
recipe knob the build ran with a different value of. It is rendered once before it
is printed, so a command this prints at least renders.

Run from anywhere, with gbserver up (the gb CLI must reach it):

    python recipes/distill/distill-eval/rerun_command.py <build-uuid> [--site <site.yaml>]

Exit status: 0 with a command on stdout, 0 with nothing on stdout when there is
nothing to re-run, 1 on an error, 2 when the build has not finished
(--allow-running overrides).
"""

import argparse
import datetime
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile

import yaml

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_RECIPE = _HERE / "build.yaml"
_PARAMS = _HERE / "parameters.yaml"
_SITE_EXAMPLE = _HERE.parent / "site.example.yaml"
_DEFAULT_SITE = "~/.granite.build/local-env/distill-ibm-site.yaml"

# Set by plan() from the build itself, never copied from parameters-applied.yaml.
_COMPUTED = {
    "RUN_NAME",
    "EXPERIMENT",
    "RERUN_TAG",
    "EXPORTS",
    "EVALS",
    "SCREEN_EVALS",
    "FULL_EVAL_RUNGS",
    "EVAL_CATALOG",
    # distill-eval's own retry policy, sized for preemption, not the build's.
    "MAX_RETRIES",
}

# The target names of recipes/distill/distill/build.yaml's shared eval block.
_EVAL_TARGET = re.compile(r"^eval-(.+)-ck(\d+)$")
_EXPORT_TARGET = re.compile(r"^export-(\d+)$")
_CSV_TARGET = re.compile(r"^export-(sage|bfcl)-ck(\d+)$")
_COMBINED = "export-combined"


class RerunError(Exception):
    pass


def _gb(*args):
    env = dict(os.environ)
    # Unset means PROD, where every server round-trip hangs.
    env.setdefault("GB_ENVIRONMENT", "STANDALONE")
    proc = subprocess.run(
        ["gb", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        check=False,
    )
    if proc.returncode != 0:
        raise RerunError(f"gb {' '.join(args)} failed:\n{proc.stderr.strip()}")
    return proc.stdout


def fetch_build(build_id):
    """The build's status, its target definitions, and its applied parameters."""
    common = ("--format", "json", "-q", "--skip-version-check")
    status = json.loads(_gb("build", "status", build_id, *common))
    definition = json.loads(_gb("build", "describe", build_id, *common))
    with tempfile.TemporaryDirectory(prefix="rerun-command-") as tmp:
        _gb(
            "build",
            "init",
            f"{tmp}/src",
            "--from-build",
            build_id,
            "-q",
            "--skip-version-check",
        )
        applied_path = pathlib.Path(tmp, "src", "parameters-applied.yaml")
        if not applied_path.is_file():
            raise RerunError(
                f"build {build_id}'s archive has no parameters-applied.yaml"
            )
        applied = yaml.safe_load(applied_path.read_text(encoding="utf-8"))
    return status, definition, applied


def _succeeded(status):
    """Target names with at least one successful attempt."""
    return {t["target_name"] for t in status["targets"] if t["status"] == "success"}


def _rung_models(status):
    """Rung -> the URI of the model its evals read.

    A distill build's export-<N> output is the model; a distill-eval build has no
    export target, and its evals' one input artifact is. Any attempt of an eval
    counts: the input is resolved before the eval runs.
    """
    models = {}
    for t in status["targets"]:
        m = _EXPORT_TARGET.match(t["target_name"])
        if m and t["status"] == "success":
            for o in t.get("output_artifacts", []):
                models[int(m.group(1))] = o["uri"]
    for t in status["targets"]:
        m = _EVAL_TARGET.match(t["target_name"])
        inputs = t.get("input_artifacts", [])
        if m and len(inputs) == 1:
            models.setdefault(int(m.group(2)), inputs[0]["uri"])
    return models


def plan(status, definition, catalog, whole_rungs=False):
    """EXPORTS and EVALS for distill-eval, and the notes that go with them.

    `definition` is the build's target list (`gb build describe --format json`), so a
    target that never dispatched counts as failed; `catalog` is distill-eval's
    EVAL_CATALOG, the names EVALS may use. Returns (None, notes) when nothing is to
    re-run.
    """
    ok = _succeeded(status)
    defined = [t["target_name"] for t in definition]
    known = {e["name"] for e in catalog} | {"bfcl"}
    notes = []

    evals_of = {}
    for name in defined:
        m = _EVAL_TARGET.match(name)
        if m:
            evals_of.setdefault(int(m.group(2)), []).append(m.group(1))
    unknown = sorted({e for es in evals_of.values() for e in es} - known)
    if unknown:
        raise RerunError(
            f"evals {unknown} are not in distill-eval's EVAL_CATALOG; was the build "
            "started from another catalog?"
        )

    lost = sorted(
        int(m.group(1))
        for m in map(_EXPORT_TARGET.match, defined)
        if m and m.group(0) not in ok
    )
    if lost:
        notes.append(
            f"rungs {','.join(map(str, lost))}: export failed, so there is no model to "
            "evaluate; nothing here can measure them"
        )

    rerun = {}
    csv_noted = False
    for n in sorted(evals_of):
        if n in lost:
            continue
        failed = [e for e in evals_of[n] if f"eval-{e}-ck{n}" not in ok]
        csv_lost = [
            f"export-{kind}-ck{n}"
            for kind in ("sage", "bfcl")
            if f"export-{kind}-ck{n}" in defined and f"export-{kind}-ck{n}" not in ok
        ]
        if failed and whole_rungs:
            failed = list(evals_of[n])
        if failed:
            rerun[n] = failed
        elif csv_lost:
            if whole_rungs:
                rerun[n] = list(evals_of[n])
            else:
                csv_noted = True
                notes.append(
                    f"{', '.join(csv_lost)} failed with every eval of rung {n} "
                    "succeeded: the results are on disk; re-export them, or pass "
                    "--whole-rungs to re-measure the rung"
                )
    if _COMBINED in defined and _COMBINED not in ok and not rerun and not csv_noted:
        notes.append(
            f"{_COMBINED} failed with every eval and CSV succeeded: re-run the "
            "table alone"
        )

    if not rerun:
        return None, notes

    models = _rung_models(status)
    missing = sorted(n for n in rerun if n not in models)
    if missing:
        raise RerunError(
            f"no model URI is recorded for rungs {missing}: neither an export-<N> "
            "output nor an eval input names one"
        )
    params = {
        "EXPORTS": ",".join(f"{n}={models[n]}" for n in rerun),
        "EVALS": ";".join(f"{n}:{','.join(es)}" for n, es in rerun.items()),
    }
    return params, notes


def overrides(applied, defaults, site_keys):
    """--params for recipe knobs distill-eval reads that the build ran differently.

    Site keys are never carried: they come from the --site file. Nor are lists and
    mappings (EVAL_CATALOG, the site's lists), which a --param cannot spell.
    """
    # Names inside template markers only: the recipe's comments name parameters too.
    text = _RECIPE.read_text(encoding="utf-8")
    markers = re.findall(r"\$\$\{(.*?)\}|<%(.*?)%>", text, re.S)
    code = " ".join(part for pair in markers for part in pair)
    referenced = set(re.findall(r"\b[A-Z][A-Z0-9_]+\b", code))
    out = {}
    for key in sorted(referenced & set(applied) - _COMPUTED - set(site_keys)):
        value = applied[key]
        if isinstance(value, (dict, list)):
            continue
        if key not in defaults or str(defaults[key]) != str(value):
            if value is None:
                value = ""
            elif isinstance(value, bool):
                value = str(value).lower()
            out[key] = str(value)
    return out


def render_check(site_path, params):
    """Render distill-eval as `gb build start` would; the target names it yields."""
    sys.path.insert(0, str(_REPO / "src"))
    from gbcli.services.service_build import get_params_from_file
    from gbcli.utils.buildutil import apply_parameters

    merged = {}
    for layer in (_PARAMS, site_path):
        merged.update(get_params_from_file(str(layer)))
    merged.update(params)
    with tempfile.TemporaryDirectory() as tmp:
        text = apply_parameters(_RECIPE.read_text(encoding="utf-8"), [], merged, tmp)
    return list(yaml.safe_load(text)["granite.build"]["targets"])


def _display(path):
    path = pathlib.Path(path).resolve()
    return str(path.relative_to(_REPO)) if path.is_relative_to(_REPO) else str(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("build_id", help="the full UUID of the build to finish")
    parser.add_argument(
        "--site",
        default=_DEFAULT_SITE,
        help=f"the site file the command layers (default: {_DEFAULT_SITE}); its "
        "values are read for the render check, never printed",
    )
    parser.add_argument(
        "--rerun-tag",
        help="the re-run's RERUN_TAG (default: rerun-<now>); must differ from the "
        "build's own",
    )
    parser.add_argument(
        "--whole-rungs",
        action="store_true",
        help="re-run every eval of a rung with a failure, so its re-run column is complete",
    )
    parser.add_argument("--space", help="passed through to gb build start")
    parser.add_argument(
        "--allow-running",
        action="store_true",
        help="plan against a build that has not finished (its retries may still succeed)",
    )
    args = parser.parse_args(argv)

    site_path = pathlib.Path(os.path.expanduser(args.site))
    if not site_path.is_file():
        print(
            f"rerun_command: no site file at {site_path}; pass --site", file=sys.stderr
        )
        return 1
    try:
        status, definition, applied = fetch_build(args.build_id)
        build_status = status["details"]["status"]
        finished = build_status in ("success", "failed", "cancelled", "invalid")
        if not finished and not args.allow_running:
            print(
                f"build {args.build_id} is {build_status}; its own retries may still "
                "succeed. Re-run this once it finishes, or pass --allow-running.",
                file=sys.stderr,
            )
            return 2
        defaults = yaml.safe_load(_PARAMS.read_text(encoding="utf-8"))
        params, notes = plan(
            status, definition, defaults["EVAL_CATALOG"], whole_rungs=args.whole_rungs
        )
    except RerunError as e:
        print(f"rerun_command: {e}", file=sys.stderr)
        return 1

    for note in notes:
        print(f"NOTE: {note}", file=sys.stderr)
    if params is None:
        print(
            f"build {args.build_id} ({build_status}): no eval to re-run.",
            file=sys.stderr,
        )
        return 0

    tag = args.rerun_tag or datetime.datetime.now().strftime("rerun-%Y%m%d_%H%M")
    old_tag = str(applied.get("RERUN_TAG") or "").strip()
    if tag == old_tag:
        print(f"rerun_command: RERUN_TAG {tag!r} is the build's own", file=sys.stderr)
        return 1
    params = {
        "RUN_NAME": str(applied["RUN_NAME"]),
        "EXPERIMENT": str(applied.get("EXPERIMENT") or "").strip()
        or str(applied["RUN_NAME"]),
        "RERUN_TAG": tag,
        **params,
    }
    site_keys = set(yaml.safe_load(_SITE_EXAMPLE.read_text(encoding="utf-8")))
    site_keys |= set(yaml.safe_load(site_path.read_text(encoding="utf-8")) or {})
    extra = overrides(applied, defaults, site_keys)
    try:
        targets = render_check(site_path, {**extra, **params})
    except Exception as e:  # pylint: disable=broad-except
        print(f"rerun_command: the command does not render: {e}", file=sys.stderr)
        return 1
    bcb = {e["name"] for e in defaults["EVAL_CATALOG"] if e.get("needs_bcb_url")}
    if any(
        _EVAL_TARGET.match(t) and _EVAL_TARGET.match(t).group(1) in bcb for t in targets
    ):
        print(
            "NOTE: BigCodeBench re-runs against the site's OE_EVAL_BCB_API_URL; check "
            "that its bcb-server answers",
            file=sys.stderr,
        )

    print(
        f"build {args.build_id} ({build_status}), experiment "
        f"{params['EXPERIMENT']}-{tag}:",
        file=sys.stderr,
    )
    print(f"  EVALS  {params['EVALS']}", file=sys.stderr)
    if extra:
        print(f"  ran with non-default {', '.join(sorted(extra))}", file=sys.stderr)
    print(
        f"  renders to {len(targets)} targets: {' '.join(sorted(targets))}",
        file=sys.stderr,
    )

    lines = [
        f"gb build start -f {_display(_RECIPE)}",
        f"--parameters-path {shlex.quote(args.site)}",
    ]
    if args.space:
        lines.append(f"--space {shlex.quote(args.space)}")
    lines += [f"--param {shlex.quote(f'{k}={v}')}" for k, v in params.items()]
    lines += [f"--param {shlex.quote(f'{k}={v}')}" for k, v in extra.items()]
    print(" \\\n  ".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
