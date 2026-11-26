# SPDX-License-Identifier: Apache-2.0

"""steps/train/distill/images/osv-gate.py: severity gate and ignore-file scoping.

Each test feeds the script a small osv-scanner-shaped JSON report on stdin, as `make
scan-lock-*` and the eval `scan-deps` do, and checks the exit code and the notes.
Bare ids (distill's and hf-export's files) must keep their old meaning: any package,
no expiry. `exp:` and `pkg:` fields narrow an entry.
"""

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "steps" / "train" / "distill" / "images" / "osv-gate.py"

_spec = importlib.util.spec_from_file_location("osv_gate", SCRIPT)
assert _spec is not None and _spec.loader is not None
osv_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(osv_gate)


def package(name, version, *groups, ecosystem="PyPI"):
    """One package with HIGH (7.5), fixable advisory groups: (id, aliases) pairs."""
    return {
        "package": {"name": name, "version": version, "ecosystem": ecosystem},
        "vulnerabilities": [
            {
                "id": vid,
                "aliases": aliases,
                "affected": [
                    {
                        "package": {"name": name, "ecosystem": ecosystem},
                        "ranges": [{"type": "ECOSYSTEM", "events": [{"introduced": "0"}, {"fixed": "99.0"}]}],
                    }
                ],
            }
            for vid, aliases in groups
        ],
        "groups": [{"ids": [vid], "aliases": [vid, *aliases], "max_severity": "7.5"} for vid, aliases in groups],
    }


def run(tmp_path, packages, ignore_text=None, severity="HIGH"):
    args = [sys.executable, "-I", str(SCRIPT), severity]
    if ignore_text is not None:
        ignore = tmp_path / "scan-ignore"
        ignore.write_text(ignore_text)
        args += ["--ignore", str(ignore)]
    report = {"results": [{"packages": packages}]}
    return subprocess.run(args, input=json.dumps(report), capture_output=True, text=True, check=False)


PILLOW = ("PYSEC-2026-4327", ["GHSA-4558-3h8r-v8xp", "CVE-2026-112394"])


def test_unlisted_fixable_high_fails(tmp_path):
    r = run(tmp_path, [package("pillow", "11.3.0", PILLOW)])
    assert r.returncode == 1
    assert "pillow 11.3.0 -> 99.0  PYSEC-2026-4327" in r.stdout


def test_bare_id_accepts_every_package(tmp_path):
    """distill's format: a bare id (or any alias) covers the advisory wherever it is."""
    r = run(
        tmp_path,
        [package("pillow", "11.3.0", PILLOW), package("pillow", "12.3.0", PILLOW)],
        "# comment\nGHSA-4558-3h8r-v8xp\n",
    )
    assert r.returncode == 0, r.stdout
    assert r.stdout.count("IGNORED") == 2


def test_scoped_id_still_fails_in_another_package_or_version(tmp_path):
    ignore = "PYSEC-2026-4327 exp:2999-12-31 pkg:pypi/pillow@11.3.0\n"
    ok = run(tmp_path, [package("pillow", "11.3.0", PILLOW)], ignore)
    assert ok.returncode == 0, ok.stdout
    other_version = run(tmp_path, [package("pillow", "12.3.0", PILLOW)], ignore)
    assert other_version.returncode == 1
    assert "is accepted for other packages only, not pillow 12.3.0" in other_version.stdout
    other_package = run(tmp_path, [package("vllm", "0.27.1", PILLOW)], ignore)
    assert other_package.returncode == 1
    assert "not vllm 0.27.1" in other_package.stdout


def test_scope_without_version_covers_every_version(tmp_path):
    r = run(
        tmp_path,
        [package("pillow", "11.3.0", PILLOW), package("pillow", "12.3.0", PILLOW)],
        "PYSEC-2026-4327 pkg:pypi/pillow\n",
    )
    assert r.returncode == 0, r.stdout


def test_several_scopes_and_pep503_names(tmp_path):
    adv = ("GHSA-xxxx-yyyy-zzzz", [])
    r = run(
        tmp_path,
        [package("Datamodel_Code.Generator", "0.25.7", adv), package("pillow", "12.3.0", adv)],
        "GHSA-xxxx-yyyy-zzzz pkg:pypi/datamodel-code-generator@0.25.7 pkg:pypi/pillow@12.3.0\n",
    )
    assert r.returncode == 0, r.stdout


def test_scope_compares_the_ecosystem(tmp_path):
    adv = ("GHSA-aaaa-bbbb-cccc", [])
    r = run(tmp_path, [package("tar", "6.2.1", adv, ecosystem="npm")], "GHSA-aaaa-bbbb-cccc pkg:pypi/tar@6.2.1\n")
    assert r.returncode == 1
    r = run(tmp_path, [package("tar", "6.2.1", adv, ecosystem="npm")], "GHSA-aaaa-bbbb-cccc pkg:npm/tar@6.2.1\n")
    assert r.returncode == 0, r.stdout


def test_expired_entry_stops_suppressing(tmp_path):
    r = run(tmp_path, [package("pillow", "11.3.0", PILLOW)], "PYSEC-2026-4327 exp:2000-01-01 pkg:pypi/pillow@11.3.0\n")
    assert r.returncode == 1
    assert "PYSEC-2026-4327 in" in r.stdout and "expired on 2000-01-01" in r.stdout
    assert "IGNORED" not in r.stdout


def test_expiry_is_trivys_utc_date_rule():
    entry = osv_gate.parse_ignore_line("CVE-1 exp:2027-05-31")
    assert not osv_gate.is_expired(entry, osv_gate.datetime.date(2027, 5, 30))
    assert osv_gate.is_expired(entry, osv_gate.datetime.date(2027, 5, 31))


@pytest.mark.parametrize(
    "line",
    ["CVE-1 exp:2027-13-01", "CVE-1 pkg:pypi", "CVE-1 pkg:pypi/pillow?arch=x", "CVE-1 until:2027-05-31"],
)
def test_malformed_entry_fails_the_gate(tmp_path, line):
    r = run(tmp_path, [package("pillow", "11.3.0", PILLOW)], line + "\n")
    assert r.returncode != 0
    assert "osv-gate:" in r.stderr and "scan-ignore:1" in r.stderr


def test_load_ignores_keeps_its_set_of_ids():
    """fix-locks.py relies on load_ignores returning every id as a set."""
    assert osv_gate.load_ignores(None) == set()


def test_load_ignores_reads_every_id(tmp_path):
    f = tmp_path / "scan-ignore"
    f.write_text("A\nB exp:2000-01-01\nC pkg:pypi/x@1  # trailing comment\n")
    assert osv_gate.load_ignores(f) == {"A", "B", "C"}


@pytest.mark.parametrize(
    "path",
    [
        "steps/train/distill/images/gpu/scan-ignore",
        "steps/train/distill/images/cpu/scan-ignore",
        "steps/aux/hf-export/skypilot/scan-ignore",
        "steps/eval/scan-ignore",
    ],
)
def test_the_repos_ignore_files_parse(path):
    """No file in the repo has a field the gate rejects (hf-export's has no entries)."""
    osv_gate.load_ignore_entries(REPO_ROOT / path)
