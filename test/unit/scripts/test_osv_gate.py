# SPDX-License-Identifier: Apache-2.0

"""steps/train/distill/images/osv-gate.py: severity gate and ignore-file scoping.

Each test feeds the script a small osv-scanner-shaped JSON report on stdin, as `make
scan-lock-*` and the eval `scan-deps` do, and checks the exit code and the notes.
Bare ids (distill's and hf-export's files) must keep their old meaning: any package,
no expiry. `exp:` and `pkg:` fields narrow an entry.
"""

import importlib.util
import json
import os
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


def package(name, version, *groups, ecosystem="PyPI", score="7.5", end=("fixed", "99.0")):
    """One package with advisory groups: (id, aliases) pairs, each scored `score` (CVSS,
    "" for unscored) and ranged introduced 0 .. `end` (an OSV event: fixed or
    last_affected, and its version)."""
    return {
        "package": {"name": name, "version": version, "ecosystem": ecosystem},
        "vulnerabilities": [
            {
                "id": vid,
                "aliases": aliases,
                "affected": [
                    {
                        "package": {"name": name, "ecosystem": ecosystem},
                        "ranges": [{"type": "ECOSYSTEM", "events": [{"introduced": "0"}, {end[0]: end[1]}]}],
                    }
                ],
            }
            for vid, aliases in groups
        ],
        "groups": [{"ids": [vid], "aliases": [vid, *aliases], "max_severity": score} for vid, aliases in groups],
    }


def run(tmp_path, packages, ignore_text=None, severity="HIGH", releases=None, env=None, extra=()):
    """Run the gate on a report. `releases` ({name: [versions]}) stands in for PyPI's
    release list, so no test reaches the network."""
    args = [sys.executable, "-I", str(SCRIPT), severity, *extra]
    if ignore_text is not None:
        ignore = tmp_path / "scan-ignore"
        ignore.write_text(ignore_text)
        args += ["--ignore", str(ignore)]
    rel = tmp_path / "releases.json"
    rel.write_text(json.dumps(releases or {}))
    full_env = {**os.environ, "OSV_GATE_RELEASES": str(rel), **(env or {})}
    report = {"results": [{"packages": packages}]}
    return subprocess.run(args, input=json.dumps(report), capture_output=True, text=True, check=False, env=full_env)


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


def test_scope_compares_the_name(tmp_path):
    """Same type and version, another package: not covered."""
    r = run(tmp_path, [package("vllm", "11.3.0", PILLOW)], "PYSEC-2026-4327 pkg:pypi/pillow@11.3.0\n")
    assert r.returncode == 1
    assert "not vllm 11.3.0" in r.stdout


@pytest.mark.parametrize(
    ("severity", "score", "fails"),
    [
        ("HIGH", "7.0", True),  # the floor itself is HIGH
        ("HIGH", "6.9", False),
        ("CRITICAL", "9.0", True),
        ("CRITICAL", "8.9", False),
        ("MEDIUM", "4.0", True),
    ],
)
def test_severity_floor_is_inclusive(tmp_path, severity, score, fails):
    r = run(tmp_path, [package("pillow", "11.3.0", PILLOW, score=score)], severity=severity)
    assert (r.returncode == 1) is fails, r.stdout


def test_unscored_fixable_finding_fails_closed(tmp_path):
    r = run(tmp_path, [package("pillow", "11.3.0", PILLOW, score="")])
    assert r.returncode == 1
    assert "UNKNOWN" in r.stdout and "pillow 11.3.0 -> 99.0" in r.stdout


def test_unscored_finding_without_a_fix_passes(tmp_path):
    r = run(tmp_path, [package("pillow", "11.3.0", PILLOW, score="", end=("last_affected", "11.3.0"))], releases={"pillow": ["11.3.0"]})
    assert r.returncode == 0, r.stdout


def test_expiry_uses_the_utc_date_not_the_local_one(tmp_path):
    """Run the gate in a time zone whose local date differs from UTC's right now, with
    an entry whose expiry sits between the two dates: only the UTC rule gives the
    expected verdict. UTC-12 is a day behind until 12:00 UTC, UTC+14 a day ahead from
    10:00 UTC, so one of them always differs."""
    now = osv_gate.datetime.datetime.now(osv_gate.datetime.timezone.utc)
    utc_date = now.date()
    if now.hour < 12:
        tz, exp, expect_expired = "Etc/GMT+12", utc_date, True  # local date is yesterday
    else:
        tz, exp, expect_expired = "Etc/GMT-14", utc_date + osv_gate.datetime.timedelta(days=1), False  # local is tomorrow
    r = run(
        tmp_path,
        [package("pillow", "11.3.0", PILLOW)],
        f"PYSEC-2026-4327 exp:{exp.isoformat()} pkg:pypi/pillow@11.3.0\n",
        env={"TZ": tz},
    )
    if expect_expired:
        assert r.returncode == 1 and "expired on" in r.stdout, r.stdout
    else:
        assert r.returncode == 0 and "IGNORED" in r.stdout, r.stdout


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


TRANSFORMERS = ("PYSEC-2026-4174", ["GHSA-x9r9-c232-4q39"])


def test_last_affected_with_a_later_release_is_fixable(tmp_path):
    """A record ranged introduced..last_affected is fixed by any later final release."""
    pkg = package("transformers", "4.57.6", TRANSFORMERS, score="7.8", end=("last_affected", "5.8.1"))
    r = run(tmp_path, [pkg], releases={"transformers": ["4.57.6", "5.8.1", "5.9.0rc1", "5.9.0", "5.17.0"]})
    assert r.returncode == 1
    assert "transformers 4.57.6 -> >5.8.1 (first: 5.9.0)" in r.stdout


def test_last_affected_without_a_later_release_is_not_fixable(tmp_path):
    pkg = package("sqlitedict", "2.1.0", ("PYSEC-2026-1939", []), score="8.8", end=("last_affected", "2.1.0"))
    r = run(tmp_path, [pkg], releases={"sqlitedict": ["2.0.0", "2.1.0", "2.2.0a1"]})
    assert r.returncode == 0, r.stdout
    assert "sqlitedict 2.1.0 -> no fix" in r.stdout


def test_last_affected_prerelease_is_fixed_by_its_final_release(tmp_path):
    pkg = package("transformers", "4.57.6", ("PYSEC-2025-217", []), score="7.8", end=("last_affected", "5.0.0-rc0"))
    r = run(tmp_path, [pkg], releases={"transformers": ["4.57.6", "5.0.0"]})
    assert r.returncode == 1
    assert "-> >5.0.0-rc0 (first: 5.0.0)" in r.stdout


def test_last_affected_fails_closed_when_releases_are_unknown(tmp_path):
    """A package the release list does not know (or a non-PyPI one) counts as fixable."""
    pkg = package("left-pad", "1.0.0", ("GHSA-aaaa-bbbb-cccc", []), ecosystem="npm", end=("last_affected", "1.0.0"))
    r = run(tmp_path, [pkg])
    assert r.returncode == 1
    assert "-> >1.0.0 (releases unknown)" in r.stdout


def test_last_affected_entry_can_be_accepted(tmp_path):
    pkg = package("transformers", "4.57.6", TRANSFORMERS, score="7.8", end=("last_affected", "5.8.1"))
    r = run(
        tmp_path,
        [pkg],
        "GHSA-x9r9-c232-4q39 exp:2999-12-31 pkg:pypi/transformers@4.57.6\n",
        releases={"transformers": ["5.9.0"]},
    )
    assert r.returncode == 0, r.stdout


@pytest.mark.parametrize(
    ("release", "last", "after"),
    [("5.9.0", "5.8.1", True), ("5.8.1", "5.8.1", False), ("5.0.0", "5.0.0-rc0", True), ("4.9", "5.0.0rc3", False), ("2.1", "2.1.0", False)],
)
def test_is_after(release, last, after):
    assert osv_gate.is_after(release, last) is after


def test_stale_entries_listed_or_summarised(tmp_path):
    """By default each unmatched entry gets a line (distill's per-image files);
    --summarise-stale gives one count. The out-of-scope note stays visible either way."""
    ignore = "PYSEC-2026-4327 pkg:pypi/pillow@12.3.0\nCVE-1\nCVE-2\nCVE-3\n"
    pkgs = [package("pillow", "11.3.0", PILLOW)]
    full = run(tmp_path, pkgs, ignore)
    assert full.stdout.count("matches no finding; remove it if fixed") == 3
    short = run(tmp_path, pkgs, ignore, extra=("--summarise-stale",))
    assert "matches no finding; remove it if fixed" not in short.stdout
    assert "3 entries in" in short.stdout and "match no finding in this scan" in short.stdout
    for r in (full, short):
        assert "PYSEC-2026-4327 in" in r.stdout and "accepted for other packages only, not pillow 11.3.0" in r.stdout
        assert r.returncode == 1


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
