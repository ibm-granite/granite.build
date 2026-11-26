# SPDX-License-Identifier: Apache-2.0

"""steps/eval/scan-ignore and steps/eval/scripts/trivy-ignore.py.

scan-ignore is the eval gates' one list of accepted advisories. osv-gate.py reads it
directly, and trivy reads the YAML that trivy-ignore.py writes from it, so both gates
enforce each entry's package scope and expiry. These tests check that:
- every entry in the real file has both;
- the generated YAML keeps every entry with its scope, expiry and section heading;
- the YAML is in the shape trivy's IgnoreConfig decodes (pkg/result/ignore.go in
  trivy 0.76.2: vulnerabilities: [{id, purls, expired_at, statement}]).
"""

import datetime
import importlib.util
import pathlib
import subprocess
import sys

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "steps" / "eval" / "scripts" / "trivy-ignore.py"
SCAN_IGNORE = REPO_ROOT / "steps" / "eval" / "scan-ignore"

_spec = importlib.util.spec_from_file_location("trivy_ignore", SCRIPT)
assert _spec is not None and _spec.loader is not None
trivy_ignore = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trivy_ignore)
osv_gate = trivy_ignore.osv_gate


def generate(tmp_path, source):
    out = tmp_path / "ignore.yaml"
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), str(source), str(out)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    return yaml.safe_load(out.read_text())


def test_every_eval_entry_is_scoped_and_expires():
    entries = osv_gate.load_ignore_entries(SCAN_IGNORE)
    assert entries
    unscoped = [e["line"] for e in entries if not e["purls"]]
    assert not unscoped, f"entries without a pkg: scope: {unscoped}"
    no_expiry = [e["line"] for e in entries if e["expires"] is None]
    assert not no_expiry, f"entries without exp: {no_expiry}"
    versionless = [e["line"] for e in entries if any(v is None for _, _, v in e["purls"])]
    assert not versionless, f"entries scoped to every version of a package: {versionless}"


def test_each_id_is_listed_once():
    ids = [e["id"] for e in osv_gate.load_ignore_entries(SCAN_IGNORE)]
    assert len(ids) == len(set(ids))


def test_generated_yaml_keeps_every_entry(tmp_path):
    entries = osv_gate.load_ignore_entries(SCAN_IGNORE)
    doc = generate(tmp_path, SCAN_IGNORE)
    assert set(doc) == {"vulnerabilities"}
    items = doc["vulnerabilities"]
    assert len(items) == len(entries)
    for e, item in zip(entries, items, strict=True):
        assert set(item) <= {"id", "purls", "expired_at", "statement"}
        assert item["id"] == e["id"]
        assert item.get("purls", []) == [trivy_ignore.purl_string(p) for p in e["purls"]]
        # yaml decodes the bare date as a date, as trivy's yaml.v3 decodes it into time.Time
        assert item.get("expired_at") == e["expires"]
        assert isinstance(item["statement"], str) and item["statement"]


def test_generated_yaml_shape(tmp_path):
    src = tmp_path / "scan-ignore"
    src.write_text(
        "# ---- pillow 11.3.0 (judged) ----\n"
        "# why\n"
        "PYSEC-2026-4327 exp:2027-05-31 pkg:pypi/pillow@11.3.0 pkg:pypi/pillow@12.3.0\n"
        "# ---- second: \"quoted\" ----\n"
        "CVE-2026-1 pkg:pypi/Datamodel_Code_Generator@0.25.7\n"
    )
    doc = generate(tmp_path, src)
    assert doc["vulnerabilities"] == [
        {
            "id": "PYSEC-2026-4327",
            "purls": ["pkg:pypi/pillow@11.3.0", "pkg:pypi/pillow@12.3.0"],
            "expired_at": datetime.date(2027, 5, 31),
            "statement": "pillow 11.3.0 (judged)",
        },
        {
            "id": "CVE-2026-1",
            "purls": ["pkg:pypi/datamodel-code-generator@0.25.7"],
            "statement": 'second: "quoted"',
        },
    ]


def test_empty_file_gives_an_empty_list(tmp_path):
    src = tmp_path / "scan-ignore"
    src.write_text("# nothing accepted\n")
    assert generate(tmp_path, src) == {"vulnerabilities": []}


def test_refuses_a_non_yaml_output_name(tmp_path):
    r = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), str(SCAN_IGNORE), str(tmp_path / "ignore.txt")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode != 0
    assert "usage" in r.stderr


def test_refuses_a_malformed_entry(tmp_path):
    src = tmp_path / "scan-ignore"
    src.write_text("CVE-1 pkg:pypi\n")
    r = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), str(src), str(tmp_path / "ignore.yaml")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode != 0
    assert "unsupported purl" in r.stderr
