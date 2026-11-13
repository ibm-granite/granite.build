# Copyright 2026 Granite.Build Authors
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
"""Gate an osv-scanner JSON report on severity, like trivy's --severity --ignore-unfixed.

osv-scanner itself exits 1 on ANY finding and has no severity filter, so `make
scan-lock-<x>` pipes its JSON here. Usage:

    osv-scanner scan source --lockfile uv.lock --format json \
        | python3 osv-gate.py CRITICAL [--ignore images/<x>/scan-ignore]

Prints one line per finding at HIGH or above (package, version, CVSS, the lowest
version that fixes it, advisory ids), and exits 1 when a finding at the given
severity or above has a fixed version. Severity is the group's max CVSS score:
CRITICAL >= 9.0, HIGH >= 7.0, MEDIUM >= 4.0, LOW > 0. A finding with no CVSS score
counts as unknown and never fails the gate (it is still printed).

--ignore names a file of accepted advisory ids, one per line, `#` comments (the
.trivyignore format, so trivy reads the same file). A finding is IGNORED when any of
its ids or aliases is listed: it is still printed, marked, and never fails the gate.
An entry that matches nothing is reported as stale, so a fixed exception is noticed.
"""

import json
import sys
from pathlib import Path

THRESHOLDS = {"CRITICAL": 9.0, "HIGH": 7.0, "MEDIUM": 4.0, "LOW": 0.1}


def label(score):
    if score is None:
        return "UNKNOWN"
    for name, floor in THRESHOLDS.items():
        if score >= floor:
            return name
    return "NONE"


def fixed_versions(vuln, package):
    """Every `fixed` event for this package in one OSV record."""
    out = []
    for affected in vuln.get("affected", []):
        if affected.get("package", {}).get("name", "").lower() != package.lower():
            continue
        for rng in affected.get("ranges", []):
            out += [e["fixed"] for e in rng.get("events", []) if "fixed" in e]
    return out


def load_ignores(path):
    """The advisory ids in an ignore file; empty for None or a missing file."""
    if not path or not Path(path).is_file():
        return set()
    ids = set()
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.add(line.split()[0])
    return ids


def group_ids(group):
    return set(group.get("ids", [])) | set(group.get("aliases", []))


def version_key(v):
    parts = []
    for p in v.replace("-", ".").split("."):
        parts.append((0, int(p), "") if p.isdigit() else (1, 0, p))
    return parts


def main():
    args = sys.argv[1:]
    ignore_path = None
    if "--ignore" in args:
        i = args.index("--ignore")
        ignore_path = args[i + 1] if i + 1 < len(args) else None
        del args[i : i + 2]
    ignores = load_ignores(ignore_path)
    gate = args[0].upper() if args else "CRITICAL"
    if gate not in THRESHOLDS:
        sys.exit(
            f"osv-gate: unknown severity {gate!r}; use one of {', '.join(THRESHOLDS)}"
        )
    report = json.load(sys.stdin)
    findings = []
    for result in report.get("results", []):
        for pkg in result.get("packages", []):
            name = pkg["package"]["name"]
            version = pkg["package"]["version"]
            vulns = {v["id"]: v for v in pkg.get("vulnerabilities", [])}
            for group in pkg.get("groups", []):
                sev = group.get("max_severity")
                score = float(sev) if sev not in (None, "") else None
                fixes = []
                for vid in group.get("ids", []):
                    if vid in vulns:
                        fixes += fixed_versions(vulns[vid], name)
                fix = max(fixes, key=version_key) if fixes else None
                ids = group_ids(group)
                findings.append(
                    (score, name, version, fix, group.get("ids", []), ids & ignores)
                )

    findings.sort(key=lambda f: (-(f[0] or 0), f[1]))
    shown = [f for f in findings if f[0] is None or f[0] >= THRESHOLDS["HIGH"]]
    for score, name, version, fix, ids, hit in shown:
        tag = f"{label(score):8} {score if score is not None else '-':>4}"
        note = "  IGNORED (" + ignore_path + ")" if hit else ""
        print(f"  {tag}  {name} {version} -> {fix or 'no fix'}  {' '.join(ids)}{note}")
    failing = [
        f
        for f in findings
        if f[0] is not None and f[0] >= THRESHOLDS[gate] and f[3] and not f[5]
    ]
    ignored = [f for f in findings if f[5]]
    used = set().union(*(f[5] for f in findings)) if findings else set()
    for stale in sorted(ignores - used):
        print(
            f"  note: {stale} in {ignore_path} matches no finding; remove it if fixed"
        )
    counts = {}
    for f in findings:
        counts[label(f[0])] = counts.get(label(f[0]), 0) + 1
    summary = ", ".join(
        f"{counts[k]} {k}"
        for k in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"]
        if k in counts
    )
    print(
        f"  {len(findings)} advisory groups ({summary or 'none'}); "
        f"{len(failing)} fixable at {gate} or above"
        + (f", {len(ignored)} ignored" if ignored else "")
    )
    if not failing:
        print(
            f"  PASS: no fixable {gate} vulnerabilities"
            + (" outside the ignore list" if ignored else "")
        )
    sys.exit(1 if failing else 0)


if __name__ == "__main__":
    main()
