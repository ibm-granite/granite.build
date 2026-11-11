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

    osv-scanner scan source --lockfile uv.lock --format json | python3 osv-gate.py CRITICAL

Prints one line per finding at HIGH or above (package, version, CVSS, the lowest
version that fixes it, advisory ids), and exits 1 when a finding at the given
severity or above has a fixed version. Severity is the group's max CVSS score:
CRITICAL >= 9.0, HIGH >= 7.0, MEDIUM >= 4.0, LOW > 0. A finding with no CVSS score
counts as unknown and never fails the gate (it is still printed).
"""

import json
import sys

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


def version_key(v):
    parts = []
    for p in v.replace("-", ".").split("."):
        parts.append((0, int(p), "") if p.isdigit() else (1, 0, p))
    return parts


def main():
    gate = sys.argv[1].upper() if len(sys.argv) > 1 else "CRITICAL"
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
                findings.append((score, name, version, fix, group.get("ids", [])))

    findings.sort(key=lambda f: (-(f[0] or 0), f[1]))
    shown = [f for f in findings if f[0] is None or f[0] >= THRESHOLDS["HIGH"]]
    for score, name, version, fix, ids in shown:
        tag = f"{label(score):8} {score if score is not None else '-':>4}"
        print(f"  {tag}  {name} {version} -> {fix or 'no fix'}  {' '.join(ids)}")
    failing = [
        f for f in findings if f[0] is not None and f[0] >= THRESHOLDS[gate] and f[3]
    ]
    counts = {}
    for f in findings:
        counts[label(f[0])] = counts.get(label(f[0]), 0) + 1
    summary = ", ".join(
        f"{counts[k]} {k}"
        for k in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"]
        if k in counts
    )
    print(
        f"  {len(findings)} advisory groups ({summary or 'none'}); {len(failing)} fixable at {gate} or above"
    )
    sys.exit(1 if failing else 0)


if __name__ == "__main__":
    main()
