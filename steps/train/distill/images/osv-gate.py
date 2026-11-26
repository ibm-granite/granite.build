# SPDX-License-Identifier: Apache-2.0

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
FAILS CLOSED when it has a fixed version (an unscored advisory is not a mild one
-- it is an unmeasured one); it is printed as UNKNOWN and never fails when no fix
exists or it is ignored.

A fix is the group's highest `fixed` event. Records that end their range with
`last_affected` instead count as fixed when a final release after it exists (PyPI's
release list, read over the network); when the releases cannot be listed, or the
package is not on PyPI, such a group fails closed as fixable.
--summarise-stale prints one count of the entries that match nothing instead of one
line each, for an ignore file shared by several scans (steps/eval's).

--ignore names a file of accepted advisory ids, one per line, `#` comments (the
.trivyignore format, so trivy reads the same file). A finding is IGNORED when any of
its ids or aliases is listed: it is still printed, marked, and never fails the gate.
An entry that matches nothing is reported as stale, so a fixed exception is noticed.

An entry may carry two optional fields after the id, as whitespace-separated tokens:

    PYSEC-2026-4327 exp:2027-05-31 pkg:pypi/pillow@11.3.0 pkg:pypi/pillow@12.3.0

- `exp:YYYY-MM-DD`, trivy's expiry: from that date (UTC) the entry no longer applies,
  and it is reported as expired.
- `pkg:<type>/<name>[@<version>]`, package URLs (purls): the entry then applies only
  to a finding in one of those packages, at that version (any version when it is
  omitted). PyPI names compare normalised (PEP 503). Several may be given.
A bare id, as in distill's and hf-export's files, applies to every package and never
expires, as before.

WARNING: only a gate that honours both fields may use them. steps/eval gives trivy a
YAML ignore file generated from its scan-ignore (scripts/trivy-ignore.py), so trivy
applies the same scope. distill's and hf-export's gates give trivy the scan-ignore
file itself, in trivy's plain format, which keeps `exp:` but ignores `pkg:`. A `pkg:`
scope there would narrow this gate and leave trivy's unscoped.
"""

import datetime
import os
import json
import re
import sys
import urllib.request
from pathlib import Path

# osv-scanner's ecosystem names -> purl types.
ECOSYSTEM_PURL_TYPE = {
    "PyPI": "pypi",
    "npm": "npm",
    "Go": "golang",
    "crates.io": "cargo",
    "Maven": "maven",
    "Debian": "deb",
}

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


def last_affected_versions(vuln, package):
    """Every `last_affected` event for this package in one OSV record. Some records
    encode the affected range as introduced..last_affected instead of
    introduced..fixed; a release after last_affected then fixes it."""
    out = []
    for affected in vuln.get("affected", []):
        if affected.get("package", {}).get("name", "").lower() != package.lower():
            continue
        for rng in affected.get("ranges", []):
            out += [e["last_affected"] for e in rng.get("events", []) if "last_affected" in e]
    return out


FINAL_RELEASE = re.compile(r"\d+(\.\d+)*(\.post\d+)?")


def _numbers(v):
    """The leading numeric release of a version (5.0.0 for 5.0.0-rc0) and whether
    anything (a pre-release tag) follows it."""
    m = re.match(r"\d+(\.\d+)*", v)
    nums = tuple(int(p) for p in m.group(0).split(".")) if m else ()
    rest = v[m.end() :] if m else v
    return nums, bool(rest) and not re.fullmatch(r"\.post\d+", rest)


def is_after(release, last_affected):
    """Is the final release `release` later than `last_affected`? A last_affected
    pre-release (5.0.0-rc0) is followed by its own final release (5.0.0)."""
    r, _ = _numbers(release)
    la, la_pre = _numbers(last_affected)
    width = max(len(r), len(la))
    r, la = r + (0,) * (width - len(r)), la + (0,) * (width - len(la))
    return r >= la if la_pre else r > la


_RELEASES = {}


def releases(ecosystem, name):
    """The package's final, unyanked releases, or None if they cannot be listed.

    PyPI only, from its JSON API (osv-scanner needs the network anyway). The
    environment variable OSV_GATE_RELEASES may name a JSON file {name: [versions]}
    used instead of PyPI (the unit tests)."""
    key = (ecosystem, normalize_name("pypi", name) if ecosystem == "PyPI" else name.lower())
    if key in _RELEASES:
        return _RELEASES[key]
    found = None
    override = os.environ.get("OSV_GATE_RELEASES")
    if override:
        found = json.loads(Path(override).read_text()).get(key[1])
    elif ecosystem == "PyPI":
        try:
            url = f"https://pypi.org/pypi/{key[1]}/json"
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.load(r)
            found = [
                v
                for v, files in data.get("releases", {}).items()
                if files and not all(f.get("yanked") for f in files)
            ]
        except (OSError, ValueError):
            found = None
    if found is not None:
        found = [v for v in found if FINAL_RELEASE.fullmatch(v)]
    _RELEASES[key] = found
    return found


def fix_after_last_affected(ecosystem, name, last):
    """For a group whose records give only last_affected: the first final release
    after it, as `>LAST (first: X)`; or, when the releases cannot be listed, fail
    closed as `>LAST (releases unknown)`. None when no release is after it."""
    newest = max(last, key=version_key)
    known = releases(ecosystem, name)
    if known is None:
        return f">{newest} (releases unknown)"
    after = sorted((v for v in known if is_after(v, newest)), key=version_key)
    return f">{newest} (first: {after[0]})" if after else None


def load_ignores(path):
    """The advisory ids in an ignore file; empty for None or a missing file.

    Every id, whatever its scope or expiry (fix-locks.py's view); the gate itself uses
    load_ignore_entries, which keeps both."""
    return {e["id"] for e in load_ignore_entries(path)}


def normalize_name(purl_type, name):
    """A package name as purls compare it: PEP 503 for PyPI, lowercase otherwise."""
    if purl_type == "pypi":
        return re.sub(r"[-_.]+", "-", name).lower()
    return name.lower()


def split_purl(token):
    """`pkg:<type>/<name>[@<version>]` -> (type, name as written, version or None).

    A namespace (`pkg:deb/debian/name`) is kept in the name. Qualifiers and subpaths
    are not supported: they are rejected rather than ignored."""
    body = token[len("pkg:") :]
    if "?" in body or "#" in body or "/" not in body:
        raise ValueError(f"unsupported purl {token!r}: use pkg:<type>/<name>[@<version>]")
    ptype, rest = body.split("/", 1)
    name, _, version = rest.rpartition("@") if "@" in rest else (rest, "", "")
    if not ptype or not name:
        raise ValueError(f"unsupported purl {token!r}: use pkg:<type>/<name>[@<version>]")
    return ptype.lower(), name, version or None


def parse_purl(token):
    """`pkg:<type>/<name>[@<version>]` -> (type, normalised name, version or None)."""
    ptype, name, version = split_purl(token)
    return ptype, normalize_name(ptype, name), version


def parse_ignore_line(line):
    """One ignore-file line -> {id, expires, purls, line}, or None for a comment/blank.

    Raises ValueError on a malformed exp: or pkg: field, so a typo fails the gate
    instead of silently widening or dropping an acceptance."""
    text = line.split("#", 1)[0].strip()
    if not text:
        return None
    fields = text.split()
    entry = {"id": fields[0], "expires": None, "purls": [], "raw_purls": [], "line": line.strip()}
    for field in fields[1:]:
        if field.startswith("exp:"):
            entry["expires"] = datetime.date.fromisoformat(field[len("exp:") :])
        elif field.startswith("pkg:"):
            entry["purls"].append(parse_purl(field))
            entry["raw_purls"].append(split_purl(field))
        else:
            raise ValueError(f"unknown field {field!r} in ignore entry {line.strip()!r}")
    return entry


def load_ignore_entries(path):
    """The entries of an ignore file, in order; empty for None or a missing file."""
    if not path or not Path(path).is_file():
        return []
    entries = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        try:
            entry = parse_ignore_line(line)
        except ValueError as e:
            raise ValueError(f"{path}:{n}: {e}") from None
        if entry:
            entries.append(entry)
    return entries


def is_expired(entry, today):
    """trivy's rule: an entry expires at 00:00 UTC on its exp: date."""
    return entry["expires"] is not None and today >= entry["expires"]


def scope_matches(entry, ecosystem, name, version):
    """Does an entry's package scope cover this package? No scope covers every one."""
    if not entry["purls"]:
        return True
    ptype = ECOSYSTEM_PURL_TYPE.get(ecosystem, (ecosystem or "").lower())
    target = normalize_name(ptype, name)
    return any(
        t == ptype and n == target and (v is None or v == version)
        for t, n, v in entry["purls"]
    )


def utc_today():
    return datetime.datetime.now(datetime.timezone.utc).date()


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
    summarise_stale = "--summarise-stale" in args
    if summarise_stale:
        args.remove("--summarise-stale")
    try:
        entries = load_ignore_entries(ignore_path)
    except ValueError as e:
        sys.exit(f"osv-gate: {e}")
    today = utc_today()
    live = [e for e in entries if not is_expired(e, today)]
    ignores = {e["id"] for e in live}
    gate = args[0].upper() if args else "CRITICAL"
    if gate not in THRESHOLDS:
        sys.exit(
            f"osv-gate: unknown severity {gate!r}; use one of {', '.join(THRESHOLDS)}"
        )
    report = json.load(sys.stdin)
    findings = []
    out_of_scope = []
    for result in report.get("results", []):
        for pkg in result.get("packages", []):
            name = pkg["package"]["name"]
            version = pkg["package"]["version"]
            ecosystem = pkg["package"].get("ecosystem", "")
            vulns = {v["id"]: v for v in pkg.get("vulnerabilities", [])}
            for group in pkg.get("groups", []):
                sev = group.get("max_severity")
                score = float(sev) if sev not in (None, "") else None
                fixes, last = [], []
                for vid in group.get("ids", []):
                    if vid in vulns:
                        fixes += fixed_versions(vulns[vid], name)
                        last += last_affected_versions(vulns[vid], name)
                fix = max(fixes, key=version_key) if fixes else None
                if fix is None and last:
                    fix = fix_after_last_affected(ecosystem, name, last)
                ids = group_ids(group)
                listed = [e for e in live if e["id"] in ids]
                hit = {
                    e["id"]
                    for e in listed
                    if scope_matches(e, ecosystem, name, version)
                }
                out_of_scope += [
                    (e["id"], name, version) for e in listed if e["id"] not in hit
                ]
                findings.append(
                    (score, name, version, fix, group.get("ids", []), hit)
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
        if (f[0] is None or f[0] >= THRESHOLDS[gate]) and f[3] and not f[5]
    ]
    ignored = [f for f in findings if f[5]]
    used = set().union(*(f[5] for f in findings)) if findings else set()
    stale = sorted(ignores - used - {o[0] for o in out_of_scope})
    if summarise_stale and stale:
        print(
            f"  note: {len(stale)} entries in {ignore_path} match no finding in this scan"
            " (they may apply to another lock or image)"
        )
    else:
        for vid in stale:
            print(
                f"  note: {vid} in {ignore_path} matches no finding; remove it if fixed"
            )
    for vid, name, version in sorted(set(out_of_scope)):
        print(
            f"  note: {vid} in {ignore_path} is accepted for other packages only, "
            f"not {name} {version}"
        )
    for e in entries:
        if is_expired(e, today):
            print(f"  note: {e['id']} in {ignore_path} expired on {e['expires']}")
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
