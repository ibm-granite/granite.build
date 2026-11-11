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
"""Raise an image lock's vulnerable packages to their fixed versions (`make fix-lock-<x>`).

    python3 fix-locks.py <image dir> <SEVERITY> -- <osv-scanner command printing JSON>

For every fixable finding at SEVERITY or above that osv-scanner reports on
<image dir>/uv.lock, add `<pkg>>=<fixed version>` to [tool.uv] constraint-dependencies in
<image dir>/pyproject.toml and run `uv lock`, which moves only the packages a constraint
forces. Then rescan and repeat (an upgrade can pull in a new vulnerable version), at most
MAX_ROUNDS times. Never touches [project] dependencies: a package the project pins
EXACTLY (`vllm==0.11.0`) below its fix is reported as BLOCKED, because that pin is a
compatibility decision for a person to revisit. If the constraints together do not
resolve, each is tried alone and the ones uv cannot satisfy (a fix that a transitive
pin rules out) are reported as UNSATISFIABLE and left out. Advisories listed in
<image dir>/scan-ignore (accepted, documented exceptions) are skipped.

Exits 0 when nothing fixable at SEVERITY remains, 1 when something does (blocked,
unsatisfiable or still present), 2 on a usage or tool error. Review `git diff` and
rebuild/test the image afterwards: a constraint can cross a major version.
"""

import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

MAX_ROUNDS = 3
HERE = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("osv_gate", HERE / "osv-gate.py")
osv_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(osv_gate)


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def say(msg):
    print(f"[fix-lock] {msg}", flush=True)


def scan(cmd, ignores=frozenset()):
    """Run osv-scanner; return {package: (version, best fix, worst score, ids)} at any severity."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        sys.stderr.write(proc.stderr)
        sys.exit(f"[fix-lock] osv-scanner failed (exit {proc.returncode})")
    report = json.loads(proc.stdout or "{}")
    out = []
    for result in report.get("results", []):
        for pkg in result.get("packages", []):
            name, version = pkg["package"]["name"], pkg["package"]["version"]
            vulns = {v["id"]: v for v in pkg.get("vulnerabilities", [])}
            for group in pkg.get("groups", []):
                if osv_gate.group_ids(group) & ignores:
                    continue  # accepted in <image dir>/scan-ignore
                sev = group.get("max_severity")
                score = float(sev) if sev not in (None, "") else None
                fixes = [
                    f
                    for vid in group.get("ids", [])
                    if vid in vulns
                    for f in osv_gate.fixed_versions(vulns[vid], name)
                ]
                fix = max(fixes, key=osv_gate.version_key) if fixes else None
                out.append((name, version, score, fix, group.get("ids", [])))
    return out


def needed(findings, floor):
    """{normalized name: (name, current version, highest fix)} for fixable findings >= floor."""
    want = {}
    for name, version, score, fix, _ in findings:
        if score is None or score < floor or not fix:
            continue
        key = norm(name)
        prev = want.get(key)
        if prev is None or osv_gate.version_key(fix) > osv_gate.version_key(prev[2]):
            want[key] = (name, version, fix)
    return want


def exact_pins(pyproject):
    pins = {}
    for dep in (
        tomllib.loads(pyproject.read_text()).get("project", {}).get("dependencies", [])
    ):
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*==\s*([^\s;,]+)", dep)
        if m:
            pins[norm(m.group(1))] = m.group(3)
    return pins


def current_constraints(pyproject):
    return list(
        tomllib.loads(pyproject.read_text())
        .get("tool", {})
        .get("uv", {})
        .get("constraint-dependencies", [])
    )


def merge(existing, add):
    """Existing constraint strings plus `name>=fix` for each of add, raising, never lowering."""
    out, seen = [], set()
    for c in existing:
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*>=\s*([^\s;,]+)\s*$", c)
        key = norm(m.group(1)) if m else None
        if key in add:
            seen.add(key)
            name, _, fix = add[key]
            if osv_gate.version_key(fix) > osv_gate.version_key(m.group(2)):
                c = f"{m.group(1)}>={fix}"
        out.append(c)
    out += [f"{add[k][0]}>={add[k][2]}" for k in sorted(add) if k not in seen]
    return out


def write_constraints(pyproject, constraints):
    """Rewrite [tool.uv] constraint-dependencies in place, leaving the rest of the file alone."""
    text = pyproject.read_text()
    block = (
        "constraint-dependencies = [\n"
        + "".join(f'    "{c}",\n' for c in constraints)
        + "]\n"
    )
    m = re.search(r"^constraint-dependencies\s*=\s*\[", text, re.M)
    if m:
        end = text.index("]", m.end()) + 1
        if end < len(text) and text[end] == "\n":
            end += 1
        text = text[: m.start()] + block + text[end:]
    else:
        h = re.search(r"^\[tool\.uv\]\s*\n", text, re.M)
        if not h:
            text = text.rstrip("\n") + "\n\n[tool.uv]\n" + block
        else:
            # after the table's leading key = value lines, before any comment/blank/sub-table
            pos = h.end()
            for line in text[pos:].splitlines(keepends=True):
                if not re.match(r"^[A-Za-z0-9_\-]+\s*=", line):
                    break
                pos += len(line)
            text = text[:pos] + block + text[pos:]
    tomllib.loads(text)  # never leave an unparseable pyproject behind
    pyproject.write_text(text)


def conflict(log):
    """uv's one-sentence reason ("Because ... unsatisfiable."), else its last line."""
    flat = " ".join(line.strip(" │╰─▶") for line in log.splitlines())
    m = re.search(r"Because .*?unsatisfiable\.", flat)
    return m.group(0) if m else (log.splitlines()[-1] if log else "uv lock failed")


def uv_lock(image_dir):
    proc = subprocess.run(["uv", "lock"], cwd=image_dir, capture_output=True, text=True)
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def main():
    if len(sys.argv) < 5 or sys.argv[3] != "--":
        sys.exit(
            "usage: fix-locks.py <image dir> <SEVERITY> -- <osv-scanner json command...>"
        )
    image_dir, gate, cmd = Path(sys.argv[1]), sys.argv[2].upper(), sys.argv[4:]
    ignores = osv_gate.load_ignores(image_dir / "scan-ignore")
    if ignores:
        say(
            f"skipping {len(ignores)} advisory ids accepted in {image_dir / 'scan-ignore'}"
        )
    if gate not in osv_gate.THRESHOLDS:
        sys.exit(f"[fix-lock] unknown severity {gate!r}")
    floor = osv_gate.THRESHOLDS[gate]
    pyproject, lock = image_dir / "pyproject.toml", image_dir / "uv.lock"
    original = (pyproject.read_text(), lock.read_text())
    pins = exact_pins(pyproject)
    blocked, unsat, applied = {}, {}, {}

    for rnd in range(1, MAX_ROUNDS + 1):
        want = needed(scan(cmd, ignores), floor)
        for k in list(want):
            if k in pins and osv_gate.version_key(pins[k]) < osv_gate.version_key(
                want[k][2]
            ):
                blocked[k] = (want.pop(k), pins[k])
            elif k in unsat or k in blocked:
                want.pop(k)
        if not want:
            break
        say(
            f"round {rnd}: raising "
            + ", ".join(f"{n} {v} -> >={f}" for n, v, f in want.values())
        )
        base = current_constraints(pyproject)
        write_constraints(pyproject, merge(base, want))
        ok, log = uv_lock(image_dir)
        if not ok:
            say("uv lock failed with all constraints together; trying each alone")
            accepted = base
            for k, item in want.items():
                trial = merge(accepted, {k: item})
                write_constraints(pyproject, trial)
                ok1, log1 = uv_lock(image_dir)
                if ok1:
                    accepted, applied[k] = trial, item
                else:
                    unsat[k] = (item, conflict(log1))
            write_constraints(pyproject, accepted)
            ok, log = uv_lock(
                image_dir
            )  # leave the lock matching the constraints that stayed
            if not ok:
                pyproject.write_text(original[0])
                lock.write_text(original[1])
                sys.exit(
                    f"[fix-lock] uv lock failed even with the accepted constraints; restored originals\n{log}"
                )
        else:
            applied.update(want)
    else:
        say(f"stopped after {MAX_ROUNDS} rounds")

    if not current_constraints(pyproject):
        # nothing stuck: drop an empty constraint list we may have written
        if "constraint-dependencies" not in original[0]:
            pyproject.write_text(original[0])
            lock.write_text(original[1])

    remaining = needed(scan(cmd, ignores), floor)
    for k, item in applied.items():
        say(
            f"FIXED         {item[0]} {item[1]} -> >={item[2]}"
            + ("  (still flagged)" if k in remaining else "")
        )
    for k, ((name, ver, fix), pin) in blocked.items():
        say(
            f"BLOCKED       {name} {ver} needs >={fix}, but [project] pins {name}=={pin}; "
            "raising it is a compatibility change, not a lock fix"
        )
    for k, ((name, ver, fix), why) in unsat.items():
        say(f"UNSATISFIABLE {name} {ver} -> >={fix}: {why}")
    left = [v for k, v in remaining.items() if k not in blocked and k not in unsat]
    for name, ver, fix in left:
        say(f"REMAINING     {name} {ver} -> >={fix}")
    if remaining:
        say(f"{len(remaining)} package(s) still have a fixable {gate} finding")
        sys.exit(1)
    say(f"no fixable {gate} findings left in {lock}")


if __name__ == "__main__":
    main()
