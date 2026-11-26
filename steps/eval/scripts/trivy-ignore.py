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
"""Write trivy's YAML ignore file from steps/eval/scan-ignore.

    python3 trivy-ignore.py <scan-ignore> <out.yaml>

scan-ignore is the one hand-maintained list of accepted advisories. Each entry is
`<id> exp:<YYYY-MM-DD> pkg:<purl> [pkg:<purl> ...]`, parsed by osv-gate.py
(steps/train/distill/images/), which the lock gate also runs. trivy's plain-text
ignore format keeps only the id and the expiry, so an id accepted for one package
would be accepted in every package. Its YAML format also takes `purls`. trivy then
suppresses a finding only when the finding's package URL matches one of them: same
type, name and, when given, version. Each entry becomes one item, so trivy enforces
the same package scope osv-gate does:

    vulnerabilities:
      - id: PYSEC-2026-4327
        purls: ["pkg:pypi/pillow@11.3.0"]
        expired_at: 2027-05-31
        statement: "pillow 11.3.0 (judged), ..."

The statement is the entry's section heading. The output file must end in .yaml:
that is how trivy tells the YAML format from the plain one. The make targets write
it under SCAN_CACHE at gate time; it is not committed.
"""

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "osv_gate", HERE.parents[1] / "train" / "distill" / "images" / "osv-gate.py"
)
osv_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(osv_gate)


def purl_string(purl):
    ptype, name, version = purl
    return f"pkg:{ptype}/{name}" + (f"@{version}" if version else "")


def sections(path):
    """Map each line number to the `# ---- <heading> ----` above it."""
    heading, out = "", {}
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if line.startswith("# ----"):
            heading = line.strip("# -").strip()
        out[n] = heading
    return out


def render(path):
    """The YAML text for one scan-ignore file. JSON strings are valid YAML scalars,
    so values are written with json.dumps and need no YAML library."""
    entries = osv_gate.load_ignore_entries(path)
    heading = sections(path)
    lines = Path(path).read_text().splitlines()
    index = {line.strip(): n for n, line in enumerate(lines, 1)}
    out = ["# Generated from " + str(path) + " by steps/eval/scripts/trivy-ignore.py.", "vulnerabilities:"]
    if not entries:
        out[-1] = "vulnerabilities: []"
    for e in entries:
        out.append(f"  - id: {json.dumps(e['id'])}")
        if e["purls"]:
            out.append("    purls: [" + ", ".join(json.dumps(purl_string(p)) for p in e["purls"]) + "]")
        if e["expires"]:
            out.append(f"    expired_at: {e['expires'].isoformat()}")
        statement = heading.get(index.get(e["line"], 0), "")
        if statement:
            out.append(f"    statement: {json.dumps(statement)}")
    return "\n".join(out) + "\n"


def main():
    if len(sys.argv) != 3 or not sys.argv[2].endswith((".yaml", ".yml")):
        sys.exit("usage: trivy-ignore.py <scan-ignore> <out.yaml>")
    try:
        text = render(sys.argv[1])
    except ValueError as e:
        sys.exit(f"trivy-ignore: {e}")
    Path(sys.argv[2]).write_text(text)


if __name__ == "__main__":
    main()
