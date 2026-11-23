# SPDX-License-Identifier: Apache-2.0

"""The shared SkyPilot-on-SLURM environment template, for the recipe site tests.

recipes/environments/skypilot-slurm.template.yaml is rendered from a private
site file by scripts/setup-local-space.sh. Each recipe that uses it ships a
site.example.yaml with fake values and a test that renders the template from
that example and scans the recipe for site-specific tokens. These helpers are
what those tests share: the paths, the renderer run as setup-local-space.sh
runs it, the template's keys and the leak guard.
"""

import os
import pathlib
import re
import subprocess
import sys
from typing import List, Set

import jinja2
from jinja2 import meta

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SLURM_TEMPLATE = REPO_ROOT / "recipes" / "environments" / "skypilot-slurm.template.yaml"
RENDER_ENV_TEMPLATE = REPO_ROOT / "scripts" / "render_env_template.py"

# Tokens that identify one site's infrastructure. Case-insensitive. The
# project's public image org, quay.io/granitebuild, is not one: the step
# images live there.
LEAK = re.compile(
    r"bluevela|rmf\.ibm|ibm\.com|/proj/|us\.icr\.io|cil15|grp_granite"
    r"|(?<!quay\.io/)granitebuild"
    r"|mlx5_|login[0-9]+\.bluevela"
    r"|data-eng|llmb-read-write|g4os|gpu-mid|bv_ssh|1password"
    r"|\bp[0-9]+-r[0-9]+-n[0-9]+\b",
    re.IGNORECASE,
)

# Keys a site file may leave out: EVAL_QOS defaults to "", and EVAL_PARTITION
# is read only when EVAL_QOS is set. A recipe with no eval lane omits both.
OPTIONAL_KEYS = frozenset({"EVAL_QOS", "EVAL_PARTITION"})

_VALUE = re.compile(r"\$\$\{\s*([A-Za-z_][A-Za-z0-9_]*)")
_LOOP_VAR = re.compile(r"<%-?\s*for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\b")


def render(site_path, out_path, template=SLURM_TEMPLATE) -> subprocess.CompletedProcess:
    """scripts/render_env_template.py TEMPLATE SITE OUT, as a subprocess."""
    return subprocess.run(
        [
            sys.executable,
            str(RENDER_ENV_TEMPLATE),
            str(template),
            str(site_path),
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "GB_ENVIRONMENT": "STANDALONE"},
    )


def value_keys(template=SLURM_TEMPLATE) -> Set[str]:
    """The keys the template substitutes as $${KEY}, loop variables excluded."""
    text = pathlib.Path(template).read_text(encoding="utf-8")
    return set(_VALUE.findall(text)) - set(_LOOP_VAR.findall(text))


def undeclared_keys(template=SLURM_TEMPLATE) -> Set[str]:
    """Every key the template reads, in a value or a <% %> block."""
    jenv = jinja2.Environment(
        variable_start_string="$${",
        variable_end_string="}",
        block_start_string="<%",
        block_end_string="%>",
    )
    text = pathlib.Path(template).read_text(encoding="utf-8")
    return meta.find_undeclared_variables(jenv.parse(text))


def leak_hits(files) -> List[str]:
    """`path:line: token` for every LEAK match in the text files given."""
    hits = []
    for p in files:
        try:
            text = pathlib.Path(p).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            m = LEAK.search(line)
            if m:
                hits.append(
                    f"{pathlib.Path(p).relative_to(REPO_ROOT)}:{n}: {m.group(0)}"
                )
    return hits
