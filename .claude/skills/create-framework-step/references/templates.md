# File skeletons

Replace `<category>`, `<name>` (may contain hyphens), `<path>` (= `<category>/<name>`),
`<cfg>` (`<name>` with hyphens turned into underscores, used in every Jinja
expression), `<Name>`, `<env>`, `<cluster>`. `<in>` and `<out>` stand for one input
or output name. Repeat the line holding it, **or the whole block it heads** (that line
plus the more-indented lines under it, e.g. `<in>:` with its `type:`/`accept:` or
`uri:`), once per name the user gave. If a name is a Python dict method (`items`,
`keys`, `get`, ...), write its binding as `bindings["<in>"]` rather than
`bindings.<in>` (see SKILL.md, "Normalizing the answers"). Drop it when they said `none`, and then drop
an emptied `required:` too (`inputs: {allow_unknown: true}`), because a bare
`required:` is null, not an empty map. With no inputs, also put a comment of at least
four words directly above `inputs:` saying why, e.g. `# This step reads no inputs;
recipes may bind any to order targets.`. test_step_inputs_declared.py requires it, and
rejects one that starts with FIXME. Working examples to copy from:
`steps/distill/vllm-server/skypilot` (nested under a category, like a new step),
`steps/data/dpk/skypilot` (public image, python `src/`, five clusters) and
`steps/eval/eval-example/skypilot` (custom image). Add the repo's Apache-2.0 header to every `.py`.

## Makefile

```make
# <name> step for the <env> environment. FIXME: one line on what it does.
# Public image: no Dockerfile here, so common.mk treats it as a public-image step.
# Custom image: the Dockerfile beside this file makes it a custom-image step.

STEP_NAME := <name>

# Custom image only. FIXME: the registry and namespace the image is pushed to.
# `?=` so `make ... REGISTRY=quay.io/<org>` can still override it.
REGISTRY ?= quay.io/FIXME

# Only for a step that must never be published, such as an exemplar or template
# (see steps/eval/eval-example). A new step is publishable by default, so normally
# leave this out. `override` stops a command-line STEP_PUBLISHABLE=true re-enabling it.
# override STEP_PUBLISHABLE := false

# Only if the slurm cluster was selected, i.e. this file defines `test-setup` below.
# It tells common.mk to drop its own no-op `test-setup`. Don't set it for kube alone
# (kube's target is `test-setup-kube`). It wouldn't fail: common.mk lists test-setup
# in .PHONY, so `make test-setup` would print "Nothing to be done" and exit 0, giving
# no sign that setup never ran. Unset, common.mk's no-op says so explicitly.
HAS_TEST_SETUP := true

include ../../../common.mk       # steps/<category>/<name>/<env>: three levels up

# One target per local cluster selected:
# slurm ->
test-setup:
	$(MAKE) -C $(REPO_ROOT) slurm-setup
# kube ->
.PHONY: test-setup-kube
test-setup-kube:
	$(MAKE) -C $(REPO_ROOT) kube-setup

# Only if src/ holds Python: keep bytecode from `make test` out of the shipped tree.
space publish-step: drop-pycache
.PHONY: drop-pycache
drop-pycache:
	@find $(SRC_DIR) -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
```

`REGISTRY` and `STEP_PUBLISHABLE` must come before the `include`, as shown.

## step-template.yaml (SkyPilot)

Shown with the placeholders from SKILL.md's table. Swap in the user's real names and
values where the prompt gave them, and drop that `FIXME`.

```yaml
name: <name>
version: 1.0.0
type: exec
config:
  # FIXME: the step's own settings go here, as the user directs. Default every key
  # (launcher rendering is strict) and pass each to src/run.sh as argv through q().
  <cfg>_config: {}
  compute_config:
    num_gpus_per_node: 0
    total_memory_per_node: "4Gi"
inputs:
  allow_unknown: true            # lets recipes bind extra inputs to order targets
  required:
    <in>:                        # FIXME: type (default dataset), required/optional
      type: dataset
      accept: [uri, binding]
outputs:
  allow_unknown: true
  required:
    <out>:                       # FIXME: type (default dataset)
      type: dataset
environment_configs:
  Skypilot:
    default_launcher: <name>
    launchers:
      <name>:
        type: skypilot
        monitors: [skypilot_monitor]
        config:
          image_id: ""           # FIXME: set a public image if the code needs one
          resources: {}
          file_mounts:
            src: src             # ships src/ to ./src before setup
          setup: |
            set -euo pipefail
            bash ./src/setup.sh
          run: |
            set -euo pipefail
            {%- macro q(v) -%}{{ v | string | replace("'", "'\"'\"'") }}{%- endmacro %}
            bash ./src/run.sh \
              --input-<in> '{{ q(bindings.<in>.binding.path) }}' \
              --output-<out> ./<out> \
              # (end of run.sh arguments: keep this line last)
    monitors:
      skypilot_monitor:
        ref: space://monitors/skypilot
```

If `setup` gains arguments rendered from config, define `q()` there too: macros are
block-scoped.

**Custom image:** the scripts are baked into the image, so change the launcher
`config` to this (no `file_mounts`; `${IMAGE_REF}` is filled in by `make space` and
`make publish-step`):

```yaml
        config:
          image_id: "docker:${IMAGE_REF}"
          resources: {}
          setup: |
            set -euo pipefail
            bash /opt/<name>/setup.sh
          run: |
            set -euo pipefail
            {%- macro q(v) -%}{{ v | string | replace("'", "'\"'\"'") }}{%- endmacro %}
            bash /opt/<name>/run.sh \
              --input-<in> '{{ q(bindings.<in>.binding.path) }}' \
              --output-<out> ./<out> \
              # (end of run.sh arguments: keep this line last)
```

## Dockerfile (custom image only)

Builds as is (checked: Debian 13 slim, runs as uid 1001, passwordless sudo works).

```dockerfile
# <name> image (custom-image step). FIXME: add the step's dependencies.
#
# SkyPilot runs apt-get inside the container to bootstrap itself, so the base MUST be
# Debian/Ubuntu (a UBI/RPM base fails provisioning), and the container user needs
# passwordless sudo. Debian slim is the smallest base that meets both.
ARG DEBIAN_VERSION=13
FROM public.ecr.aws/docker/library/debian:${DEBIAN_VERSION}-slim

# sudo is what SkyPilot needs; FIXME: add the step's own packages to this line.
RUN apt-get update \
 && apt-get install -y --no-install-recommends sudo \
 && rm -rf /var/lib/apt/lists/*

# Non-root runtime user. SkyPilot requires passwordless sudo for its in-container
# setup (docs/environments/skypilot-lsf.md); that is the only privilege granted.
RUN useradd -m -u 1001 appuser \
 && echo 'appuser ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/appuser \
 && chmod 0440 /etc/sudoers.d/appuser

# The step's scripts, baked in. step.yaml's setup:/run: call them at /opt/<name>/.
WORKDIR /opt/<name>
COPY --chown=appuser:appuser src/ /opt/<name>/

USER 1001
```

## .gbignore

Write it as is: no placeholders. It is required: `make space` and `make publish-step`
copy it beside the rendered `step.yaml` and refuse a step that has none. Every existing
step's `.gbignore` has these same two patterns (a unit test checks that).

```gitignore
# Files gbserver ships verbatim instead of Jinja-rendering them, one glob per line.
# Additive only: there is no `!` negation. `make space` (and so `make test`) and
# `make publish-step` both copy this file, so the step renders the same way when
# tested and when published. Remove a line only if those files must be rendered.
#
# Docs: USAGE.md and README.md may show {{ ... }} examples.
**/*.md
# Scripts: src/ runs as is; its values arrive as argv from step-template.yaml.
src/**
```

## src/setup.sh and src/run.sh (comment-only stubs)

Generate exactly these two files in `src/` and nothing else there. They hold comments
only, no implementation: the user writes the code. Fill in the argument list from the
template's actual `run:` call.

```bash
#!/usr/bin/env bash
# <name> setup script. FIXME: implement.
#
# Purpose: prepare the node before run.sh: install the step's dependencies (e.g. a
# venv or pip packages). Leave it empty if the image already has everything.
#
# How step.yaml uses it: the Skypilot launcher's `setup:` block runs
#   bash ./src/setup.sh
# once per cluster, in the step's working directory. file_mounts: {src: src} ships
# this directory to ./src before setup runs. (Custom image: it is baked in instead,
# and setup: runs bash /opt/<name>/setup.sh.) It takes no arguments today; add any as
# argv rendered from config.<cfg>_config.* through q(), defining q() in setup too.
#
# Exit non-zero to fail the step before run.sh starts.
```

```bash
#!/usr/bin/env bash
# <name> run script. FIXME: implement.
#
# Purpose: the step's actual work: read the inputs, write the outputs.
#
# How step.yaml uses it: the Skypilot launcher's `run:` block calls
#   bash ./src/run.sh --input-<in> <path> ... --output-<out> ./<out> ...
# (custom image: bash /opt/<name>/run.sh ...) from the step's working directory:
#   --input-<in>   one per input: {{ bindings.<in>.binding.path }} (escaped with
#                  q()), where that input was staged
#   --output-<out> one per output: where to write it, relative to the working dir
# FIXME list here the actual arguments generated, one line each.
# Settings added to config.<cfg>_config arrive as further arguments, the same way.
#
# Contract with the server:
# - Make each output path absolute before reporting it (the server reads it off-node).
# - Register each output by printing one line per output, at the end:
#     GB_ARTIFACT_ID:<out> GB_ARTIFACT_PATH:<absolute output path>
#   The ID must equal the output's name in step.yaml and the build.yaml.
# - Optionally record results a build test can assert:
#     GB_STEP_METADATA_KEY:<key> GB_STEP_METADATA_VALUE:<value>
# - Exit non-zero on failure; print no artifact marker then.
```

**Docker or Bash env:** replace the `Skypilot` block with `Docker` (launcher
`type: docker`, monitor `ref: space://monitors/docker`) or `Bash` (launcher
`type: nohup`, monitor `ref: space://monitors/bash`). Copy the launcher `config` shape
from `src/gbserver/builtins/steps/<docker|bash>/command/step.yaml`.

## test/test_<env>_<category>_<name>_render.py (render test)

The one test that passes as soon as the skeleton exists: it renders the launcher the way
gbserver does and checks the plumbing, not the step's behaviour. Cluster-agnostic, so it
sits at `test/`'s root and runs on every `make test` (Mode 1 only; not published, and not
in PR CI, which runs only `test/`). `steps/data/dpk/skypilot/test/test_dpk_step_render.py`
shows how far it can grow. Generate it with one `_BINDINGS` entry and one
`--input-<in>`/`--output-<out>` pair per name, and drop the `@pytest.mark.xfail` line once
`run.sh` prints its artifact markers.

```python
"""Render test for the <name> step's step-template.yaml.

Renders the launcher's setup/run blocks the way gbserver does (strict Jinja, so a
missing config key or binding fails here rather than on a cluster), then runs them
with stub scripts to check the argv each script receives. Mode 1 only.
FIXME: add a test per config setting as it's added (its default and an override).
"""

import pathlib
import shutil
import subprocess

import pytest
import yaml

from gbcommon.utils.template import fill_template

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_TEMPLATE = yaml.safe_load((_STEP_DIR / "step-template.yaml").read_text())
_LAUNCHER = _TEMPLATE["environment_configs"]["Skypilot"]["launchers"]["<name>"]["config"]
# One entry per declared input; a quote in a path checks the q() escaping.
_BINDINGS = {
    "<in>": {"binding": {"path": "/staged/<in> it's"}},
}
_CONTEXT = {"config": _TEMPLATE["config"], "bindings": _BINDINGS}


def _render(block: str) -> str:
    """Render one launcher block with gbserver's strict renderer.

    Args:
        block: ``"setup"`` or ``"run"``.

    Returns:
        The rendered shell text.
    """
    return fill_template(_LAUNCHER[block], _CONTEXT, strict=True)


def _argv(block: str, tmp_path: pathlib.Path) -> list[str]:
    """Run a rendered block with stub scripts that print their argv.

    Args:
        block: ``"setup"`` or ``"run"``.
        tmp_path: scratch dir to run in, so the step tree is never touched.

    Returns:
        The arguments the block passed to its ``src/`` script, one per item.
    """
    if shutil.which("bash") is None:  # pragma: no cover - bash is present in CI
        pytest.skip("bash not available")
    stub = '#!/usr/bin/env bash\nfor a in "$@"; do echo "ARG:$a"; done\n'
    for name in ("setup.sh", "run.sh"):
        (tmp_path / "src").mkdir(exist_ok=True)
        (tmp_path / "src" / name).write_text(stub)
    proc = subprocess.run(
        ["bash", "-c", _render(block)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return [ln[4:] for ln in proc.stdout.splitlines() if ln.startswith("ARG:")]


@pytest.mark.parametrize("block", ["setup", "run"])
def test_block_renders_and_is_valid_bash(block):
    """Each block renders strictly and bash can parse the result."""
    rendered = _render(block)
    assert subprocess.run(["bash", "-n"], input=rendered, text=True).returncode == 0


def test_run_passes_each_input_path_verbatim(tmp_path):
    """Every input reaches run.sh as one argument, quotes intact."""
    argv = _argv("run", tmp_path)
    for name, binding in _BINDINGS.items():
        flag = f"--input-{name}"
        assert argv[argv.index(flag) + 1] == binding["binding"]["path"]


def test_run_passes_each_output_dir(tmp_path):
    """Every output gets an --output-<name> argument."""
    argv = _argv("run", tmp_path)
    for name in _TEMPLATE["outputs"].get("required", {}):
        assert f"--output-{name}" in argv


@pytest.mark.xfail(reason="FIXME: src/run.sh doesn't print GB_ARTIFACT_ID yet", strict=True)
def test_run_sh_registers_each_output():
    """src/run.sh prints a GB_ARTIFACT_ID marker for every declared output.

    Only code counts: the stub's comments describe the marker, so comment lines are
    skipped, or the comment alone would satisfy the check.
    """
    code = "\n".join(
        line
        for line in (_STEP_DIR / "src" / "run.sh").read_text().splitlines()
        if not line.lstrip().startswith("#")
    )
    for name in _TEMPLATE["outputs"].get("required", {}):
        assert f"GB_ARTIFACT_ID:{name}" in code or "GB_ARTIFACT_ID:$" in code
```

`strict=True` on the xfail makes it fail loudly once `run.sh` does print the markers, so
the reminder to remove it can't be missed. For a custom image, the launcher calls
`/opt/<name>/*.sh`; point the stubs there instead (write them to `tmp_path` and replace
`/opt/<name>/` with `./src/` in the rendered text before running).

## test/<cluster>/test_<env>_<cluster>_<category>_<name>.py

```python
"""<name> on <cluster>: <what reaching SUCCESS proves>.

<Infra needed and how it is gated.> Fixtures live in the test-data/ dir mirroring
this file (get_test_data_dir_for), so the pairing holds in both test modes.
"""

from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only
from libgbtest.hfpush import HfPushCleanupMixin    # only when the fixture has hf:// outputs
# + the cluster's gate import from clusters.md

pytestmark = pytest.mark.skypilot_integration      # IBM space/cluster: pytest.mark.ibm


@extended_testing_only
@pytest.mark.skipif(not <gate>(), reason="<infra> not reachable (run: make test-setup)")
# kube instead: @KUBE_XDIST_GROUP above, and
#   @pytest.mark.skipif(not kind_cluster_reachable(), reason=kind_skip_reason())
# BlueVela instead: @pytest.mark.xdist_group(name="buildtest_bv") + the RUNNING_IN_CICD skip
# HfPushCleanupMixin first: it names each hf:// output per run, checks it landed on the
# Hub, and deletes it afterwards. Drop it when the step has no outputs.
class Test<Env><Cluster><Name>(HfPushCleanupMixin, AbstractYamlBuildRunnerTest):
    """<name> step, end to end on <cluster>."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)
```

Add the timeout override only when `timeout_minutes` > 20:

```python
    @pytest.mark.timeout(2100)  # above buildtest.yaml's timeout_minutes (30)
    def test_runner(self):
        """Run the happy-path build with room for the buildtest timeout."""
        super().test_runner()
```

## test-data/<cluster>/build.yaml

```yaml
granite.build:
  name: <env>-<cluster>-<name>
  targets:
    <target>:
      environment_uri: <from clusters.md>
      inputs:
        <in>:
          uri: FIXME          # FIXME: small public dataset, e.g. hf:///datasets/<org>/<repo> (no HF_TOKEN needed)
      outputs:
        <out>:
          # FIXME: dataset name. The __GBTEST_HF_PUSH_*__ tokens are filled in per run by
          # HfPushCleanupMixin (org from GBTEST_HF_PUSH_ORG, default ibm-research).
          uri: hf://huggingface.co/datasets/__GBTEST_HF_PUSH_ORG__/<name>-<out>___GBTEST_HF_PUSH_RUN__
      steps:
        - step_uri: space://steps/<path>
          config:
            poll_interval_seconds: 5
            # FIXME: add <cfg>_config settings here once the step has them
```

## test-data/<cluster>/buildtest.yaml

```yaml
targets: [<target>]
target_expectations:
  - target_name: <target>
    step_count: 3              # 1 + one hfpull per hf:// input (not aws: inline) + one hfpush per hf:// output. CALIBRATE on first green run
    input_artifact_count: 1    # number of inputs
    output_artifact_count: 1   # number of outputs (hf://, pushed by the hfpush step)
    # Later: expected_steps[].config (keys the build.yaml set) and .metadata
    # (GB_STEP_METADATA values run.sh prints), once the step has them.
space_uri: ../../space         # IBM: git+ssh://github.ibm.com/granite-dot-build/gb-test.git@gbspace-config
tests: [runner]                # add runner_cancellation only if the workload runs long enough
simulate_step_failure: false
timeout_minutes: 20
```

## USAGE.md outline

1. One paragraph: what the step does and where it runs.
2. Referencing it: `step_uri: space://steps/<path>`.
3. Inputs and outputs table (name, direction, type, required, meaning).
4. Config contract table (key, type, default, purpose): a `FIXME` row until the
   user directs the settings. Once `output_path` exists, say when it
   must be set.
5. A complete worked `build.yaml` target.

No relative links: the file is copied several directories deep on publish. Name repo
docs by plain-text path.
