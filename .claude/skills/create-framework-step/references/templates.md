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
`required:` is null, not an empty map. Working examples to copy from:
`steps/distill/vllm-server/skypilot` (nested under a category, like a new step),
`steps/dpk/skypilot` (public image, python `src/`, five clusters) and
`steps/eval/skypilot` (custom image). Add the repo's Apache-2.0 header to every `.py`.

## Makefile

```make
# <name> step for the <env> environment. FIXME: one line on what it does.
# Public image: no Dockerfile here, so common.mk treats it as a public-image step.
# Custom image: the Dockerfile beside this file makes it a custom-image step.

STEP_NAME := <name>

# Custom image only. FIXME: the registry and namespace the image is pushed to.
REGISTRY ?= quay.io/FIXME

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

A custom-image step also sets `REGISTRY := quay.io/<org>` before the `include`.

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
