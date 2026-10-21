---
name: create-framework-step
description: Use when creating a new Granite.build step under the repo's steps/ step framework (steps/<category>/<name>/<env>/ with step-template.yaml, Makefile including common.mk, USAGE.md, per-cluster build tests, make publish-step), or adding build tests for another cluster to such a step. Not for ad-hoc file:// steps in a user project (use create-step) or inline command workloads (use create-build).
argument-hint: "[step-name]"
---

# Create a step in the steps/ framework

A framework step is **source** under `steps/<path>/<env>/`, where `<path>` is
`<category>/<name>`, that `make publish-step`
promotes into three committed, generated trees (`configurations/assets/...`,
`test/steps/...`, `test-data/steps/...`). You author only the source; never hand-edit
the generated trees. Framework rules live in `steps/README.md`; this skill is the
authoring recipe.

## 1. Ask, then generate

Make **two** AskUserQuestion calls (the tool takes at most four questions per call),
skipping any question the prompt already answers. Ask nothing else before generating.
Free-text answers come through the tool's "Other" box, so each such question offers a
default option and says in its text to type the answer under "Other".

**Call 1: what the step is**

1. **Step name** (`<name>`): options `new-step` (placeholder, rename later) and
   "Type a name under Other". Lowercase, hyphens ok, no slashes.
2. **Category** (`<category>`): options `eval`, `train`, `tune`, `data` (the tool adds
   "Other" for the user's own). It becomes the directory under `steps/`:
   `steps/<category>/<name>/<env>/`, published as `space://steps/<category>/<name>`.
3. **Inputs**: "A comma-separated list of input names, typed under Other." Options
   `input` (one input, the default) and `none`.
4. **Outputs**: "A comma-separated list of output names, typed under Other." Options
   `output` (one output, the default) and `none`.

**Call 2: where it runs**

1. **Environment dir** (`<env>`): `skypilot` (Recommended; almost all new steps),
   `docker`, or `bash`.
2. **Step image**: `Public image` (Recommended; no Dockerfile, runs on the bare node
   or a public image) or `Custom image` (a Dockerfile builds the step's own image,
   pushed to a registry).
3. **Build-test clusters** (multiSelect): the rows for that env in
   [references/clusters.md](references/clusters.md). Each selected cluster becomes one
   `test/<cluster>/` + `test-data/<cluster>/` pair. Never pick clusters yourself. Note
   in the local `slurm` option that it runs public-image steps only (no Pyxis). If the
   user still picks it with a custom image, generate it and flag it in the hand-off.

**Normalizing the answers.** Split the input and output lists on commas and trim them.
Turn each name into a Jinja-safe identifier: lowercase, hyphens and spaces become
underscores (`bindings.<name>` and `GB_ARTIFACT_ID:<name>` both use it). If you changed
a name, say so in the hand-off. `none` means an empty list.

**Input names that are Python dict methods** (`items`, `keys`, `values`, `get`, `pop`,
`update`, `copy`, `clear`, `setdefault`, `popitem`, `fromkeys`): in Jinja,
`bindings.items` is the dict's method, not the binding, and the launcher fails to
render at launch. Keep the name, but write every reference to it with a subscript:
`bindings["items"].binding.path`, and `"items" in bindings` instead of
`bindings.items is defined`. Say in the hand-off which inputs got this, and that
renaming them is the simpler fix.

Then generate the whole step. Take every other decision from the prompt when it's
there; otherwise use the placeholder below. Every placeholder carries a `FIXME`
comment (or is the literal value `FIXME`), so `grep -rn FIXME` finds all of them.

| Decision | Rule | Placeholder when not given |
|---|---|---|
| **Name** | Asked (call 1). `STEP_NAME` is a slash-free leaf, placed under the category: `steps/<category>/<name>/<env>/`. Check `configurations/assets/environments/*/steps/` and `src/gbserver/builtins/steps/` for an existing step of that name first, and if one exists, say so in the hand-off. If `steps/<category>/` is itself a step (it has an `<env>/` dir with a `step-template.yaml`), say so too: once that step is published, its `check-published` reports the nested step as drift. No category dir is a step today, but check. Generate anyway. | When `new-step` was chosen: `# FIXME: rename (git mv the dir; update STEP_NAME, name:, <cfg>_config and every step_uri)` on its own line above `STEP_NAME` (a trailing comment on a Make assignment keeps the spaces in the value) |
| **Step image** | Asked (call 2). **Public:** no Dockerfile; `image_id: ""  # FIXME: set a public image if the code needs one`; `file_mounts: {src: src}` ships the scripts. **Custom:** the Dockerfile skeleton, `REGISTRY` in the Makefile, `image_id: "docker:${IMAGE_REF}"`, scripts baked in at `/opt/<name>/` (no `file_mounts`). Both variants are in [references/templates.md](references/templates.md). | — |
| **Inputs** | Names asked (call 1). Each name becomes a required input with `type: dataset`, `accept: [uri, binding]` and `# FIXME: type, required/optional`, a `--input-<name>` argument in `run:`, and a `uri: FIXME` input in each fixture. | — |
| **Outputs** | Names asked (call 1). Each name becomes a required output with `type: dataset` and `# FIXME: type`, a `--output-<name> ./<name>` argument in `run:`, one artifact marker in the `run.sh` stub's contract, and an `hf://` output in each fixture: `hf://huggingface.co/datasets/__GBTEST_HF_PUSH_ORG__/<name>-<output>___GBTEST_HF_PUSH_RUN__` with a `# FIXME: dataset name`. Never `env://` in fixtures. Each build test then mixes in `HfPushCleanupMixin` (`libgbtest.hfpush`), which fills in those tokens per run, checks the push landed, and deletes the dataset. A `mem://` value output (`GB_ARTIFACT_STATE`) is a later change. | — |
| **Configuration** | One block `config.<cfg>_config:`, where `<cfg>` is the name with hyphens turned into underscores (`jsonl-dedup` → `jsonl_dedup_config`; a hyphen parses as subtraction in Jinja). **Every key defaulted**, plus `compute_config` if it needs sizing. | **Leave it empty** (`<cfg>_config: {}`) with a `FIXME`, even if the prompt describes settings: the user directs those after the skeleton exists. Nothing references a config key yet |
| **Implementation** | Scripts in `src/`, called from `setup`/`run` with values as argv. | **Don't implement anything.** Create only `src/setup.sh` and `src/run.sh`, holding the comment-only stubs in [references/templates.md](references/templates.md): what each is for and how step.yaml calls it. No other files in `src/` |
| **Test input data** | A small, public dataset (usually `hf:///datasets/...`, so no `HF_TOKEN`) with a known result. Don't invent a repo id. | `uri: FIXME` in every fixture `build.yaml` (fails build validation until replaced) |
| **Expected counts** | `input_artifact_count` = number of inputs, `output_artifact_count` = number of outputs. `step_count` = 1, plus one hfpull per `hf://` input (none on aws, which pulls inline), plus one hfpush per `hf://` output. | The computed `step_count` with `# CALIBRATE against the first green run` |

## 2. Files to create

Skeletons for every file are in [references/templates.md](references/templates.md).

```
steps/<path>/<env>/
  Makefile              STEP_NAME + include ../../../common.mk (category/name/env = 3 levels)
  step-template.yaml    name, version, type, config, inputs, outputs, environment_configs
  USAGE.md              user doc, published as README.md; self-contained (no relative links)
  README.md             dev doc; not published
  src/setup.sh          comment-only stub, called by the launcher's setup:
  src/run.sh            comment-only stub, called by the launcher's run:
  Dockerfile            custom-image steps only (skeleton in templates.md)
  .gbignore             files shipped verbatim, not Jinja-rendered (docs, src/); required:
                        make space and publish-step copy it, and refuse a step without one
  test/test_<env>_<category>_<name>_render.py   render test: passes on the skeleton (Mode 1 only)
  test/<cluster>/test_<env>_<cluster>_<category>_<name>.py   one build test per cluster
  test-data/<cluster>/{build.yaml,buildtest.yaml}    its fixtures
```

## 3. Rules for each part

**Inputs.** Declare every input the step reads and read it as
`{{ bindings.<name>.binding.path }}` (`.state` for `mem://`). Guard optional ones with
`{% if bindings.<name> is defined %}`. Don't pass input paths through config keys.
For a name that is a dict method, use the subscript forms from "Normalizing the
answers" above.

**Outputs.** Output URIs are **not** in the render context, so the step can't template
an output path. It writes to a path it picks (default `./output`, made absolute) or
one from config (`<cfg>_config.output_path`, which must equal the path in an `env://`
output `uri`), then prints `GB_ARTIFACT_ID:<output> GB_ARTIFACT_PATH:<abs-path>`. The ID
must equal the target's output name.

**Configuration.** Launcher text renders strictly: a missing leaf key renders as `""`
and a missing parent block is an error at launch. So every key needs a default in
`step-template.yaml`. Escape every value that goes into the shell with a `q()` macro,
defined in **both** `setup` and `run`, because Jinja macros are block-scoped. Pass
values to `src/*.sh` as argv; keep shell logic in scripts, not in the template.
The dict-method trap applies to config keys too: when the user later asks for a key
like `items` or `keys`, suggest another name, or read it as
`config.<cfg>_config["items"]`.

**Build tests.**
- Basenames must be unique across all cluster dirs (no `__init__.py`; duplicates
  fail collection), including other categories' published copies under `test/steps/`. Use
  `test_<env>_<cluster>[_<variant>]_<category>_<name>.py`, with every hyphen
  turned into an underscore (`test_skypilot_slurm_bluevela_text_norm.py`).
- Every build test gets `@extended_testing_only` and a skip gate for its infrastructure.
- Mode-1 fixtures use `space_uri: ../../space`; `publish-step` rewrites it.
- **A fixture whose `space_uri` points at `github.ibm.com` gets
  `pytestmark = pytest.mark.ibm`.** So does a test on an IBM-only cluster (BlueVela) that uses the
  local space. A remote `space_uri` is copied through publish unchanged, and it resolves
  the step **from that remote space**, not from your source. Say so in the test
  docstring. Tell the user the step must be published into the assets repo that
  space's `base_uris` chain to, before that test can pass. That publish is outside
  `make publish-step`; leave it to the user.
- `timeout_minutes` above 20 needs an override, because pyproject's global pytest
  timeout is 1200s: `@pytest.mark.timeout(<minutes*60+300>)` on a `test_runner` that
  calls `super().test_runner()`.
- Assert what the build set (`expected_steps[].config`), not template defaults; those
  aren't persisted. To assert the step's actual result (the mixin only checks that
  each `hf://` output landed, not what's in it), have the run script emit
  `GB_STEP_METADATA_KEY:<k> GB_STEP_METADATA_VALUE:<v>` and check it with
  `expected_steps[].metadata` (exact) or `metadata_matches` (regex).
- One setup target per local cluster: `test-setup` for slurm, `test-setup-kube` for
  kube (see `steps/data/dpk/skypilot/Makefile`). Set `HAS_TEST_SETUP := true` only when `test-setup`
  itself is defined (slurm selected); otherwise common.mk's no-op default must stay.
- `step_count` differs by cluster: see the catalogue. Mark uncalibrated counts
  `# CALIBRATE against the first green run`.

## 4. Don't run anything

Generate the files and stop. Don't run `make`, the tests, a build, or a formatter: the
user fills in the `FIXME`s first and runs everything later. Never run
`make publish-step` (it would commit a placeholder step into `configurations/assets`)
and don't commit.

## 5. Hand off: tell the user their next steps

End with this report. Build the placeholder lines from `grep -rn FIXME steps/<path>/<env>`.

1. **What was generated**: the directory, the answers from both calls, and any
   values taken from the prompt.
2. **Warnings**, if any apply: a custom image with the local `slurm` cluster selected;
   a category dir that is itself a step (see the Name row).
3. **Your next steps**, as a numbered list in this order. Under each, list its
   `FIXME`s as `file:line`: what it stands for and what to put there.
   1. **Only if the name is still `new-step`**: rename it (`git mv` the directory, then
      update `STEP_NAME`, `name:`, `<cfg>_config` and every `step_uri`). Otherwise skip.
   2. **Check the input and output types**: each defaults to `type: dataset`, required.
      Change any that should be `fileset`, `model` or `bucket`, or optional.
   3. **Describe the configuration** you want in `<cfg>_config`. Say the user can just
      tell Claude the settings, and Claude will add them to the template, `run:` and
      `USAGE.md`.
   4. **Image**: public: set `image_id` if the code needs one. Custom: set `REGISTRY`
      and add the step's dependencies to the `Dockerfile`.
   5. **Implement `src/setup.sh` and `src/run.sh`**. The build tests can't pass until
      `run.sh` prints the `GB_ARTIFACT_ID … GB_ARTIFACT_PATH …` marker. When it does, the
      render test's `xfail` turns into a failure: delete that marker line. Add a render
      test for each config setting as you add it.
   6. **Pick test data**: replace each `uri: FIXME` input in the fixture `build.yaml`s
      (they fail validation until then), and confirm each output's dataset name (the
      `# FIXME` above its `hf://` uri). Pushing needs an `HF_TOKEN` that can write
      `GBTEST_HF_PUSH_ORG` (default `ibm-research`; locally use your own HF user). Add
      `expected_steps` checks once results are known.
   7. **Write `USAGE.md`** for the step's users, and `README.md` for its developers.
   8. **Run it**. The render test passes from the start and is the fastest check; it
      runs only here, not in PR CI (which runs only `test/`):

      ```bash
      make -C steps/<path>/<env> space                  # check the template renders
      make -C steps/<path>/<env> test-setup             # once, if slurm was selected
      make -C steps/<path>/<env> test-setup-kube        # once, if kube was selected
      make -C steps/<path>/<env> image publish-image    # custom image only (needs REGISTRY + login)
      make -C steps/<path>/<env> test                   # render test + build tests (custom image: builds it first)
      ```

      `make test` sets `GB_ENVIRONMENT=STANDALONE` and
      `GBTEST_STANDALONE_ENVIRONMENT=STAGING` unless you set them, so `hf://` pushes go
      to the staging group (see steps/README.md, "`test`").

      Then calibrate each `step_count` marked `CALIBRATE` from the first green run, and
      format the test files (`isort --profile black`, `black`; `mypy` on any Python in
      `src/`).
   9. **Publish** when no `FIXME` remains: `make -C steps/<path>/<env> publish-step`,
      then `check-published` (must print OK). For an IBM-space test, also publish the
      step into the assets repo that space's `base_uris` chain to.
      **Then add the step to the `EXPECTED` inventory** in
      `test/unit/asset/test_step_inputs_declared.py`, or PR CI fails
      `test_the_inventory_matches_disk`. Give the user the exact line, built from the
      published dir and the step's inputs (see "The `EXPECTED` line" below). Re-check it
      if inputs were renamed, retyped or made optional since generation.
   10. **Commit** the source and the generated trees.

### The `EXPECTED` line

The key is the published dir relative to `configurations/assets`:
`environments/<env>/steps/<category>/<name>` (`PUBLISH_STEP_DIR` in common.mk). For
`<env>` = `skypilot`, use the file's `_SKY` prefix. The value is
`(required, optional)`, the input names as Python sets; write `set()` for empty:

```python
    f"{_SKY}/data/text-norm": ({"docs", "vocab"}, set()),
```

All of the skeleton's inputs start out required. If the step has **no** inputs, the
entry is `(set(), set())`, and the test also requires a comment of at least four words
directly above `inputs:` saying why (not starting with `FIXME`). The skeleton writes
one when the user answered `none`; if inputs were removed later, add it.

## Common mistakes

| Mistake | Fix |
|---|---|
| Editing `configurations/assets/.../steps/<name>/` or `test/steps/` | Edit `steps/<path>/<env>/`, re-run `make publish-step`. |
| Choosing clusters without asking | Ask (multi-select) from the catalogue. |
| Asking anything beyond the two calls' seven questions (config, types, image settings, test data) before generating | Generate with placeholders; list them in the hand-off. |
| Running `make`, tests or formatters after generating | Don't. List the commands in the hand-off. |
| A placeholder with no `FIXME` marker | Every placeholder carries `FIXME`, or the hand-off grep misses it. |
| IBM-space test without `pytest.mark.ibm` | Set `pytestmark = pytest.mark.ibm`. |
| Same test filename in two cluster dirs | Make basenames unique. |
| `build.yaml` output `env:///x` but the step writes `./output` | Set `<cfg>_config.output_path: /x`. |
| Cross-target output in the per-run workdir | A terminal `env://` output can be any node path (`env:///tmp/<name>-out`). One that another target reads needs a path on the cluster's shared filesystem (`/shared` on local slurm and kube; BlueVela: the environment's `shared_workdir`), because the workdir is deleted at target teardown. |
| A long-running test killed at 20 min | Add the `@pytest.mark.timeout` override. |
| Running `make test` from the repo without `-s` | `make test` already passes `-s`; SkyPilot multi-launch tests need it. |
