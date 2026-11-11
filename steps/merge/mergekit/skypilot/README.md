# mergekit step (developer notes)

> **Using the step?** See [USAGE.md](USAGE.md). It is published as the step's `README.md`.

Source for `space://steps/merge/mergekit` (issue #238). `make publish-step` renders it into
`configurations/assets/environments/skypilot/steps/merge/mergekit/`, with build tests under
`test/steps/merge/mergekit/skypilot/` and fixtures under `test-data/steps/...`. Never
hand-edit those copies. Edit the source here and re-publish. Framework rules are in
[steps/README.md](../../../README.md).

## Type

Public-image step: there is no Dockerfile. `setup:` installs `mergekit==<mergekit_version>`
into `./venv` with uv (the `data/data-designer` pattern), with the uv cache and the managed
Python under `GB_SHARED_WORKDIR`. Setting `mk_image` skips the install.

## Design decisions

- **Inputs:** `model_1` (required), `model_2`–`model_4` and `base_model` (optional), all
  `type: model`. MergeKit's own examples merge 2–3 models.
- **Model references:** `merge_config` is MergeKit's recipe, passed through unchanged
  except for one substitution. A `<model>` or `<lora>` part naming a bound input is
  replaced with that input's staged path. This applies to the string form and to
  MergeKit's mapping form (`{model, lora, override_architecture}`).
  - A `{path, revision}` ModelPath is literal in MergeKit (no `@`/`+` parsing), so its
    whole `path` is matched against the input names and `revision` is left alone.
  - Hub ids and node paths pass through untouched, so MergeKit downloads or reads them
    itself.
  - An input name that is used but not bound is refused. Otherwise MergeKit would try to
    fetch it as a Hub repo, and fail only after provisioning.
  - A bound input the recipe never names is refused too: it is most often a typo
    (`model1` for `model_1`) that would also go to the Hub, or merge without that input.
  - Input names rather than `{{ bindings.<name>.binding.path }}`. A bound one renders to
    the staged path, which counts as naming the input; an unbound one reaches the node
    as literal `{{ ... }}` text (the non-strict config fill keeps unknown expressions),
    which is refused at render time and on the node.
  - A staged path containing `@` or `+` is refused, since MergeKit would mis-parse it.
- **Inline `chat_template`:** gbserver's non-strict config fill (`targetsteprun.py`,
  `fill_objtemplate`) drops `{% %}` blocks and writes unknown `{{ var }}` back as text,
  so an unwrapped template arrives empty or mangled. Only three values are accepted:
  - `null`;
  - a template name MergeKit ships (`auto` plus its `*.jinja` files);
  - a template that still has its `{%` blocks, which survive only inside `{% raw %}`.

  A looser name test lets plain-text leftovers such as `ASSISTANT:` through, and MergeKit
  then rejects them inside a `try` that hides the error (see the output checks below).
- **Flags:** `args` follows dpk's mapping, adapted to MergeKit's click on/off flags (`true`
  → `--k`, `false` → `--no-k`). Flags the step owns are refused, including
  `safe-serialization`, because the output check counts `.safetensors` tensors. Empty
  values are refused, because run.sh reads the flags back one per line and an empty line
  would shift the next flag onto that option.
- **Precision** stays in the recipe (`dtype`, `out_dtype`). There is no step-level key.
- **Staging:** MergeKit writes into `<output parent>/.mergekit-stage.*/<output name>`,
  which is renamed onto the output only once every check passes. MergeKit therefore sees
  the real output name, which it uses as the title of the model card it writes.
- **Retries:** a retry of the step finds the output an earlier attempt already moved
  into place, which `run.sh` would otherwise refuse as existing. The usual one is the
  in-place build retry (`retries.max_retries`, and `gb build restart`): it keeps the
  build id, skips targets that succeeded (with target reuse on, the default), and
  re-runs a failed target with new target and step runs. gbserver's step-level retry
  (`retry_enabled`, off for this step by default) relaunches the same step run. So the
  step passes `--retry-key` `<build id>/<target>/<step index>`, the same on both, rather
  than a run id. Before the move, `run.sh` writes `.<output name>.mergekit-done` beside
  the output (not in it, so a push of the model doesn't carry it) with that key, a hash
  of the merge's inputs (recipe, `args`, flags, input and output paths, not the inputs'
  contents) and the markers. A retry with the same key and hash re-emits the markers,
  plus `reused_earlier_attempt: true`; anything else (another build or step, or a
  changed recipe) still gets the "already exists" refusal. `test/slurm-retry` runs an
  in-place build retry end to end.
- **Caches:** the uv cache, the managed Python and MergeKit's Hub download cache
  (`.mergekit-hf-cache`) are under `${GB_SHARED_WORKDIR:-$PWD}`, so a shared workdir
  downloads each once per cluster. They are created at the first user's umask and never
  pruned; a workdir shared between users needs them group-writable. The LoRA merge cache
  stays per run: MergeKit reuses any directory already at a merged adapter's path, so a
  shared one could serve one run another's partial merge.
- **Fixture output** is `env:///shared/...` because `HfPushCleanupMixin` only handles
  dataset repos.

## Checks

**At render time**, the `validate` macro in `run:` raises at step creation, so a bad
config fails before the merge node is provisioned. The target's hfpull steps run first,
so they still download. The macro checks:

- `merge_config` is present, and has exactly one of `models`, `slices` and `modules`
  (lists, and a mapping for `modules`).
- `merge_method` is a non-empty string, and one of MergeKit 0.1.5's registered methods.
  `test_merge_method_list_matches_mergekit_registry` keeps the list in sync. The same
  goes for the chat-template names. Both are checked only when the step installs 0.1.5
  itself; another `mergekit_version`, or an `mk_image`, may differ, and MergeKit then
  checks them itself.
- `args` is a mapping, with well-formed keys and scalar, non-empty values, and sets no
  step-owned flag.
- The booleans are booleans, and the string keys are strings. `mergekit_version` is
  required unless `mk_image` is set.
- `chat_template` passes the rule above, and has a tokenizer to go in: `copy_tokenizer`,
  or a `tokenizer_source` or `tokenizer` block.
- Every input name used in a string reference in `base_model`, `models[]` or
  `slices[].sources[]` is bound, no such reference is an unfilled `{{ ... }}`, and every
  bound input is named by some string value in the recipe (as a reference's `<model>` or
  `<lora>` part, or by its staged path). Keys don't count, so a `base_model:` key alone
  doesn't make a bound `base_model` used. For references written as input names or in
  the bindings form, every reference field is a string value, so this doesn't refuse
  what prepare accepts (a hand-written path that differs from the binding only by a
  trailing `/` or a symlink is the exception). Prepare checks the reference fields
  themselves, and also catches a name that appears only outside them.
- `num_nodes` is 1, and a GPU count comes with a GPU request (`accelerators` or
  `instance_type`, #536). Both use gbserver's own rules for these values.

`mergekit_prepare.py` repeats the recipe-shape, `args`, `chat_template` and binding
checks on the node, for a direct run. It also covers what only the node
can see: the installed mergekit's methods and template names, and mapping-form
references. `test_render_and_prepare_agree` runs one table of cases through both and
checks they agree. The `num_nodes`, GPU and `mergekit_version` checks are render-only,
since they concern the launch rather than the recipe.

**After the merge**, `run.sh` checks the output before registering it, because MergeKit
reports some failures as success:

- **Copying** a tokenizer (`copy_tokenizer`) happens inside a `try` that logs "Failed to
  copy tokenizer ... still successful" on any error, including a copy that fails partway.
  `run.sh` tees MergeKit's stderr to `.mergekit/mergekit.log` and fails on that line.
- **Building** a tokenizer (`tokenizer_source` or `tokenizer`) happens outside that
  `try`, so a failure there fails the merge.
- `chat_template: auto` sets no template at all when none of the input models has one.

A tokenizer is `tokenizer_config.json` plus a vocabulary file: `tokenizer.json`, or a slow
tokenizer's own (`tokenizer.model`, `spiece.model`, `sentencepiece.bpe.model`,
`vocab.json`, `vocab.txt`, `tokenizer.tiktoken`), since MergeKit's reserialize path
(`save_pretrained`) writes only those for a tokenizer with no fast version.

`mergekit_prepare.py` decides once what the output must contain, and writes it to
`.mergekit/expect`: `tokenizer` when MergeKit copies or builds one, and `chat_template`
when one is requested. `run.sh` then checks the output against that, on every path. The
output must also have a `config.json` and more than 0 tensors.

## Layout

| Path | What |
|---|---|
| `step-template.yaml` | The step's `step.yaml` source, including the `validate` macro |
| `src/setup.sh` | uv install of `mergekit==<mergekit_version>` (data-designer's `dd_setup.sh` contract) |
| `src/mergekit_prepare.py` | Input-name substitution, `args` to flags, the node-side config checks, and the `expect` file. Standard library plus PyYAML, so it runs in the venv or an `mk_image` |
| `src/run.sh` | Path checks, the merge into the staging dir, the output checks, the move into place, and the markers |
| `test/test_skypilot_merge_mergekit_render.py` | Render and `validate` tests with gbserver's strict renderer, and the render/prepare agreement table |
| `test/test_mergekit_prepare.py`, `test/test_mergekit_prepare_checks.py` | Substitution, flag rendering and config checks (the second: recipe shape and bindings) |
| `test/test_mergekit_run_sh.py`, `test/test_mergekit_run_sh_retry.py` | `run.sh` against a stub `mergekit-yaml`, and real-MergeKit tests (see below); the second: retries, the shared Hub cache, slow-tokenizer files |
| `test/mergekit_test_helpers.py` | The stub `run.sh` harness and the `main()` runner the test files share (a module, not a `conftest.py`, so `make test-mergekit` loads it with `--noconftest`) |
| `test/test_mergekit_setup_sh.py` | `setup.sh` with stub `pip`/`uv`, including a second run in the same workdir (a SkyPilot retry), plus one real-`uv` run |
| `test/slurm/`, `test-data/slurm/` | Build test on the local Docker SLURM cluster (`make test-setup`) |
| `test/kube/`, `test-data/kube/` | Build test on the local kind cluster (`make test-setup-kube`) |
| `test/slurm-retry/`, `test-data/slurm-retry/` | An in-place build retry on local SLURM: a step after the merge fails once, and the retried merge reuses its output |

## Running

```bash
make -C steps/merge/mergekit/skypilot space              # check the template renders
make -C steps/merge/mergekit/skypilot test-setup         # once: local SLURM
make -C steps/merge/mergekit/skypilot test-setup-kube    # once: local kind
make -C steps/merge/mergekit/skypilot test               # Mode 1 tests + build tests
make -C steps/merge/mergekit/skypilot test TESTS=test/slurm
make -C steps/merge/mergekit/skypilot test-mergekit      # the real-MergeKit tests (below)
```

## Tests

`make test` runs the render, prepare and stub-based `run.sh` tests, plus the build tests
when a cluster is up. The tests that run real `mergekit-yaml` skip there, because the repo
venv must not get mergekit: its dependencies move `click` past SkyPilot's `<8.2` pin. They
are the MergeKit round-trips and sync checks, the real merges, and
`test_real_merge_matrix`.

`make test-mergekit` runs them in a separate venv, built once in your cache dir
(`~/.cache/granite-build/mergekit-test-venv-0.1.5`, or set `MERGEKIT_TEST_VENV`), outside
the repo so `make lint` and `make format` never scan it. A venv from an older checkout at
`.venv-mergekit-test` in this directory is no longer used and can be deleted. The run takes
about five minutes. `test_real_merge_matrix` runs 60 real merges: 6 tokenizer paths × 5
chat-template options × inputs with and without a template. For each, it checks plain
MergeKit's output independently, and asserts the step registers exactly the good merges.

**Re-running a build test:** the fixtures write to a fixed path, and `run.sh` refuses to
overwrite a non-empty output, so one build can't overwrite another's model. Clear the
previous output first:

```bash
docker exec slurm-c2 rm -rf /shared/mergekit-test                                 # local SLURM
docker exec skypilot-control-plane rm -rf /var/local/gb-shared/mergekit-test      # local kind (the gb-shared PV)
```

**Not covered by a build test yet:** GPU merges (`--cuda`), `mk_image`, `hf://` outputs,
aws and BlueVela, LoRA merges (`<model>+<lora>`), and `modules` recipes. A LoRA merge has a
real-MergeKit test in `make test-mergekit`; `modules` has unit tests only.
