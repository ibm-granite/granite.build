# lm-engine-pretrain

Pretrain an [lm-engine](https://github.com/IBM/lm-engine) model from scratch on a
SkyPilot-on-Kubernetes cluster. One target is one pretraining job: it renders
lm-engine's `config.yml` from `config.lme`, launches the run, and emits the
checkpoints as artifacts.

Reference it from a `build.yaml` as:

```yaml
step_uri: space://steps/train/lm-engine/pretrain
```

## Configuration

| Key | Default | What it does |
| --- | --- | --- |
| `image` | `""` | The lm-engine image reference, pinned by tag. Required. |
| `phase` | `train` | `train` runs pretraining; `data-check` only validates and summarises the tokenized corpus, then exits. |
| `train_dir` | `""` | The run's training directory. Required, and it must be a path on the cluster's shared filesystem that is stable across attempts of the same build, so a relaunch resumes from what the previous attempt wrote. |
| `lme` | `{}` | Every lm-engine knob, as the nested mapping lm-engine's `config.yml` expects. `{}` leaves lm-engine's own defaults. |
| `ckpt_rungs` | `""` | Comma-separated optimizer step counts at which to emit a checkpoint artifact mid-run, e.g. `"2000,4000"`. |
| `emit_checkpoint_artifacts` | `false` | Run the rank-0 checkpoint watcher at all. Set it to `true` when `ckpt_rungs` is set. |
| `checkpoint_watch_interval_seconds` | `60` | How often the watcher looks for a newly completed rung. |
| `require_progress` | `true` | Fail the step when the job made no optimizer progress, instead of reporting success for a job that wrote nothing. |
| `resume_from_checkpoint_dir` | `""` | Resume from this directory instead of the latest checkpoint under `train_dir`. Empty auto-detects, and starts fresh when `train_dir` holds no checkpoint. |
| `nccl_env` | `{}` | Cluster facts (interface names, HCA lists, timeouts), exported verbatim into the job's environment. Keys are used as given, so a cluster's `NCCL_*` and `UCX_*` settings belong here. |
| `nccl_debug` | `WARN` | `NCCL_DEBUG` for the job. |
| `wandb.project` | `""` | Weights & Biases project. |
| `wandb.mode` | `offline` | `offline` needs no API key; `online` needs one, `disabled` turns tracking off. |
| `wandb.name` | `""` | Run name; empty lets lm-engine pick one. |
| `wandb.entity` | `""` | W&B entity. |

## Inputs

The step declares no named inputs: the corpus and the tokenizer arrive through
`config.lme`, as paths inside the image or on the cluster's shared filesystem, not
as bindings. `inputs.allow_unknown` is `true`, so a build may bind an
ordering-only input (a corpus-prep target's output, say) to sequence the targets
without the step reading it.

## Outputs

| Output | Type | When |
| --- | --- | --- |
| `checkpoint` | `model` | Optional: the final checkpoint of a `train` run. A `data-check` run emits nothing. |
| `checkpoint_<N>` | `model` | One per entry in `ckpt_rungs`, emitted as the run passes step `N`. The step cannot declare these — the rungs are a build-time choice, so the build that sets `ckpt_rungs` declares the matching outputs. |

## Example

```yaml
targets:
  - name: pretrain
    step_uri: space://steps/train/lm-engine/pretrain
    config:
      image: quay.io/granitebuild/gb-step-lm-engine:v0.1.0
      train_dir: /shared/runs/my-run/train
      ckpt_rungs: "2000,4000"
      emit_checkpoint_artifacts: true
      lme:
        model:
          n_layers: 24
        trainer:
          max_steps: 4000
      wandb:
        project: my-pretrain
    outputs:
      checkpoint: env:///shared/runs/my-run/artifacts/final
      checkpoint_2000: env:///shared/runs/my-run/artifacts/step-2000
      checkpoint_4000: env:///shared/runs/my-run/artifacts/step-4000
```

An `env://` output that another target reads must be a path on the cluster's shared
filesystem: the per-run working directory is deleted at target teardown.
