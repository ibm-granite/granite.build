# Concepts

AutoTuneX is a service for automated fine-tuning and hyperparameter optimization (HPO) of large language models; this page explains the mental model behind it before you touch any endpoint.

If you just want to get something running, start with [Getting started](getting-started.md). This page is conceptual — it explains *what the pieces are and how they relate*, not which HTTP calls to make (see [API overview](api/overview.md) for that).

## What AutoTuneX is

Suppose you want to adapt an existing language model to your own task — teaching it your domain, your tone, your instructions. That adaptation is called **fine-tuning**: you continue training a pretrained model on your own dataset so it behaves the way you need.

Fine-tuning is governed by knobs that you set *before* training starts and that are not themselves learned from the data — the learning rate, the number of epochs, the batch size, and so on. These are the **hyperparameters**. The catch is that the best combination is rarely obvious, and a bad combination can waste hours of compute or produce a worse model than the one you started with.

**Hyperparameter optimization (HPO)** automates the search for a good combination. Instead of guessing, you describe a *space* of possible values, and the system tries candidate points from that space — each candidate is a full training run — and compares how they did.

AutoTuneX ties these together. You describe what to optimize once, as a reusable **configuration**. You submit a **job** that points at that configuration and at a **dataset**. The job then searches the hyperparameter space by running one training **trial** per candidate point, records the metrics each trial produced, and reports which configuration performed best. In one sentence:

> A job searches a hyperparameter space — drawn from a reusable configuration — by running one training trial per candidate point, and reports which configuration performed best.

AutoTuneX is the FastAPI service that stores these records, enforces the rules between them, and hands accepted jobs off to be executed. It does not, by itself, decide *how* your training runs — that is the job of the configured backend, described further down.

## The domain model

Everything in AutoTuneX is one of a small number of entities. Understanding how they relate is most of understanding the product.

| Entity | What it is | Belongs to / references |
| --- | --- | --- |
| **User** | An identity that owns things | — |
| **Configuration** | A reusable set of tuning settings | a user |
| **Dataset** | A named reference to training data | a user |
| **Job** | One optimization run | a user; references a configuration and a dataset |
| **Trial** | One training run inside a job | a job |
| **Result** | The metrics a trial reported | one trial (one-to-one) |
| **Task** | A build/deployment step for a job | a job |

### User

A **user** is an identity — the owner of configurations, datasets, and jobs. It is not itself an owned resource; it is *who owns* the resources.

A user's **role** is strict on the way in and lenient on the way out: a write accepts only `admin` or `user` (`PATCH /users/{id}` rejects anything else with a `422`, and the same two values are the only ones `standalone_role` will start up with), while a read tolerates whatever the column happens to hold — it is nullable, and may carry a legacy or tuning-pipeline-written value. On that read side only the exact value `admin` grants the ability to widen scope, and anything else — including an unset role — is a regular user. Ownership matters on every read: resources are scoped per caller, so by default you see your own configurations and datasets plus a **shared system tier** — curated starter content owned by a reserved system user, visible to everyone — and only your own jobs, which have no shared tier at all. That default holds for admins too. An admin can *choose* to widen a request to see everyone's data, but being an admin does not remove the ownership filter automatically; it grants the ability to ask for the wider view. A regular user cannot ask for it at all.

The practical consequence: when you list your jobs, you get *your* jobs. That is a feature, not a limitation.

### Configuration

A **configuration** is a named, reusable set of tuning settings. You define it once and reference it from as many jobs as you like.

Its settings live in a schema-less JSON object called `config_data`. "Schema-less" is deliberate: the tuning pipeline writes a rich, evolving structure into it, and AutoTuneX does **not** validate that structure against a fixed schema. The only rule the API enforces is that `config_data` must be a non-empty JSON object. This keeps the configuration format free to evolve without the API rejecting valid, real-world configurations.

A configuration also carries two optional labels:

- `tuner_type` — the kind of tuner this configuration describes.
- `rl_tuner_type` — the kind of reinforcement-learning tuner, when the configuration is for RL-based tuning.

Configurations support full create/read/update/delete through the API — they are the resource you author and revise most directly.

Alongside the ones you create, your configuration list also includes the shared system tier: curated starter configurations owned by a reserved system user. You can read them and reference them from your jobs exactly like your own, but they are not yours to change — an attempt to update one behaves as though it did not exist, and a delete is refused for *every* caller with a `403` (`System Resource Protected`): starter content is shared by the whole deployment, so a delete is not a per-tenant loss. The one exemption is an admin with an active impersonation overlay onto the system user (`POST /auth/assume/{id}`).

### Dataset

A **dataset** is a named reference to training data. It has:

- a `name`,
- an optional `description`,
- a `data_format`, one of `jsonl`, `csv`, or `parquet`.

A dataset is more than a label: you upload a file to it, and it moves through a status lifecycle as that upload progresses.

```
empty  →  uploading  →  ready
                  ↘
                   error

ready  →  uploading            (a re-upload)
error  →  uploading            (a re-upload)
```

The lifecycle is not one-way: a `ready` or `error` dataset can be uploaded again, re-entering `uploading`. The only status that refuses a new upload is `uploading` itself — an in-flight upload has to finish first.

| Status | Meaning |
| --- | --- |
| `empty` | Created, but no data uploaded yet. |
| `uploading` | A file upload is in progress. |
| `ready` | Data is present and usable — a job may reference it. |
| `error` | The upload failed. |

A job may only be submitted against a dataset that is `ready`. This is why the dataset lifecycle exists as a first-class thing: it is the signal that the training data is actually available.

Datasets carry the same shared system tier as configurations: alongside your own, your list includes curated starter datasets owned by the reserved system user. You can read them and submit jobs against them, but not change them — an update or an upload aimed at one behaves as though the dataset were not there, while a delete is refused with the same `403` a shared configuration's is, for every caller and with the same single impersonation exemption.

### Job

A **job** is one optimization run, and it is the main unit you interact with. A job references:

- a **configuration** (via `config_id`) — the source of the hyperparameter space to search,
- a **dataset** (via `dataset_id`) — the training data, which must be `ready`.

Alongside those references it carries what to fine-tune and how:

| Field | Meaning |
| --- | --- |
| `model` | Which model to fine-tune. |
| `model_source` | Where the model comes from: `huggingface` or `custom_path`. |
| `experiment_name` | A human-readable name for the run. |
| `tuning_type` | The kind of tuning — **derived** from the referenced configuration, not set by you. |
| `seed` | Random seed for reproducibility (defaults to `42`). |
| `autotune` | Whether HPO is enabled for this run (defaults to on). |
| reward function | An optional `reward_function_code` / `reward_function_name`, required only for online-RL tuning (see below). |

The single most important thing to understand about a job is the **configuration snapshot**. At submission time, the job copies the configuration it references into its own `config_snapshot`. From that moment, the job records what it *actually ran*. If you later edit the configuration, past jobs are unaffected — they still report the exact settings they used, read from their own `config_snapshot`. Deleting it is a different matter: a configuration a job still references **cannot be deleted at all**, and neither can such a dataset — the API refuses the delete with a `409 Conflict`. This makes a completed job an honest, immutable record rather than a pointer that can be changed out from under you.

A job **owns many trials** and **zero or more tasks**.

### Trial

A **trial** is one training run inside a job, evaluating exactly **one concrete point** from the hyperparameter space. If a job's search space has many candidate combinations, the job spawns one trial per combination it decides to try.

A trial carries:

- a short opaque `id` — a brief identifier assigned by the tuning pipeline, not a UUID and not a sequential number,
- a `config` — the *concrete* parameter assignment this trial tested (one point from the space, with actual values filled in),
- its own `status` (the same lifecycle a job has).

A trial is where hyperparameters stop being a range and become a single decided value.

### Result

A **result** is the set of metrics a trial reported — for example an evaluation loss or an accuracy score. It is **one-to-one** with a trial: each trial has at most one result row, and each result belongs to one trial. Keeping metrics in a separate result rather than on the trial itself means a trial that has not reported yet simply has no metrics, rather than a trial cluttered with empty fields.

Results are what the job compares to decide which configuration performed best.

### Task

A **task** is a build or deployment step attached to a job — for example, running the actual tuning, or downloading a produced artifact. A single job may have several tasks, and the API nests them as a `tasks` array on the job **detail** response (`GET /jobs/{id}`) rather than flattening them into the job's own row.

Tasks are how a job's out-of-band build and deployment work is tracked. Treat them as the operational steps that surround a run; the specific kinds available depend on how your deployment is configured.

## The lifecycle: six shared states, one job state machine

Jobs, trials, and tasks all move through the **same six states**:

| State | Terminal? | Meaning |
| --- | --- | --- |
| `pending` | no | Accepted, not yet started. |
| `running` | no | Executing. |
| `paused` | no | Temporarily halted; can resume. |
| `completed` | **yes** | Finished successfully. |
| `error` | **yes** | Finished with a failure. |
| `terminated` | **yes** | Stopped before finishing. |

"Terminal" means there are no outgoing transitions: once a job reaches `completed`, `error`, or `terminated`, it stays there. The one exception is an admin-only force-reconcile: `POST /jobs/{id}/reconcile` writes the status the build backend reports straight onto the job, deliberately bypassing these transitions so a job left in the *wrong* terminal state can be repaired — and it can never rewind a job to a pre-run state. See the [Jobs API](api/jobs.md) and [Job execution and backends](operations/job-backends.md).

The six states are shared vocabulary; the *enforced* transitions are the **job** state machine. Every job status write is validated against the map below, and a move it does not allow is rejected. Trials and tasks report the same six values, but their statuses are recorded as the tuning pipeline reports them — no transition check is applied to them.

For a job, the allowed transitions are exactly these. Reading the arrows as "may move to":

```
  pending  ──→  running          (start executing)
  pending  ──→  completed / error / terminated

  running  ──→  paused           (halt, keep the option to resume)
  running  ──→  completed / error / terminated

  paused   ──→  running          (resume)
  paused   ──→  error / terminated

  completed   ──→  (nothing — terminal)
  error       ──→  (nothing — terminal)
  terminated  ──→  (nothing — terminal)
```

The same job state machine as a table of "from → allowed next":

| From | Allowed transitions |
| --- | --- |
| `pending` | `running`, `completed`, `terminated`, `error` |
| `running` | `paused`, `completed`, `error`, `terminated` |
| `paused` | `running`, `terminated`, `error` |
| `completed` | *(none — terminal)* |
| `error` | *(none — terminal)* |
| `terminated` | *(none — terminal)* |

Two subtleties worth calling out:

- **`paused` can only return to `running`, or move to a terminal `error`/`terminated`.** A paused run cannot jump straight to `completed`; it resumes first.
- **`pending → completed` is legal.** A run can be observed as already finished — for instance, if the service was not watching during the entire `running` phase and only checked in after the work was done. Rather than leave such a run stuck forever, AutoTuneX records what genuinely happened.

## How a job flows end to end

Putting the entities and the state machine together, a typical job looks like this:

1. **You submit a job.** It references a configuration and a `ready` dataset. AutoTuneX validates that you own both — or that they are shared system content — snapshots the configuration into `config_snapshot`, and accepts the job in the `pending` state. The job is yours either way: drawing on shared starter content does not make the job shared. Submission never blocks on training — tuning runs are long, so the API hands the job off and returns immediately.
2. **A runner picks it up.** A backend executes the job, moving it to `running`.
3. **Trials run.** For each candidate point in the search space, the job runs a trial, and each trial records its metrics as a result.
4. **The job reaches a terminal state** — `completed` on success, or `error` / `terminated` otherwise.
5. **The best-performing configuration is reported**, chosen by comparing the trials' results.

There is one crucial caveat at step 2. **Whether a submitted job actually executes depends on the configured job backend.** AutoTuneX ships with a default backend that *accepts* a job and stores it, but leaves it `pending` — nothing runs it. To have jobs actually execute, you configure a real backend. See [Job execution and backends](operations/job-backends.md) for what is available and how to choose one.

So a job sitting at `pending` is not necessarily broken — it may simply mean the default no-op backend is in effect.

## Online-RL versus offline tuning

Some tuning methods are based on reinforcement learning (RL), and among those, **online-RL** methods generate fresh model outputs during training and need to *score* them. That scoring is done by a **reward function**.

Because of that, online-RL tuners require a reward function to be supplied when you submit the job. AutoTuneX keys this requirement on the configuration's **`rl_tuner_type`** field (not the more general `tuner_type`): if `rl_tuner_type` is one of `ppo`, `grpo`, or `dapo`, a reward function is mandatory. The comparison is case-insensitive. Offline-RL tuners (`dpo`, `kto`) and plain supervised fine-tuning carry no such requirement:

| `rl_tuner_type` | Kind | Reward function required? |
| --- | --- | --- |
| `ppo` | online-RL | **yes** |
| `grpo` | online-RL | **yes** |
| `dapo` | online-RL | **yes** |
| `dpo` | offline-RL | no |
| `kto` | offline-RL | no |
| *(unset — plain SFT)* | non-RL | no |

If your configuration is for an online-RL tuner and you submit a job without a reward function, AutoTuneX rejects the submission. Offline-RL and plain supervised fine-tuning need no reward function — they learn directly from the dataset.

## The relationship in one line

> A **job** references a **configuration**'s search space and a **dataset**, is owned by a **user**, runs **trials** (each reporting metrics via **results**), and may have **tasks** attached.

Once that sentence reads as obvious, you understand AutoTuneX's model. From here:

- [Getting started](getting-started.md) — set up and make your first request.
- [API overview](api/overview.md) — conventions shared across every endpoint.
- [Job execution and backends](operations/job-backends.md) — what actually runs your jobs.
