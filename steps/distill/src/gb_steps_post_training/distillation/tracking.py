"""ClearML / W&B experiment tracking: ONE place that decides whether either backend is on, and why.

THE RULES THIS ENCODES (as specified for this project):
  * ClearML is the DEFAULT backend. W&B is offered as an OPTIONAL extra.
  * A backend is enabled only with a COMPLETE config -- project, and for W&B also entity.
  * No credentials present -> that backend is not used.
  * The run name may be omitted, and is then auto-generated from the build yaml.

WHY A MODULE AND NOT A FEW LINES IN gold.py. Two callers need the same answer: the launcher
preflight, which must refuse a bad tracking config BEFORE sixteen GPUs are allocated and a
multi-day run starts, and gold.py itself, which must set `report_to` before the trainer is
constructed. Two copies of one decision agreeing proves nothing, so the launcher shells out to
this file (`--explain`) rather than reimplementing the rules in bash.

WHAT THIS MODULE DOES NOT DO: decide which PLOT a metric lands in. `ClearMLCallback.on_log`
titles every non-`eval_`/`test_` key "train", which put 11 series of wildly different scale
in one set of axes. `clearml_scalar_groups.py` subclasses the callback to split them by
quantity, and gold.py/sft.py install it after the trainer is built. It is a separate module
because this one runs as a launcher preflight with no trainer in sight.

WHAT EACH BACKEND ACTUALLY READS -- measured against transformers 5.8.0, not assumed:

  ClearMLCallback.setup   os.getenv("CLEARML_PROJECT", "HuggingFace Transformers")
                          os.getenv("CLEARML_TASK",    "Trainer")
  ClearMLCallback.on_save os.getenv("CLEARML_LOG_MODEL", "TRUE")  <- defaults ON upstream; this
                          module defaults it to FALSE, see the note at the env-setting site
                          It IGNORES args.run_name entirely.
  WandbCallback.setup     os.getenv("WANDB_PROJECT",   "huggingface")
                          name  <- args.run_name  (transformers passes it as `name`)
                          entity <- WANDB_ENTITY, read by wandb itself; transformers does not
                          pass entity at all.

So one auto-generated run name has to be plumbed to two DIFFERENT places -- CLEARML_TASK for
ClearML, `training_args.run_name` for W&B -- which is exactly the kind of asymmetry that gets
half-implemented if it is not written down once.

Both callbacks guard their init with `state.is_world_process_zero`, so a 16-rank run creates ONE
task rather than sixteen. That was checked rather than hoped for.

WHY A PARTIAL CONFIG IS REFUSED RATHER THAN FILLED IN. Look at those upstream defaults again: an
incomplete ClearML config does not fail. It logs a multi-day distillation run into a project
called "HuggingFace Transformers" under the name "Trainer", and you find out when you go looking
for the curve. A config that names a project but no entity is an EXPRESSED INTENT to track that
cannot be honoured as written, so it aborts the launch. Nothing configured at all is a different
condition -- that is simply "tracking off", and it is reported, not fatal.

"ENTITY" IS NOT SYMMETRIC BETWEEN THE TWO, and that is a reading of the spec rather than a
detail. W&B's entity is a real, separate coordinate: the team that owns the run, independent of
the credentials. ClearML has no entity -- which workspace receives a task is carried by the
CREDENTIALS (clearml.conf, or CLEARML_API_HOST/ACCESS_KEY/SECRET_KEY), not by the run config. So
ClearML's third coordinate is established by the same check that answers "is a key present",
and inventing a `clearml_entity` field would mean shipping a field that does nothing. FLAGGED
FOR HEW rather than silently decided, in HANDOFF.md.

CREDENTIALS ARE NOT READ, ONLY DETECTED. This module never opens a secret to look at its value;
it establishes presence and reports which source supplied it. A tracking key must not end up in
a log line, and the only way to guarantee that is to never hold it.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

# The config surface. These names are fields on gold.py's `CustomArguments`, and this tuple is
# the single source of truth for them: the launcher's preflight reads the yaml through THIS
# module, so a field added here needs no second edit in bash.
CLEARML_CONFIG_FIELDS = ("clearml_project", "clearml_run_name")
WANDB_CONFIG_FIELDS = ("wandb_entity", "wandb_project", "wandb_run_name")
ALL_CONFIG_FIELDS = CLEARML_CONFIG_FIELDS + WANDB_CONFIG_FIELDS

# What must be present for a backend to be COMPLETE. The run-name fields are deliberately absent:
# they are auto-fillable, which is the whole point of auto_run_name() below.
REQUIRED = {
    "clearml": ("clearml_project",),
    "wandb": ("wandb_entity", "wandb_project"),
}

# ClearML is first because it is the default backend; the order is also the order of
# `report_to`, which is the order the callbacks run in.
BACKENDS = ("clearml", "wandb")


@dataclass
class Plan:
    """The resolved decision. `fatal` non-empty means: do not launch."""

    enabled: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    run_name: str | None = None
    lines: list[str] = field(default_factory=list)
    fatal: list[str] = field(default_factory=list)

    def report(self, emit=print) -> None:
        for ln in self.lines:
            emit(ln)
        for f in self.fatal:
            emit(f"FATAL: {f}")

    def apply(self, training_args, environ: dict | None = None, task_init=None) -> None:
        """Set the env vars and `report_to` the callbacks will read. Called by gold.py only.

        `report_to` is EXTENDED rather than replaced: a config that already asked for
        tensorboard keeps it. transformers 5.8.0 normalises `report_to="none"` to [], so the
        common case starts empty.
        """
        env = os.environ if environ is None else environ
        env.update(self.env)
        current = list(getattr(training_args, "report_to", None) or [])
        for b in self.enabled:
            if b not in current:
                current.append(b)
        training_args.report_to = current
        # W&B takes its display name from run_name; ClearML does not look at it. Setting it also
        # silences transformers' "run_name is the same as output_dir" warning, which is emitted
        # because run_name DEFAULTS to output_dir.
        if self.run_name and "wandb" in self.enabled:
            training_args.run_name = self.run_name
        if "wandb" in self.enabled:
            self._pin_wandb_run_id(training_args, env)
        if "clearml" in self.enabled:
            # `task_init` exists so self_test() can exercise this without a ClearML server: the real
            # one is imported lazily inside. gold.py calls apply() with one argument and gets the
            # real thing.
            self._continue_clearml_task(training_args, env, task_init=task_init)

    @staticmethod
    def _pin_wandb_run_id(training_args, env: dict) -> None:
        """Give W&B a DETERMINISTIC run id keyed on output_dir, so a restart continues one run.

        THE PROBLEM THIS SOLVES, measured rather than assumed. Preemption on this cluster's
        `preemptable` queue is requeue-and-restart-from-argv, not suspend/resume, confirmed
        directly: it printed "preflight OK" three times. transformers' WandbCallback calls `wandb.init()` with
        no run id, and a W&B run is identified by its ID, NOT its name -- names are free to
        collide. So every restart opens a SECOND run, and the loss curve of a preempted job
        arrives as N disjoint fragments that no view stitches back together.

        WHY THE NAME DOES NOT ALREADY HANDLE IT. auto_run_name() appends the LSF job id, and its
        docstring's reasoning is right as far as it goes -- LSF preserves the job id across
        preemption and requeue, so the segments of one preempted run do share a name. But a name
        is not an identity to W&B, so sharing one changes nothing; and the reasoning does not
        cover the other resume path at all, since `bkill` plus resubmit is a NEW job id and
        therefore even a different name.

        WHY output_dir IS THE RIGHT KEY. It is where the checkpoints live, which makes it the
        thing that actually defines run continuity: two invocations that share an output_dir
        resume from each other's checkpoints and ARE one training run, whatever the job id or
        the config filename says. Two that do not share one are separate runs even if every other
        field matches. It is also invariant across exactly the events that break the alternatives
        -- requeue, resubmit, a renamed config -- and it is already absolute in every config here.

        WANDB_RESUME=allow, not "must": `allow` continues the run if that id exists and creates
        it otherwise, so the FIRST launch works unchanged. "must" would make a fresh run fatal.

        setdefault semantics for the same reason the CLEARML_LOG_MODEL line above uses them: an
        operator who exports WANDB_RUN_ID or WANDB_RESUME means it, and apply() does env.update(),
        which would otherwise clobber a deliberate choice.

        THE SAME PROBLEM ON CLEARML is handled by `_continue_clearml_task` below, which had to be
        done differently -- ClearML cannot be steered into continuing a task through the
        environment at all, so it takes a real `Task.init` call rather than two env vars. An
        earlier version of this docstring recorded ClearML as unfixable here; that is no longer
        true, and the reason it looked unfixable is written out in that method.
        """
        out = getattr(training_args, "output_dir", None)
        if not out:
            return
        if not os.path.isabs(str(out)):
            # Every launcher here passes an absolute path ($REPO/data/distillation/checkpoints/...),
            # so this branch is for a hand-set OUTPUT_DIR. Say so rather than pin silently: abspath
            # of a relative path is resolved against the CWD, and gold-train-*.sh does
            # `cd "$GOLD_SRC"` before exec'ing the trainer, so the same relative string typed from
            # the repo root and from the submit script would key two DIFFERENT runs -- the exact
            # failure this function exists to remove.
            print(f"[tracking] WARNING: output_dir {out!r} is relative, so the pinned W&B run id "
                  f"depends on the CWD ({os.getcwd()}). Pass an absolute --output_dir for a run "
                  "id that survives a resubmit.")
        key = os.path.abspath(str(out))
        # 16 hex chars of sha256. Short enough to read in a URL, far past collision concern for
        # the number of output dirs this recipe will ever have, and stable across machines --
        # which `hash()` would not be, since Python salts str hashing per process.
        env.setdefault("WANDB_RUN_ID", hashlib.sha256(key.encode()).hexdigest()[:16])
        env.setdefault("WANDB_RESUME", "allow")
        # And put the local run directory WITH the run. wandb defaults WANDB_DIR to the CWD, and
        # gold-train-*.sh does `cd "$GOLD_SRC"` first, so a direct measurement wrote its run data into
        # src/gb_steps_post_training/distillation/wandb/ -- inside the source tree, under a path
        # that says nothing about which run it belongs to, and picked up by `git status`. Keying it
        # on output_dir puts it beside the checkpoints it describes, which is also what makes
        # `wandb sync` usable after a run that was killed before it could flush.
        env.setdefault("WANDB_DIR", key)

    # The file that carries ClearML run continuity across a restart. Inside output_dir for exactly
    # the reason the W&B run id is keyed on output_dir: that is where the checkpoints live, so it is
    # what actually defines whether two invocations are one training run.
    CLEARML_ID_FILE = ".clearml-task-id"

    @classmethod
    def _continue_clearml_task(cls, training_args, env: dict, task_init=None) -> None:
        """Make a ClearML relaunch APPEND to the existing task instead of erasing it.

        WHY THIS IS WORSE THAN THE W&B CASE, and why it is not the same fix. On W&B a restart opened
        a second run and the curve arrived in fragments -- annoying, but nothing was lost. ClearML
        loses data. `Task.init` defaults to `reuse_last_task_id=True`, and its own docstring is
        explicit about what that means: "When a Task is reused, the previous execution outputs are
        deleted, including console outputs and logs." transformers' ClearMLCallback calls
        `Task.init(project_name=..., task_name=...)` with that default, so the SECOND leg of a
        preempted run deletes the first leg's curve and console log.

        WHAT THAT LINE DOES AND DOES NOT TELL YOU. An earlier version of this docstring said job
        1153643's "ClearML Task: overwriting (reusing) task id=..." line was the deletion
        announcement, and treated its absence as the success criterion. That is wrong, and jobs
        1159990 -> 1160045 measured it: the healthy continuation prints that line too. ClearML
        prints it whenever `reuse_last_task_id` resolves to an existing task, which is exactly what
        the fix below asks it to do; whether the previous outputs survive is decided by
        `continue_last_task`, which the line says nothing about. So do not read it as data loss --
        a reader who does will try to remove it and, in removing it, remove the continuation.

        The criterion that DOES answer the question is the server's own state, which is why
        a companion resume-verification tool exists rather than a grep. Measured directly on a
        real task: after leg 2 resumed from checkpoint-25, `train/loss`
        still held 50 points spanning iterations 1..50 -- 320 scalar points below iteration 26,
        which only leg 1 can have written -- and the console log's earliest line was still leg 1's
        own "created new task" at 14:01:57, thirteen minutes before leg 2 started. Both preserved.

        WHY IT CANNOT BE DONE WITH ENV VARS. The callback reads only CLEARML_PROJECT, CLEARML_TASK
        and CLEARML_LOG_MODEL; nothing it reads reaches `reuse_last_task_id` or
        `continue_last_task`. But it has one door, and it is documented in its own source
        (integration_utils.py, ClearMLCallback.setup): "This might happen when running inside of a
        pipeline, where the task is already initialized from outside of Hugging Face" -- if
        `Task.current_task()` already exists, the callback ADOPTS it instead of creating one. So the
        fix is to be that outside-of-Hugging-Face initializer, in this process, before the trainer
        is constructed. apply() is already called from exactly there (gold.py, before the Trainer).

        THE EXACT CALL, and every argument in it is load-bearing:

          reuse_last_task_id=<task id>  -- resolves the task to continue by ID rather than by
              project+name. Name resolution would not do: auto_run_name() appends the LSF job id,
              so a bkill-plus-resubmit produces a different name, and a renamed config produces a
              different name again, while the run is the same run.

          continue_last_task=0  -- an INT, not True, and the difference matters. `True` calls
              `set_initial_iteration(last_iteration + 1)` (Task._create_dev_task:82), which offsets
              every subsequent report. transformers already reports `iteration=state.global_step`,
              and a resumed run's global_step ALREADY continues (26, 27, ...), so the automatic
              offset would plot step 26 at 51 and the curve would fold over itself. `0` takes the
              same continue branch (the `isinstance(int) and not isinstance(bool)` clause at
              _create_dev_task:74-77) and sets the offset to zero, which is the identity. Reading
              `continue_last_task=0` as "do not continue" is the natural misreading; it is False
              that means that.

          auto_connect_frameworks={"tensorboard": False, "pytorch": False}  -- what the callback
              itself passes. Left at the default, ClearML would hook torch.save and upload
              checkpoints; a 3B student's checkpoints are not going to a logging file server.

          No output_uri  -- the callback passes output_uri=True when IT creates the task, which
              makes the file server the default artifact destination. An externally created task
              also flips the callback's own `_log_model` default from TRUE to FALSE, which agrees
              with the CLEARML_LOG_MODEL=FALSE this module already sets.

        FAILURE POLICY: never fatal, and never fall back to upstream's default. By the time this
        runs the allocation is already held, so a ClearML server hiccup must not end a job that
        waited hours in the queue. But simply returning would hand the task back to the callback
        with `reuse_last_task_id=True` -- the log-deleting path -- so the fallback creates a fresh
        task with `reuse_last_task_id=False` instead. Worst case a restart's curve lands in a new
        task; no case deletes the previous one.

        Rank 0 only. The callback initialises on `state.is_world_process_zero`, so a Task.init on
        every rank of a 16-rank job would create 15 stray tasks. RANK rather than
        PartialState().is_main_process for the reason gold.py already gives at its own rank guard:
        constructing a PartialState initialises the distributed state, and asking a question about
        logging must not move when that happens.
        """
        if int(env.get("RANK", "0") or "0") != 0:
            return
        out = getattr(training_args, "output_dir", None)
        if not out:
            # Nothing to key continuity on. Leaving the callback to its own devices here is the
            # lesser evil: with no output_dir there are no checkpoints, so there is no resume for
            # a task to be continued ACROSS.
            return
        if str(env.get("CLEARML_OFFLINE_MODE", "")).strip().lower() in {"1", "true", "yes"}:
            print("[tracking] clearml offline mode: not attempting task continuation (an offline "
                  "task has no server-side id to continue). Legs of a restarted run will be "
                  "separate offline tasks.")
            return

        out = os.path.abspath(str(out))
        id_file = os.path.join(out, cls.CLEARML_ID_FILE)
        prior = None
        try:
            if os.path.isfile(id_file):
                prior = (open(id_file).read().strip() or None)
        except OSError as exc:
            print(f"[tracking] WARNING: cannot read {id_file}: {exc}. Treating this as a new run.")

        if task_init is None:
            try:
                from clearml import Task
            except Exception as exc:  # pragma: no cover - resolve() already required it
                print(f"[tracking] WARNING: clearml import failed at Task.init time ({exc}). "
                      "Leaving the task to transformers' callback.")
                return
            task_init = Task.init

        kwargs = dict(
            project_name=env.get("CLEARML_PROJECT"),
            task_name=env.get("CLEARML_TASK"),
            auto_connect_frameworks={"tensorboard": False, "pytorch": False},
        )
        task = None
        if prior:
            try:
                task = task_init(reuse_last_task_id=prior, continue_last_task=0, **kwargs)
                print(f"[tracking] clearml continuing task {prior} from {cls.CLEARML_ID_FILE} "
                      "(previous console output and scalars kept, no iteration offset)")
            except Exception as exc:
                print(f"[tracking] WARNING: could not continue clearml task {prior} ({exc}). "
                      "Creating a NEW task rather than letting the callback reuse-and-delete one.")
                task = None
        if task is None:
            try:
                task = task_init(reuse_last_task_id=False, **kwargs)
            except Exception as exc:
                print(f"[tracking] WARNING: clearml Task.init failed ({exc}). Continuing without "
                      "tracking rather than failing a job that already holds its allocation.")
                return
            print(f"[tracking] clearml new task {getattr(task, 'id', '?')} "
                  f"(reuse_last_task_id=False, so no previous task's logs were deleted)")

        # Record the id LAST and only when it changed, so the file always names a task that exists.
        new_id = getattr(task, "id", None)
        if new_id and new_id != prior:
            try:
                os.makedirs(out, exist_ok=True)
                tmp = id_file + ".tmp"
                with open(tmp, "w") as fh:
                    fh.write(str(new_id) + "\n")
                os.replace(tmp, id_file)
            except OSError as exc:
                print(f"[tracking] WARNING: could not record the clearml task id in {id_file} "
                      f"({exc}). A restart of this output_dir will start a new task instead of "
                      "continuing this one.")


def _clean(value) -> str | None:
    """A YAML key present but empty means "not set". step-template.yaml renders every unset
    string key as "", so treating "" as a value would make every granite.build run look
    half-configured and abort."""
    if value is None:
        return None
    s = str(value).strip()
    return s or None


# The KEY NAME IS ITSELF QUOTED in a real clearml.conf -- the file the ClearML UI hands you reads
#
#     api {
#       credentials {
#         "access_key" = "..."
#         "secret_key" = "..."
#       }
#     }
#
# so a pattern anchored as `access_key\s*[:=]` never matches: a `"` sits between the name and the
# `=`. The first version of this check was written against the DOCUMENTED `access_key: value` form,
# passed its own fixtures, and then reported hew's real credentials as absent. Hence `"?` on both
# sides of the name, and hence the fixture in case 13 is now a verbatim copy of the real layout
# rather than a plausible-looking one.
#
# `[^"\s]` and not `\S` for the VALUE, for a different reason: with `\S`, `access_key = ""`
# matches, because the optional quote eats the opening `"` and `\S` then matches the closing one.
# A default clearml.conf skeleton carries exactly that empty pair.
_KEY_RE = r'"?%s"?\s*[:=]\s*"?[^"\s]'


def clearml_credentials(environ: Mapping[str, str], home: Path) -> tuple[bool, str]:
    """(present, where). Presence only -- the value is never read or returned."""
    if environ.get("CLEARML_API_ACCESS_KEY") and environ.get("CLEARML_API_SECRET_KEY"):
        return True, "CLEARML_API_ACCESS_KEY/SECRET_KEY in the environment"
    conf = environ.get("CLEARML_CONFIG_FILE")
    candidates = [Path(conf)] if conf else [home / "clearml.conf", home / ".clearml.conf"]
    for c in candidates:
        try:
            text = c.read_text(errors="replace")
        except OSError:
            continue
        # Presence of a non-empty access_key/secret_key pair. Deliberately crude: this is a
        # HOCON file and parsing it properly would mean holding the secret.
        if re.search(_KEY_RE % "access_key", text) and re.search(_KEY_RE % "secret_key", text):
            return True, f"{c}"
    return False, "no CLEARML_API_* env vars and no clearml.conf with a key pair"


def wandb_credentials(environ: Mapping[str, str], home: Path) -> tuple[bool, str]:
    if environ.get("WANDB_API_KEY"):
        return True, "WANDB_API_KEY in the environment"
    netrc = Path(environ["NETRC"]) if environ.get("NETRC") else home / ".netrc"
    try:
        text = netrc.read_text(errors="replace")
    except OSError:
        text = ""
    # `machine api.wandb.ai` followed by a password somewhere in the same entry.
    if re.search(r"machine\s+api\.wandb\.ai", text) and re.search(r'password\s+"?[^"\s]', text):
        return True, f"{netrc} (machine api.wandb.ai)"
    return False, "no WANDB_API_KEY and no ~/.netrc entry for api.wandb.ai"


def is_offline(backend: str, environ: Mapping[str, str]) -> bool:
    """Offline mode needs no credentials, so it SATISFIES the key requirement.

    No new config field for this: both libraries already have a standard env var, and a gold
    yaml can set either through TrlParser's native `env:` block. Recognising them here is what
    keeps "offline" from being reported as "no credentials, tracking off".
    """
    if backend == "clearml":
        return str(environ.get("CLEARML_OFFLINE_MODE", "")).strip().lower() in {"1", "true", "yes"}
    return str(environ.get("WANDB_MODE", "")).strip().lower() in {"offline", "dryrun"}


def _short(path_like: str | None) -> str | None:
    """Last path component, so a run name is not 120 characters of absolute path."""
    if not path_like:
        return None
    return Path(str(path_like)).name or None


GENERIC_STEMS = {"gold-config.rendered", "gold-config", "config", "gold", "rendered",
                 "sft-config.rendered", "sft-config", "sft"}


def auto_run_name(config_path: str | None, hints: Mapping[str, object], environ: Mapping[str, str]) -> str:
    """A run name derived from the build yaml, for when the config does not give one.

    TWO SOURCES, in order, because the two launch paths differ in what is meaningful:

      1. The config FILE STEM. In the reference training environment the gold yamls are
         already named
         `granite-4.1-3b-base_from-4.2-30b_en-sft-4.1-0.2-16K_onpolicy_node2` -- student,
         teacher, corpus, arm and topology, which is a better run name than anything this
         function could compose. Use it.
      2. COMPOSED from the semantic keys, when the stem carries no information. Under
         granite.build the config is RENDERED by run-gold.sh to `gold-config.rendered.yaml`,
         so every run would otherwise be called the same thing -- which is why step 2 exists
         and is not defensive padding.

    Then the LSF job id is appended when there is one. A timestamp was the obvious alternative
    and is worse here: LSF keeps the job id across preemption and requeue, and this recipe runs
    on preemptable capacity where that happens routinely. Job id therefore groups the segments
    of one preempted run under one name, where a timestamp would scatter them.
    """
    stem = Path(config_path).stem if config_path else None
    if stem and stem.lower() not in GENERIC_STEMS:
        base = stem
    else:
        student = _short(hints.get("model_name_or_path")) or "student"
        teacher = _short(hints.get("teacher_model_name_or_path"))
        corpus = _short(hints.get("dataset_name")) or "corpus"
        corpus = re.sub(r"\.(jsonl|json|parquet)$", "", corpus)
        lmbda = hints.get("lmbda")
        if teacher is None and lmbda is None:
            # NEITHER a teacher NOR an lmbda: a plain SFT run -- distill-sft-baseline's control,
            # whose config has no such keys because there is no second model and no mixing
            # coefficient. Falling through to the branch below would name it
            # `student_from-teacher_corpus_offpolicy`, which asserts two things that are not
            # true (a teacher exists; an on/off-policy choice was made) about the one run whose
            # entire value is being the arm that did NOT distill. A control mislabelled as a
            # distillation arm is worse than an ugly name.
            #
            # Note this is deliberately NOT reached by the KD variant of the same step: setting
            # `precomputed_logits_dir` makes sft.py pass that directory as the teacher hint, so a
            # forward-KL run composes through the branch below and reads as a distillation run.
            base = f"{student}_{corpus}_sft"
        else:
            arm = "onpolicy" if (lmbda is not None and float(lmbda) > 0) else "offpolicy"
            base = f"{student}_from-{teacher or 'teacher'}_{corpus}_{arm}"
            if lmbda is not None:
                base += f"_lmbda{lmbda}"
    job = _clean(environ.get("LSB_JOBID"))
    return f"{base}-lsf{job}" if job else base


def resolve(
    fields: Mapping[str, object],
    hints: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    config_path: str | None = None,
    is_installed=None,
) -> Plan:
    """Decide, for each backend, on/off/fatal -- and say why in every case.

    `fields` are the CustomArguments tracking values (or the raw yaml keys, for the preflight).
    `hints` are the semantic config keys used only to compose a fallback run name.
    `is_installed` is injectable so the self-test can exercise the ENABLED path in an
    interpreter where neither package is installed -- which is the state of this venv today.
    Without that seam every happy-path case would abort on the package check and the on-path
    would never be tested at all.
    """
    environ = os.environ if environ is None else environ
    home = Path(os.path.expanduser("~")) if home is None else home
    hints = hints or {}

    plan = Plan()
    vals = {k: _clean(fields.get(k)) for k in ALL_CONFIG_FIELDS}
    generated = auto_run_name(config_path, hints, environ)
    enabled: list[str] = []

    for backend in BACKENDS:
        own = CLEARML_CONFIG_FIELDS if backend == "clearml" else WANDB_CONFIG_FIELDS
        set_here = {k: v for k, v in vals.items() if k in own and v}
        role = "default" if backend == "clearml" else "optional"

        if not set_here:
            plan.lines.append(f"tracking : {backend:8s} off      ({role}; nothing configured)")
            continue

        missing = [k for k in REQUIRED[backend] if not vals.get(k)]
        if missing:
            plan.fatal.append(
                f"{backend} is partially configured: {', '.join(sorted(set_here))} given, but "
                f"{', '.join(missing)} missing. A partial tracking config is an expressed intent "
                f"to track that cannot be honoured -- and {backend} would not fail, it would log "
                f"to its own default project. Complete it or remove it."
            )
            continue

        offline = is_offline(backend, environ)
        if offline:
            keyed, where = True, "offline mode -- no credentials needed"
        elif backend == "clearml":
            keyed, where = clearml_credentials(environ, home)
        else:
            keyed, where = wandb_credentials(environ, home)

        run_name = vals.get(f"{backend}_run_name") or generated
        origin = "from config" if vals.get(f"{backend}_run_name") else "auto-generated"

        if not keyed:
            # Configured completely, but no key. Per the rules this is OFF, not fatal -- and it
            # is printed at full volume, because "I configured tracking and got none" is the
            # exact surprise this line exists to prevent.
            plan.lines.append(
                f"tracking : {backend:8s} OFF      config is complete but NO CREDENTIALS ({where}). "
                f"Not tracking. Would have been: {run_name}"
            )
            continue

        enabled.append(backend)
        if backend == "clearml":
            plan.env["CLEARML_PROJECT"] = vals["clearml_project"]
            plan.env["CLEARML_TASK"] = run_name
            # CHECKPOINT UPLOAD OFF BY DEFAULT, and this is not a preference.
            #
            # transformers' ClearMLCallback hardcodes `output_uri=True` and then sets
            # `_log_model = os.getenv("CLEARML_LOG_MODEL", "TRUE")` -- note the default -- for a
            # task it created itself (integration_utils.py:1914-1922). `on_save` therefore zips
            # the ENTIRE checkpoint directory and uploads it, synchronously, inside training
            # ("Logging checkpoint artifact ... This may take some time"). Under ZeRO-3 that
            # directory holds sharded optimizer state, so it is tens of GB per save, and every
            # gold config sets save_steps. Nobody asked for that, and on a preemptable
            # allocation it is wall-clock spent not training.
            #
            # Where it would go, since that was worth checking rather than assuming:
            # ~/clearml.conf declares no `files_server`, and clearml composes one by appending
            # :8081 to the app host (session.py:774-806) -- the self-hosted ClearML server's
            # own host. The public files.clear.ml default applies only when api_server is
            # clearml's own, which it is not here. So this is not weights leaving the
            # organization; it is a large synchronous upload to a port whose reachability is
            # a separate question.
            #
            # setdefault semantics, not policy: an operator who exports CLEARML_LOG_MODEL means
            # it, and Plan.apply() does env.update(), which would otherwise clobber the
            # deliberate choice. Turning this on for real wants a config key rather than an
            # export -- deliberately not added until somebody wants the behaviour.
            if "CLEARML_LOG_MODEL" not in environ:
                plan.env["CLEARML_LOG_MODEL"] = "FALSE"
            plan.lines.append(
                f"tracking : clearml  ON       project={vals['clearml_project']} "
                f"task={run_name} ({origin}); key: {where}"
            )
        else:
            plan.env["WANDB_ENTITY"] = vals["wandb_entity"]
            plan.env["WANDB_PROJECT"] = vals["wandb_project"]
            plan.run_name = run_name
            plan.lines.append(
                f"tracking : wandb    ON       entity={vals['wandb_entity']} "
                f"project={vals['wandb_project']} name={run_name} ({origin}); key: {where}"
            )

    plan.enabled = tuple(enabled)

    # An enabled backend whose package is absent is fatal, and deliberately so. transformers
    # would raise from inside the callback anyway; catching it here means the launch dies in
    # the preflight instead of after the allocation. It is NOT the same condition as "no key":
    # a missing key is an unconfigured environment, a missing package is a broken one.
    if is_installed is None:
        def is_installed(name: str) -> bool:
            import importlib.util

            return importlib.util.find_spec(name) is not None

    for b in list(plan.enabled):
        if not is_installed(b):
            plan.fatal.append(
                f"{b} is configured, credentialed and enabled, but the `{b}` package is not "
                f"installed in this interpreter. Training would proceed UNTRACKED, which is the "
                f"failure this config exists to prevent. Install it or remove the {b}_* keys."
            )
    return plan


# --------------------------------------------------------------------------------------------
# Self-test and CLI.
#
# The CLI is what the launcher calls in its preflight, so that bash never re-decides anything.


def load_yaml_fields(path: Path) -> tuple[dict, dict]:
    """(tracking fields, run-name hints) read straight from a gold yaml.

    NOT via TrlParser, and the difference is worth stating: TrlParser also applies command-line
    overrides, so this preflight sees the FILE and not necessarily the final parsed values. That
    is the right trade for a preflight (it runs before the trainer exists at all), and gold.py
    re-resolves from the parsed dataclass anyway, which is the authoritative pass. If the two
    ever disagree, gold.py wins -- and it prints its own report, so the disagreement is visible.
    """
    import yaml

    doc = yaml.safe_load(path.read_text()) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"{path} is not a YAML mapping")
    fields = {k: doc.get(k) for k in ALL_CONFIG_FIELDS}
    hints = {k: doc.get(k) for k in ("model_name_or_path", "teacher_model_name_or_path", "dataset_name", "lmbda")}
    return fields, hints


def self_test() -> int:
    import tempfile

    bad = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal bad
        if cond:
            print(f"  OK   {name}")
        else:
            print(f"  FAIL {name}{': ' + detail if detail else ''}")
            bad = 1

    yes = lambda _n: True  # both packages "installed", so the ON path is reachable here
    no = lambda _n: False

    # 1. Nothing configured. The default-off case, which must be silent-but-reported, not fatal.
    p = resolve({}, environ={}, home=Path("/nonexistent"), is_installed=yes)
    check("nothing configured -> both off, no fatal",
          p.enabled == () and not p.fatal and len(p.lines) == 2, f"{p.enabled} {p.fatal}")

    # 2. ClearML complete, key in env. The default backend on its happy path.
    env = {"CLEARML_API_ACCESS_KEY": "x", "CLEARML_API_SECRET_KEY": "y"}
    p = resolve({"clearml_project": "distill"}, environ=env, home=Path("/nonexistent"),
                config_path="/c/granite-4.1-3b_onpolicy_node2.yaml", is_installed=yes)
    check("clearml complete + key -> ON, env set, name from stem",
          p.enabled == ("clearml",)
          and p.env["CLEARML_PROJECT"] == "distill"
          and p.env["CLEARML_TASK"] == "granite-4.1-3b_onpolicy_node2"
          and not p.fatal, f"{p.enabled} {p.env} {p.fatal}")

    # 3. Partial: a run name but no project. Must ABORT, because ClearML would silently use its
    #    own default project instead of failing.
    p = resolve({"clearml_run_name": "mine"}, environ=env, home=Path("/nonexistent"), is_installed=yes)
    check("clearml run_name without project -> FATAL",
          not p.enabled and any("partially configured" in f and "clearml_project" in f for f in p.fatal),
          f"{p.fatal}")

    # 4. Partial the other way round, on the backend where entity is real.
    p = resolve({"wandb_entity": "ibm"}, environ={"WANDB_API_KEY": "k"}, home=Path("/nonexistent"),
                is_installed=yes)
    check("wandb entity without project -> FATAL",
          not p.enabled and any("wandb_project" in f for f in p.fatal), f"{p.fatal}")

    # 5. W&B complete. Note run_name lands on the PLAN (for training_args.run_name), not in env --
    #    the asymmetry the docstring describes.
    p = resolve({"wandb_entity": "ibm", "wandb_project": "distill", "wandb_run_name": "r1"},
                environ={"WANDB_API_KEY": "k"}, home=Path("/nonexistent"), is_installed=yes)
    check("wandb complete -> ON, entity+project in env, name on the plan",
          p.enabled == ("wandb",) and p.env["WANDB_ENTITY"] == "ibm"
          and p.env["WANDB_PROJECT"] == "distill" and p.run_name == "r1"
          and "WANDB_NAME" not in p.env, f"{p.enabled} {p.env} {p.run_name}")

    # 6. Complete but unkeyed -> off, NOT fatal, and loud. This is the specified behaviour and
    #    the one most likely to be quietly mis-implemented as either fatal or silent.
    p = resolve({"clearml_project": "distill"}, environ={}, home=Path("/nonexistent"), is_installed=yes)
    check("complete but no key -> off, not fatal, says NO CREDENTIALS",
          not p.enabled and not p.fatal and any("NO CREDENTIALS" in l for l in p.lines), f"{p.lines}")

    # 7. Offline mode needs no key, so it must NOT be reported as unkeyed.
    p = resolve({"clearml_project": "d"}, environ={"CLEARML_OFFLINE_MODE": "1"},
                home=Path("/nonexistent"), is_installed=yes)
    check("CLEARML_OFFLINE_MODE=1 satisfies the key requirement",
          p.enabled == ("clearml",) and any("offline mode" in l for l in p.lines), f"{p.lines}")
    p = resolve({"wandb_entity": "e", "wandb_project": "p"}, environ={"WANDB_MODE": "offline"},
                home=Path("/nonexistent"), is_installed=yes)
    check("WANDB_MODE=offline satisfies the key requirement", p.enabled == ("wandb",), f"{p.lines}")

    # 8. A name in the config beats the generated one, and says so.
    p = resolve({"clearml_project": "d", "clearml_run_name": "explicit"}, environ=env,
                home=Path("/nonexistent"), config_path="/c/some_config.yaml", is_installed=yes)
    check("config run name wins over auto",
          p.env["CLEARML_TASK"] == "explicit" and any("from config" in l for l in p.lines), f"{p.lines}")

    # 9. The granite.build path: the rendered stem carries nothing, so compose from the config.
    name = auto_run_name("/out/gold-config.rendered.yaml", {
        "model_name_or_path": "/m/granite-4.1-3b-base_retagged_v2",
        "teacher_model_name_or_path": "/hub/snapshots/a0057a78",
        "dataset_name": "/d/subsampled_0.2_shuffled.jsonl",
        "lmbda": 0.3,
    }, {})
    check("generic rendered stem -> composed semantic name",
          name == "granite-4.1-3b-base_retagged_v2_from-a0057a78_subsampled_0.2_shuffled_onpolicy_lmbda0.3",
          name)

    # 10. lmbda 0.0 is the off-policy arm and must not be called onpolicy -- the arm is the single
    #     most consequential key in this recipe, so a run name that lies about it is a real cost.
    off = auto_run_name("/out/gold-config.rendered.yaml", {"lmbda": 0.0}, {})
    check("lmbda 0.0 -> offpolicy in the composed name", "offpolicy" in off and "onpolicy" not in off, off)

    # 10b. ...and a run with NEITHER a teacher NOR an lmbda is a plain SFT control, which must
    #      not be named as if it had either. Sits beside 10 because both are the same class of
    #      bug: a run name that misreports which arm produced the numbers. distill-sft-baseline
    #      renders to `sft-config.rendered.yaml`, a generic stem, so this branch is what its
    #      auto-generated names actually come from.
    sft = auto_run_name("/out/sft-config.rendered.yaml", {
        "model_name_or_path": "/m/granite-4.1-3b-base_retagged_v2",
        "dataset_name": "/d/subsampled_0.2_shuffled.jsonl",
    }, {})
    check("no teacher and no lmbda -> plain sft name",
          sft == "granite-4.1-3b-base_retagged_v2_subsampled_0.2_shuffled_sft", sft)
    #      The KD variant of that same step DOES have a teacher (the precompute directory), and
    #      must read as a distillation run rather than as the control.
    kd = auto_run_name("/out/sft-config.rendered.yaml", {
        "model_name_or_path": "/m/student",
        "teacher_model_name_or_path": "/d/logits_granite-4.2-30b_top256",
        "dataset_name": "/d/corpus.jsonl",
    }, {})
    check("teacher present, lmbda absent -> named as a distillation run",
          kd == "student_from-logits_granite-4.2-30b_top256_corpus_offpolicy", kd)

    # 11. The LSF job id is appended, because it survives preemption and requeue where a
    #     timestamp would split one run into several.
    check("LSB_JOBID is appended",
          auto_run_name("/c/cfg_node2.yaml", {}, {"LSB_JOBID": "1152213"}) == "cfg_node2-lsf1152213")

    # 12. Enabled + package absent -> FATAL. Distinct from the no-key case above: this one refuses
    #     to train, because the config is fully honoured except by the interpreter.
    p = resolve({"clearml_project": "d"}, environ=env, home=Path("/nonexistent"), is_installed=no)
    check("enabled but package missing -> FATAL",
          any("not installed" in f for f in p.fatal), f"{p.fatal}")

    # 13. Credential detection off disk, both backends, in a real temp HOME.
    with tempfile.TemporaryDirectory() as td:
        h = Path(td)
        check("no creds on a bare home", not clearml_credentials({}, h)[0] and not wandb_credentials({}, h)[0])
        (h / "clearml.conf").write_text('api {\n  access_key: "AK123"\n  secret_key: "SK456"\n}\n')
        ok, where = clearml_credentials({}, h)
        check("clearml.conf, documented `key: value` form, is detected", ok and "clearml.conf" in where, where)
        # THE REAL LAYOUT, verbatim from the file the ClearML UI generates -- quoted key names and
        # `=`. The first version of this detector handled only the form above, and reported a real
        # credential file as absent. A fixture that is a copy of the real thing is the only kind
        # that would have caught it.
        (h / "clearml.conf").write_text(
            'api {\n'
            '  web_server: https://clearml-ext.example.com\n'
            '  api_server: https://clearml-ext.example.com:8008\n'
            '\n'
            '  credentials {\n'
            '    "access_key" = "AK123456"\n'
            '    "secret_key" = "SK789012"\n'
            '  }\n'
            '}\n')
        ok, where = clearml_credentials({}, h)
        check("clearml.conf, REAL quoted-name `\"access_key\" = ...` form, is detected",
              ok and "clearml.conf" in where, where)
        # A conf that exists but carries no key must NOT read as credentialed -- otherwise a
        # default clearml.conf skeleton would enable tracking that then fails at Task.init.
        (h / "clearml.conf").write_text('api {\n  access_key: ""\n  secret_key: ""\n}\n')
        check("clearml.conf with EMPTY keys is not credentials", not clearml_credentials({}, h)[0])
        (h / ".netrc").write_text("machine api.wandb.ai\n  login user\n  password abc123\n")
        ok, where = wandb_credentials({}, h)
        check("netrc entry for api.wandb.ai is detected", ok and ".netrc" in where, where)
        (h / ".netrc").write_text("machine example.com\n  login user\n  password abc123\n")
        check("netrc for a DIFFERENT machine is not wandb credentials", not wandb_credentials({}, h)[0])

    # 14. "" is not a value. step-template.yaml renders every unset string key as "", so if this
    #     were wrong every granite.build run would abort as half-configured.
    p = resolve({"clearml_project": "", "clearml_run_name": "  "}, environ=env,
                home=Path("/nonexistent"), is_installed=yes)
    check('empty-string yaml keys count as unset', p.enabled == () and not p.fatal, f"{p.enabled} {p.fatal}")

    # 15. apply(): report_to is EXTENDED, not replaced, and run_name is only touched for wandb.
    class FakeArgs:
        report_to = ["tensorboard"]
        run_name = "/some/output_dir"

    a = FakeArgs()
    env2: dict[str, str] = {}
    resolve({"wandb_entity": "e", "wandb_project": "p"}, environ={"WANDB_API_KEY": "k"},
            home=Path("/nonexistent"), config_path="/c/cfg.yaml", is_installed=yes).apply(a, env2)
    check("apply extends report_to and sets run_name for wandb",
          a.report_to == ["tensorboard", "wandb"] and a.run_name == "cfg" and env2["WANDB_ENTITY"] == "e",
          f"{a.report_to} {a.run_name} {env2}")

    b = FakeArgs()
    resolve({"clearml_project": "d"}, environ=env, home=Path("/nonexistent"),
            config_path="/c/cfg.yaml", is_installed=yes).apply(b, {})
    check("apply does NOT touch run_name for clearml (it reads CLEARML_TASK)",
          b.report_to == ["tensorboard", "clearml"] and b.run_name == "/some/output_dir",
          f"{b.report_to} {b.run_name}")

    # 17. The wandb run id is pinned to output_dir, and only to output_dir.
    #     This is the one case where the assertion IS the design argument: the fix is worthless
    #     unless the id is stable across the thing that actually varies (the job id) and unstable
    #     across the thing that actually distinguishes runs (the output directory). Both are
    #     checked, because pinning to a CONSTANT would pass a stability test alone and would
    #     merge every run in the project into one curve.
    class ArgsWithOut:
        report_to = ["tensorboard"]
        run_name = "x"

        def __init__(self, out):
            self.output_dir = out

    # A stand-in for clearml.Task.init. Every pin() call gets one, including the wandb-only ones:
    # apply() dispatches on `enabled`, and a helper that only injects it "when needed" would reach
    # the real Task.init -- and therefore a real ClearML server -- the day a case adds a
    # clearml_project. `boom` makes the failure path reachable without breaking anything.
    class FakeTask:
        def __init__(self, tid):
            self.id = tid

    class Recorder:
        def __init__(self, tid="task-aaa", boom=False):
            self.calls: list[dict] = []
            self.tid, self.boom = tid, boom

        def __call__(self, **kw):
            self.calls.append(kw)
            if self.boom and "reuse_last_task_id" in kw and isinstance(kw["reuse_last_task_id"], str):
                raise RuntimeError("simulated: server rejected the task id")
            return FakeTask(self.tid)

    def pin(out, environ=None, cfg=None, rec=None):
        # `environ` feeds BOTH resolve() and apply(), which it did not originally. That mattered:
        # resolve() decides whether a backend is enabled at all, and it was being handed only
        # WANDB_API_KEY -- so the pre-existing clearml-only case below had no ClearML credentials,
        # resolve() marked it unkeyed, apply() enabled nothing, and the case passed by asserting
        # something about a plan that did nothing. Merging is also what a real run looks like: one
        # environment, read by both.
        e: dict[str, str] = dict(environ or {})
        resolve(cfg or {"wandb_entity": "e", "wandb_project": "p"},
                environ={"WANDB_API_KEY": "k", **(environ or {})}, home=Path("/nonexistent"),
                config_path="/c/cfg.yaml", is_installed=yes).apply(
                    ArgsWithOut(out), e, task_init=rec or Recorder())
        return e

    j1 = pin("/runs/mine", {"LSB_JOBID": "111"})
    j2 = pin("/runs/mine", {"LSB_JOBID": "222"})
    check("wandb run id is the same for one output_dir under two different LSF job ids",
          j1.get("WANDB_RUN_ID") and j1["WANDB_RUN_ID"] == j2["WANDB_RUN_ID"],
          f"{j1.get('WANDB_RUN_ID')} vs {j2.get('WANDB_RUN_ID')}")
    check("wandb resume policy is allow, so the FIRST launch is not an error",
          j1.get("WANDB_RESUME") == "allow", f"{j1.get('WANDB_RESUME')}")
    # A trailing slash and a relative path must land on the same run as the absolute form,
    # because a resubmit typed by hand is exactly where those differ.
    check("wandb run id is path-normalised, not string-keyed",
          pin("/runs/mine/").get("WANDB_RUN_ID") == j1["WANDB_RUN_ID"],
          f"{pin('/runs/mine/').get('WANDB_RUN_ID')} vs {j1['WANDB_RUN_ID']}")
    check("a DIFFERENT output_dir gets a different run id",
          pin("/runs/other").get("WANDB_RUN_ID") != j1["WANDB_RUN_ID"])
    # setdefault, for the reason CLEARML_LOG_MODEL uses it: an operator who set the id by hand
    # meant it, and this code must not be able to overrule them.
    check("a caller-set WANDB_RUN_ID survives",
          pin("/runs/mine", {"WANDB_RUN_ID": "hand-picked"})["WANDB_RUN_ID"] == "hand-picked")
    check("no output_dir -> no run id invented",
          "WANDB_RUN_ID" not in env2, f"{env2}")
    clearml_env = pin("/runs/mine", {"CLEARML_API_ACCESS_KEY": "a", "CLEARML_API_SECRET_KEY": "b"},
                      cfg={"clearml_project": "d"})
    check("clearml-only run does not get a WANDB_RUN_ID",
          "WANDB_RUN_ID" not in clearml_env, f"{clearml_env}")

    # 18. ClearML task continuation. Nine cases, and the reason there are nine rather than two is
    #     that every one of them corresponds to a way this can silently do the WRONG thing --
    #     delete a previous leg's logs, fold a resumed curve over itself, or create one task per
    #     rank. The kwargs are asserted individually, not as a blob, because `continue_last_task=0`
    #     versus `False` and `reuse_last_task_id=<id>` versus `True` are single-token differences
    #     with opposite consequences.
    cl_cfg = {"clearml_project": "d"}
    cl_env = {"CLEARML_API_ACCESS_KEY": "a", "CLEARML_API_SECRET_KEY": "b"}
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ckpt-dir"
        rec = Recorder("task-first")
        pin(str(out), cl_env, cfg=cl_cfg, rec=rec)
        first = rec.calls[-1]
        check("first launch creates a task with reuse_last_task_id=False (upstream's True DELETES "
              "the previous execution's logs)",
              len(rec.calls) == 1 and first.get("reuse_last_task_id") is False
              and "continue_last_task" not in first, f"{rec.calls}")
        check("first launch records the task id inside output_dir",
              (out / Plan.CLEARML_ID_FILE).read_text().strip() == "task-first",
              f"{list(out.iterdir())}")
        check("the task carries the project and name the callback would have used",
              first.get("project_name") == "d" and first.get("task_name"),
              f"{first}")
        check("frameworks are NOT auto-connected, so clearml does not upload checkpoints",
              first.get("auto_connect_frameworks") == {"tensorboard": False, "pytorch": False},
              f"{first}")
        check("output_uri is left unset, so models do not default to the file server",
              "output_uri" not in first, f"{first}")

        rec2 = Recorder("task-first")
        pin(str(out), cl_env, cfg=cl_cfg, rec=rec2)
        second = rec2.calls[-1]
        check("a RELAUNCH into the same output_dir continues that task by id",
              len(rec2.calls) == 1 and second.get("reuse_last_task_id") == "task-first",
              f"{rec2.calls}")
        check("the relaunch passes continue_last_task=0 -- the int, which continues with NO "
              "iteration offset (True would replot resumed step 26 at 51)",
              second.get("continue_last_task") == 0
              and second["continue_last_task"] is not False, f"{second}")

        # The failure path. What must NOT happen is a fall back to the callback's default, which
        # would reuse-and-delete; a fresh task loses continuity but never data.
        rec3 = Recorder("task-second", boom=True)
        pin(str(out), cl_env, cfg=cl_cfg, rec=rec3)
        check("if continuation is refused, a NEW task is created with reuse_last_task_id=False "
              "rather than deleting the old one",
              len(rec3.calls) == 2 and rec3.calls[1].get("reuse_last_task_id") is False,
              f"{rec3.calls}")
        check("and the recorded id moves to the task that now exists",
              (out / Plan.CLEARML_ID_FILE).read_text().strip() == "task-second")

        rec4 = Recorder()
        pin(str(out), {**cl_env, "RANK": "3"}, cfg=cl_cfg, rec=rec4)
        check("a non-zero RANK initialises nothing, so a 16-rank job makes 1 task and not 16",
              rec4.calls == [], f"{rec4.calls}")

        rec5 = Recorder()
        pin(str(out), {**cl_env, "CLEARML_OFFLINE_MODE": "1"}, cfg=cl_cfg, rec=rec5)
        check("offline mode does not try to continue a task that has no server-side id",
              rec5.calls == [], f"{rec5.calls}")

        rec6 = Recorder()
        e6 = dict(cl_env)
        resolve(cl_cfg, environ={"WANDB_API_KEY": "k"}, home=Path("/nonexistent"),
                config_path="/c/cfg.yaml", is_installed=yes).apply(
                    FakeArgs(), e6, task_init=rec6)
        check("no output_dir -> no task pinned, because there is no resume to continue across",
              rec6.calls == [], f"{rec6.calls}")

    # 16. Both at once. ClearML first, because it is the default.
    p = resolve({"clearml_project": "d", "wandb_entity": "e", "wandb_project": "p"},
                environ={**env, "WANDB_API_KEY": "k"}, home=Path("/nonexistent"),
                config_path="/c/cfg.yaml", is_installed=yes)
    check("both backends -> clearml first in report_to", p.enabled == ("clearml", "wandb"), f"{p.enabled}")

    if not bad:
        print("\nTRACKING RESOLUTION EXERCISED: on, off, partial-fatal, unkeyed, offline, "
              "auto-name (both sources), credential detection on disk, apply(), the "
              "output_dir-pinned W&B run id, and output_dir-keyed ClearML task continuation")
    return bad


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()

    if "--config" not in argv:
        print("usage: tracking.py --config <gold.yaml> [--explain] | --self-test")
        return 2
    cfg = Path(argv[argv.index("--config") + 1])
    fields, hints = load_yaml_fields(cfg)
    plan = resolve(fields, hints, config_path=str(cfg))
    plan.report()
    if plan.fatal:
        return 1
    # Exit 0 whether or not anything is enabled: "tracking off" is a legitimate configuration and
    # must not block a launch. Only an unhonourable config does.
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
