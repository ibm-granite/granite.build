# SPDX-License-Identifier: Apache-2.0

"""ClearML / W&B experiment tracking: ONE place that decides whether a backend is on, and why.

THE RULES THIS ENCODES (as specified for this project):
  * ONE tool per run, chosen by `tracking_tool`: "wandb", "clearml", or empty / "none" (off).
    Off is the default. The two tools are never on together.
  * Setting another tool's keys, or any tool's keys without choosing a tool, is refused: it is
    an expressed intent to track that the config does not say how to honour.
  * The chosen tool is enabled only with a COMPLETE config -- project, and for W&B also entity.
  * Each tool talks to its PUBLIC server unless the user names another. W&B: this module sets
    WANDB_BASE_URL=https://api.wandb.ai when neither the environment nor wandb's settings file
    names a server (wandb >= 0.31's own default is https://api.forge.coreweave.com). ClearML:
    this module sets NOTHING. clearml resolves its server from its environment and config file,
    and falls back to https://api.clear.ml (app.clear.ml, files.clear.ml) on its own
    (clearml/backend_api/session/session.py, Session.default_host/_web/_files). Setting a
    ClearML server here would override a self-hosted one the config file names: clearml lets the
    environment win over its config, so the file's credentials would then go to api.clear.ml.
  * No credentials present -> the tool is not used. They are looked up the way the tool itself
    looks them up, against the server it will talk to: for ClearML by clearml's own config
    loader, run in a subprocess (clearml_probe()).
  * The chosen tool is the only remote tracker the Trainer reports to: apply() drops any other
    integration a config put in `report_to` (tensorboard, which writes local files, stays).
  * The run name may be omitted, and is then auto-generated from the build yaml.

WHERE IT RUNS. gold.py and sft.py call resolve() and Plan.apply() after parsing their config
and before the trainer is built, on every rank; that is the only check, and a refusal there
ends the job after the allocation is granted but before any model loads. (The upstream
launcher ran this file as a preflight before allocating; no distill step does, and the recipe
refuses the obvious mistakes at render time instead.) `tracking.py --config <yaml>` reports the
same decision for a person checking a config by hand.

WHAT THIS MODULE DOES NOT DO: decide which PLOT a metric lands in. `ClearMLCallback.on_log`
titles every non-`eval_`/`test_` key "train", which put 11 series of wildly different scale
in one set of axes. `clearml_scalar_groups.py` subclasses the callback to split them by
quantity, and gold.py/sft.py install it after the trainer is built. It is a separate module
because this one decides before any trainer exists.

WHAT EACH BACKEND ACTUALLY READS -- measured against transformers 5.8.0, and read again in
5.20.2, the GPU image's (integrations/integration_utils.py), not assumed:

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

CREDENTIALS ARE DETECTED, NEVER REPORTED. This module establishes presence and reports which
source supplied it; a key never ends up in a log line or a return value. ClearML's are checked
by clearml's own loader in a subprocess that prints only booleans. The W&B netrc entry is
parsed with the stdlib `netrc` module, as wandb itself parses it, so the password passes
through that parser; it is tested for being non-empty and dropped.
"""

from __future__ import annotations

import configparser
import hashlib
import importlib.util
import json
import netrc as _netrc
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

# The config surface. These names are fields on gold.py's and sft.py's `CustomArguments`, and
# this tuple is the single source of truth for them: the steps' renderers emit these keys, and
# gold.py/sft.py read every name listed here off their parsed dataclasses.
TOOL_FIELD = "tracking_tool"
CLEARML_CONFIG_FIELDS = ("clearml_project", "clearml_run_name")
WANDB_CONFIG_FIELDS = ("wandb_entity", "wandb_project", "wandb_run_name")
ALL_CONFIG_FIELDS = (TOOL_FIELD,) + CLEARML_CONFIG_FIELDS + WANDB_CONFIG_FIELDS

# What must be present for a backend to be COMPLETE. The run-name fields are deliberately absent:
# they are auto-fillable, which is the whole point of auto_run_name() below.
REQUIRED = {
    "clearml": ("clearml_project",),
    "wandb": ("wandb_entity", "wandb_project"),
}

# The values `tracking_tool` accepts. "" and "none" both mean off.
BACKENDS = ("clearml", "wandb")
TOOL_OFF = ("", "none")
OWN_FIELDS = {"clearml": CLEARML_CONFIG_FIELDS, "wandb": WANDB_CONFIG_FIELDS}

# The W&B server this module sets when the user set none (see resolve()). ClearML gets none
# from here: clearml's own default, CLEARML_PUBLIC_API, is the public server already.
WANDB_DEFAULT_BASE_URL = "https://api.wandb.ai"
CLEARML_PUBLIC_API = "https://api.clear.ml"

# Integrations that only write local files; every other `report_to` entry is a remote tracker,
# and only the chosen tool may be one.
LOCAL_REPORTERS = ("tensorboard",)


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
        """Set the env vars and `report_to` the callbacks will read. Called by gold.py/sft.py.

        `report_to` ends up as the chosen tool (if any) plus the local-only integrations
        (LOCAL_REPORTERS) the config already asked for. Every other entry is dropped, and said so:
        a config can name trackers through student-sft's extra_config_yaml, and `report_to: all`
        turns on every installed integration (transformers normalises it to that list in
        TrainingArguments.__post_init__, and "none" to []), so without this `tool: none` would
        not stop wandb or clearml.
        """
        env = os.environ if environ is None else environ
        env.update(self.env)
        current = list(getattr(training_args, "report_to", None) or [])
        kept = [r for r in current if r in LOCAL_REPORTERS]
        dropped = [
            r for r in current if r not in LOCAL_REPORTERS and r not in self.enabled
        ]
        if dropped:
            print(
                f"[tracking] report_to: dropping {', '.join(dropped)}. tracking_tool chooses "
                f"the tracking tool ({', '.join(self.enabled) or 'none'}); set it there instead."
            )
        training_args.report_to = kept + [b for b in self.enabled if b not in kept]
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
            if not self._continue_clearml_task(training_args, env, task_init=task_init):
                # No task could be created. Leaving clearml in report_to would let transformers'
                # callback call Task.init again, with upstream's defaults (reuse_last_task_id=True,
                # which deletes the reused task's outputs) and a second chance to fail the job.
                training_args.report_to = [
                    r for r in training_args.report_to if r != "clearml"
                ]
                print(
                    "[tracking] clearml: no task was created, so clearml is removed from "
                    "report_to and this run trains untracked."
                )

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
            print(
                f"[tracking] WARNING: output_dir {out!r} is relative, so the pinned W&B run id "
                f"depends on the CWD ({os.getcwd()}). Pass an absolute --output_dir for a run "
                "id that survives a resubmit."
            )
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
    def _continue_clearml_task(cls, training_args, env: dict, task_init=None) -> bool:
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

        Returns False only when no task could be created at all; apply() then removes clearml
        from report_to, so the callback does not try again with upstream's defaults.

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
            return True
        out = getattr(training_args, "output_dir", None)
        if not out:
            # Nothing to key continuity on. Leaving the callback to its own devices here is the
            # lesser evil: with no output_dir there are no checkpoints, so there is no resume for
            # a task to be continued ACROSS.
            return True
        if is_offline("clearml", env):
            print(
                "[tracking] clearml offline mode: not attempting task continuation (an offline "
                "task has no server-side id to continue). Legs of a restarted run will be "
                "separate offline tasks."
            )
            return True

        out = os.path.abspath(str(out))
        id_file = os.path.join(out, cls.CLEARML_ID_FILE)
        prior = None
        try:
            if os.path.isfile(id_file):
                prior = open(id_file).read().strip() or None
        except OSError as exc:
            print(
                f"[tracking] WARNING: cannot read {id_file}: {exc}. Treating this as a new run."
            )

        if task_init is None:
            try:
                from clearml import Task
            except Exception as exc:  # pragma: no cover - resolve() already required it
                print(
                    f"[tracking] WARNING: clearml import failed at Task.init time ({exc})."
                )
                return False
            task_init = Task.init

        kwargs = dict(
            project_name=env.get("CLEARML_PROJECT"),
            task_name=env.get("CLEARML_TASK"),
            auto_connect_frameworks={"tensorboard": False, "pytorch": False},
        )
        task = None
        if prior:
            try:
                task = task_init(
                    reuse_last_task_id=prior, continue_last_task=0, **kwargs
                )
                print(
                    f"[tracking] clearml continuing task {prior} from {cls.CLEARML_ID_FILE} "
                    "(previous console output and scalars kept, no iteration offset)"
                )
            except Exception as exc:
                print(
                    f"[tracking] WARNING: could not continue clearml task {prior} ({exc}). "
                    "Creating a NEW task rather than letting the callback reuse-and-delete one."
                )
                task = None
        if task is None:
            try:
                task = task_init(reuse_last_task_id=False, **kwargs)
            except Exception as exc:
                print(
                    f"[tracking] WARNING: clearml Task.init failed ({exc}). Continuing without "
                    "tracking rather than failing a job that already holds its allocation."
                )
                return False
            print(
                f"[tracking] clearml new task {getattr(task, 'id', '?')} "
                f"(reuse_last_task_id=False, so no previous task's logs were deleted)"
            )

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
                print(
                    f"[tracking] WARNING: could not record the clearml task id in {id_file} "
                    f"({exc}). A restart of this output_dir will start a new task instead of "
                    "continuing this one."
                )
        return True


def _clean(value) -> str | None:
    """A YAML key present but empty means "not set". step-template.yaml renders every unset
    string key as "", so treating "" as a value would make every granite.build run look
    half-configured and abort."""
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _truthy(value) -> bool:
    """clearml's EnvEntry bool (backend_config/converters.safe_text_to_bool): distutils'
    strtobool, and for any other non-empty string, True."""
    v = str(value or "").strip().lower()
    if v in ("y", "yes", "t", "true", "on", "1"):
        return True
    if v in ("n", "no", "f", "false", "off", "0", ""):
        return False
    return True


# ---- ClearML -----------------------------------------------------------------------------------
#
# clearml resolves its server and credentials from its environment variables and its config
# files (CLEARML_CONFIG_FILE, CLEARML_CONFIG_PATH, ~/clearml.conf, ~/trains.conf, in HOCON: nested,
# dotted, one-line, commented) and falls back to the public server. Rather than re-implement
# that, clearml_probe() runs clearml's own loader -- clearml.config.config_obj and
# Session.get_api_server_host / get_app_server_host / get_files_server_host, and the expressions
# Session.__init__ uses for the keys and CLEARML_AUTH_TOKEN -- in a subprocess with the run's
# environment and HOME. It needs no network. The subprocess prints booleans and server URLs
# only, never a key. Where clearml is not importable there is no probe (None), and resolve()
# then refuses the clearml tool for the missing package anyway.

_CLEARML_PROBE = r"""
import json
try:
    from clearml.config import config_obj as c
    from clearml.backend_api.session.session import Session
    from clearml.backend_api.session.defs import ENV_ACCESS_KEY, ENV_SECRET_KEY, ENV_AUTH_TOKEN
    out = {"public": Session.get_api_server_host(c) == Session.default_host}
    out["token"] = bool(ENV_AUTH_TOKEN.get())
    out["access"] = bool(ENV_ACCESS_KEY.get(default=(c.get("api.credentials.access_key", None) or Session.default_key)))
    out["secret"] = bool(ENV_SECRET_KEY.get(default=(c.get("api.credentials.secret_key", None) or Session.default_secret)))
except Exception as e:
    out = {"error": type(e).__name__}
print("CLEARML_PROBE" + json.dumps(out))
"""


def clearml_probe(environ: Mapping[str, str], home: Path) -> dict | None:
    """clearml's own resolution, as booleans, or None where clearml is not importable.

    {"public": the API server is clearml's public default, "token", "access", "secret": whether
    each credential is present}, or {"error": <exception type>}. The subprocess runs with -I
    (isolated: no user site, no PYTHON* variables, no script directory on sys.path) and prints
    nothing else: never a server URL, which can carry userinfo, nor a message or a stderr line.
    """
    if importlib.util.find_spec("clearml") is None:
        return None
    env = {k: v for k, v in environ.items()}
    env["HOME"] = str(home)
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", _CLEARML_PROBE],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": type(exc).__name__}
    for line in result.stdout.splitlines():
        if line.startswith("CLEARML_PROBE"):
            return json.loads(line[len("CLEARML_PROBE") :])
    return {"error": f"probe exited {result.returncode} without a result"}


def clearml_verdict(probe: dict | None) -> tuple[bool, str, str]:
    """(credentials present, where, server description) from clearml_probe()'s result.

    Present means what clearml needs to log in: CLEARML_AUTH_TOKEN, or BOTH keys (it uses basic
    auth only with both, backend_api/session/session.py, Session._do_refresh_token). One key alone
    gets past clearml's MissingConfigError but fails at login, so it is not credentials here.
    The server is reported as public or self-hosted only, never by its URL.
    """
    if probe is None:
        return (
            False,
            "clearml is not importable here, so its configuration was not read",
            "server not checked",
        )
    if "error" in probe:
        return (
            False,
            f"clearml's config could not be read ({probe['error']})",
            "server not checked",
        )
    if probe.get("public"):
        server = "server: the public ClearML server (clearml's default)"
    else:
        server = "server: self-hosted (from clearml's config or environment)"
    if probe.get("token"):
        return True, "CLEARML_AUTH_TOKEN", server
    if probe.get("access") and probe.get("secret"):
        return (
            True,
            "an access/secret key pair (clearml's environment or config file)",
            server,
        )
    if probe.get("access") or probe.get("secret"):
        return False, "only one of the access key and the secret key is set", server
    return False, "no CLEARML_AUTH_TOKEN and no access/secret key pair", server


# ---- W&B ---------------------------------------------------------------------------------------
#
# What wandb 0.31.2 does, read from its source (sdk/lib/urls.py, sdk/lib/wbauth/host_url.py,
# wbnetrc.py, authenticate.py) and checked case by case against it (see the parity cases in
# self_test()):
#   * the server: WANDB_BASE_URL, else `base_url` in its settings files, else
#     https://api.forge.coreweave.com. This module puts https://api.wandb.ai in that last place.
#   * the URL is validated (http/https with a host); a Forge host is accepted only as its
#     `https://<forge>/api/wandb` proxy form, without a port, which becomes `https://api.<forge>`;
#     a wandb.ai address must be an `api.` one and https.
#   * credentials: WANDB_API_KEY, or WANDB_IDENTITY_TOKEN_FILE (both at once is an error);
#     else the netrc entry for the server's netloc, then, for a Forge server, the entry for the
#     W&B API host behind it. An entry found but without a password is NOT followed by the
#     fallback. The netrc file is $NETRC, else ~/.netrc, else ~/_netrc.
#   * the mode: WANDB_MODE, case-sensitive, one of online, offline, shared, disabled, dryrun,
#     run; offline and dryrun need no credentials, and any other value makes wandb raise.
#
# One wandb source is not read here: an `api_key` in a wandb settings file, which wandb uses
# only after the environment and netrc and only for the same base_url.

WANDB_FORGE_HOSTS = {
    "forge.coreweave.com": "api.wandb.ai",
    "qa.forge.coreweave.com": "api.qa.wandb.ai",
}
WANDB_FORGE_API_PATH = "/api/wandb"

# Validate W&B server addresses with wandb's own Settings validator when wandb is importable.
# The self-test turns this off, so it checks this module's own rules (written against wandb
# 0.31.2) whatever wandb the interpreter carries; test_tracking.py checks wandb's in the image.
USE_WANDB_VALIDATOR = True


def wandb_normalize_base_url(url: str) -> str:
    """The server URL wandb would use for `url`, or ValueError where wandb raises one.

    Where wandb is importable (the GPU image), its own Settings validator decides, so the
    verdict is wandb's. Otherwise (the repo venv) this re-implements it: pydantic's strict URL
    check (http/https, a valid host and port, no whitespace) and wandb's Forge and wandb.ai rules.
    """
    if USE_WANDB_VALIDATOR and importlib.util.find_spec("wandb") is not None:
        try:
            from wandb.sdk.wandb_settings import Settings  # noqa: PLC0415

            return Settings(base_url=url).base_url
        except ImportError:
            pass
        except (
            Exception
        ) as exc:  # pydantic's ValidationError, or wandb's own ValueError
            raise ValueError(f"{url!r} is not a valid server address: {exc}") from None
    if url != url.strip() or any(ch.isspace() for ch in url):
        raise ValueError(f"{url!r} is not a valid server address (whitespace)")
    parts = urlsplit(url)
    try:
        parts.port
    except ValueError as exc:
        raise ValueError(f"{url!r} is not a valid server address ({exc})") from None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(
            f"{url!r} is not a valid server address (need http(s)://<host>)"
        )
    if parts.hostname in WANDB_FORGE_HOSTS:
        if (
            parts.scheme == "https"
            and parts.port is None
            and parts.path.rstrip("/") == WANDB_FORGE_API_PATH
        ):
            url = f"https://api.{parts.hostname}"
        else:
            raise ValueError(
                f"{url!r} is not a valid Forge server address; use https://api.{parts.hostname}"
            )
    if re.match(r".*wandb\.ai[^\.]*$", url):
        if "api." not in url:
            raise ValueError(
                f"{url!r} is not a valid server address (use an api. address)"
            )
        if not url.startswith("https"):
            raise ValueError(f"{url!r}: http is not secure, use https")
    return url.rstrip("/")


def wandb_forge_upstream(url: str) -> str:
    """The W&B API URL behind a Forge URL (urls.forge_upstream_url), or the URL as is."""
    hostname = urlsplit(url).hostname
    for forge, api in WANDB_FORGE_HOSTS.items():
        if hostname in (forge, f"api.{forge}", api):
            return f"https://{api}"
    return url


def wandb_settings_base_url(
    environ: Mapping[str, str], home: Path, cwd: Path | None = None
) -> tuple[str, str] | None:
    """(base_url, file) from wandb's settings files, if one sets it.

    wandb reads two (Settings.read_system_settings): the workspace file, <cwd>/wandb/settings,
    which wins, and the global one, $WANDB_CONFIG_DIR/settings, else ~/.config/wandb/settings. It
    reads them before it applies the environment, so WANDB_DIR does not move the workspace file:
    it is the one under the directory the trainer starts in. Then WANDB_BASE_URL, which
    overrides both.
    """
    cfg_dir = (environ.get("WANDB_CONFIG_DIR") or "").strip()
    global_file = (
        Path(cfg_dir).expanduser() / "settings"
        if cfg_dir
        else home / ".config" / "wandb" / "settings"
    )
    workspace_file = (Path.cwd() if cwd is None else cwd) / "wandb" / "settings"
    for path in (workspace_file, global_file):
        parser = configparser.ConfigParser()
        try:
            parser.read_string(path.read_text(errors="replace"))
        except (OSError, configparser.Error):
            continue
        value = parser.get("default", "base_url", fallback="").strip()
        if value:
            return value, str(path)
    return None


def wandb_base_url(
    environ: Mapping[str, str], home: Path, strip: bool = False, cwd: Path | None = None
) -> tuple[str, str]:
    """(the server URL as configured, where it came from). Not yet validated.

    The value is NOT stripped by default: wandb reads WANDB_BASE_URL as it is, so a stray space
    that wandb rejects is rejected here too. An empty or all-space value counts as unset.
    """
    raw = environ.get("WANDB_BASE_URL") or ""
    if raw.strip():
        return (raw.strip() if strip else raw), "WANDB_BASE_URL"
    found = wandb_settings_base_url(environ, home, cwd)
    if found:
        return found[0], found[1]
    return WANDB_DEFAULT_BASE_URL, "default"


def wandb_netrc_path(environ: Mapping[str, str], home: Path) -> Path:
    """wbnetrc._get_netrc_file_path, with `home` for ~."""
    if environ.get("NETRC"):
        return Path(environ["NETRC"]).expanduser()
    for name in (".netrc", "_netrc"):
        if (home / name).exists():
            return home / name
    return home / ".netrc"


def wandb_netrc_machines(url: str) -> tuple[str, ...]:
    """The netrc machines wandb tries for the (normalized) server URL, in order."""
    netloc = urlsplit(url).netloc
    machines = [netloc] if netloc else []
    upstream = wandb_forge_upstream(url)
    if upstream != url:
        machines.append(urlsplit(upstream).netloc)
    return tuple(machines)


def wandb_auth_conflict(environ: Mapping[str, str]) -> str | None:
    if environ.get("WANDB_API_KEY") and environ.get("WANDB_IDENTITY_TOKEN_FILE"):
        return (
            "both WANDB_API_KEY and WANDB_IDENTITY_TOKEN_FILE are set; wandb refuses that "
            "combination at login. Unset one of them."
        )
    return None


def wandb_credentials(environ: Mapping[str, str], home: Path) -> tuple[bool, str]:
    """(present, where) for the server wandb will use. Presence only."""
    if wandb_auth_conflict(environ):
        return False, "WANDB_API_KEY and WANDB_IDENTITY_TOKEN_FILE are both set"
    if environ.get("WANDB_API_KEY"):
        return True, "WANDB_API_KEY in the environment"
    token_file = (environ.get("WANDB_IDENTITY_TOKEN_FILE") or "").strip()
    if token_file:
        if Path(token_file).expanduser().is_file():
            return True, f"WANDB_IDENTITY_TOKEN_FILE ({token_file})"
        return (
            False,
            f"WANDB_IDENTITY_TOKEN_FILE names {token_file}, which is not a file",
        )
    raw, _ = wandb_base_url(environ, home, strip=False)
    try:
        url = wandb_normalize_base_url(raw)
    except ValueError as exc:
        return False, f"invalid server address: {exc}"
    path = wandb_netrc_path(environ, home)
    try:
        parsed = _netrc.netrc(str(path))
    except FileNotFoundError:
        parsed = None
    except (_netrc.NetrcParseError, OSError) as exc:
        return False, f"cannot read {path}: {getattr(exc, 'msg', exc)}"
    machines = wandb_netrc_machines(url)
    if parsed is not None:
        for m in machines:
            creds = parsed.authenticators(m)
            if not creds:
                continue
            if not creds[2]:
                # wandb stops at the first entry it finds, and warns, even if a later one has a key.
                return False, f"{path}: the entry for {m} has no password"
            entry = f"machine {m}" if m in parsed.hosts else "default entry"
            return True, f"{path} ({entry})"
    return False, f"no WANDB_API_KEY and no {path} entry for {' or '.join(machines)}"


WANDB_MODES = ("online", "offline", "shared", "disabled", "dryrun", "run")


def _clearml_env(environ: Mapping[str, str], name: str) -> str:
    """clearml's EnvEntry: CLEARML_<x>, else TRAINS_<x>; a value that strips to empty is unset."""
    for key in (name, "TRAINS_" + name[len("CLEARML_") :]):
        value = (environ.get(key) or "").strip()
        if value:
            return value
    return ""


def is_offline(backend: str, environ: Mapping[str, str]) -> bool:
    """Offline mode needs no credentials, so it SATISFIES the key requirement.

    No new config field for this: both libraries already have a standard env var, and a gold
    yaml can set either through TrlParser's native `env:` block. Recognising them here is what
    keeps "offline" from being reported as "no credentials, tracking off". Each is read the way
    its library reads it: clearml's CLEARML_OFFLINE_MODE / TRAINS_OFFLINE_MODE through its bool
    parser; WANDB_MODE exactly, since wandb's mode values are case-sensitive.
    """
    if backend == "clearml":
        return _truthy(_clearml_env(environ, "CLEARML_OFFLINE_MODE"))
    return environ.get("WANDB_MODE") in ("offline", "dryrun")


def is_disabled(backend: str, environ: Mapping[str, str]) -> bool:
    """WANDB_MODE=disabled makes every wandb call a no-op: the tool is chosen but logs nothing."""
    return backend == "wandb" and environ.get("WANDB_MODE") == "disabled"


def wandb_mode_problem(environ: Mapping[str, str]) -> str | None:
    """A WANDB_MODE wandb would refuse at wandb.init (its Settings.mode is a case-sensitive Literal)."""
    mode = environ.get("WANDB_MODE")
    if mode is None or mode == "" or mode in WANDB_MODES:
        return None
    return (
        f"WANDB_MODE={mode!r} is not a mode wandb accepts; it is case-sensitive, one of "
        f"{', '.join(WANDB_MODES)}. wandb would refuse it when the run starts."
    )


def _short(path_like: str | None) -> str | None:
    """Last path component, so a run name is not 120 characters of absolute path."""
    if not path_like:
        return None
    return Path(str(path_like)).name or None


GENERIC_STEMS = {
    "gold-config.rendered",
    "gold-config",
    "config",
    "gold",
    "rendered",
    "sft-config.rendered",
    "sft-config",
    "sft",
}


def auto_run_name(
    config_path: str | None, hints: Mapping[str, object], environ: Mapping[str, str]
) -> str:
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
            # NEITHER a teacher NOR an lmbda: a plain SFT run -- student-sft's control,
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
            arm = (
                "onpolicy" if (lmbda is not None and float(lmbda) > 0) else "offpolicy"
            )
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
    clearml_probe_fn=None,
) -> Plan:
    """Decide on/off/fatal for the chosen tool -- and say why in every case.

    `fields` are the CustomArguments tracking values (or the raw yaml keys, for the CLI).
    `hints` are the semantic config keys used only to compose a fallback run name.
    `is_installed` is injectable so the self-test can exercise the ENABLED path in an
    interpreter where neither package is installed -- which is the state of the repo venv.
    Without that seam every happy-path case would abort on the package check and the on-path
    would never be tested at all. `clearml_probe_fn` replaces clearml_probe() the same way.
    """
    environ = os.environ if environ is None else environ
    home = Path(os.path.expanduser("~")) if home is None else home
    hints = hints or {}

    plan = Plan()
    vals = {k: _clean(fields.get(k)) for k in ALL_CONFIG_FIELDS}
    tool = (vals.get(TOOL_FIELD) or "").lower()
    set_keys = {b: sorted(k for k in OWN_FIELDS[b] if vals.get(k)) for b in BACKENDS}

    # ---- Which tool. One, explicitly, or none.
    if tool not in TOOL_OFF + BACKENDS:
        plan.fatal.append(
            f"tracking_tool={vals.get(TOOL_FIELD)!r} is not one of wandb, clearml, none "
            "(empty means none)."
        )
        return plan
    if tool in TOOL_OFF:
        stray = [k for b in BACKENDS for k in set_keys[b]]
        if stray:
            plan.fatal.append(
                f"{', '.join(stray)} set, but tracking_tool is not: choose the tool "
                "(tracking_tool: wandb or clearml) or remove those keys. Nothing is inferred "
                "from which keys happen to be set."
            )
        else:
            plan.lines.append(
                "tracking : off      (tracking_tool not set; nothing configured)"
            )
        return plan
    other = next(b for b in BACKENDS if b != tool)
    if set_keys[other]:
        plan.fatal.append(
            f"tracking_tool is {tool}, but {', '.join(set_keys[other])} ({other}'s keys) are set "
            f"too. One tool per run: remove the {other} keys, or choose {other}."
        )
        return plan

    backend = tool
    missing = [k for k in REQUIRED[backend] if not vals.get(k)]
    if missing:
        given = ", ".join(set_keys[backend]) or "no project keys"
        plan.fatal.append(
            f"{backend} is partially configured: {given} given, but {', '.join(missing)} "
            f"missing. A partial tracking config is an expressed intent to track that cannot "
            f"be honoured -- and {backend} would not fail, it would log to its own default "
            f"project. Complete it, or set tracking_tool to none."
        )
        return plan

    # The package first: without it there is nothing to talk to (and for ClearML, nothing to
    # read its configuration with). A missing key is an unconfigured environment; a missing
    # package is a broken one, so this is fatal, credentials or not.
    if is_installed is None:

        def is_installed(name: str) -> bool:
            return importlib.util.find_spec(name) is not None

    if not is_installed(backend):
        plan.fatal.append(
            f"tracking_tool is {backend}, but the `{backend}` package is not installed in this "
            f"interpreter. Training would proceed UNTRACKED, which is the failure this config "
            f"exists to prevent. Use the distill GPU image, or set tracking_tool to none."
        )
        return plan

    if backend == "wandb":
        problem = wandb_mode_problem(environ)
        if problem:
            plan.fatal.append(f"wandb: {problem}")
            return plan

    if is_disabled(backend, environ):
        plan.lines.append(
            "tracking : wandb    OFF      WANDB_MODE=disabled: wandb logs nothing, so it is not "
            "added to report_to."
        )
        return plan

    # ---- Which server. The public one unless the user set another (see the module docstring).
    probe = None
    if backend == "wandb":
        conflict = wandb_auth_conflict(environ)
        if conflict:
            plan.fatal.append(f"wandb: {conflict}")
            return plan
        raw, source = wandb_base_url(environ, home)
        try:
            url = wandb_normalize_base_url(raw)
        except ValueError as exc:
            if source == "WANDB_BASE_URL":
                why = "wandb would refuse it when the run starts."
            else:
                why = (
                    "wandb itself would print an error, ignore the file's settings and use its own "
                    "default server; refused here instead, so the run does not silently go "
                    "somewhere other than the file says."
                )
            plan.fatal.append(f"wandb: {exc} (from {source}). {why}")
            return plan
        if source == "default":
            plan.env["WANDB_BASE_URL"] = WANDB_DEFAULT_BASE_URL
            server = f"server={url} (default)"
        else:
            server = f"server={url} (set by {source})"
    else:
        # Nothing is set for ClearML: clearml's own resolution is reported, not overridden.
        probe = (clearml_probe_fn or clearml_probe)(environ, home)
        _, _, server = clearml_verdict(probe)

    # ---- Credentials, for that server.
    if is_offline(backend, environ):
        keyed, where = True, "offline mode -- no credentials needed"
    elif backend == "clearml":
        keyed, where, _ = clearml_verdict(probe)
    else:
        keyed, where = wandb_credentials({**environ, **plan.env}, home)

    run_name = vals.get(f"{backend}_run_name") or auto_run_name(
        config_path, hints, environ
    )
    origin = "from config" if vals.get(f"{backend}_run_name") else "auto-generated"

    if not keyed:
        # Configured completely, but no key. Per the rules this is OFF, not fatal -- and it
        # is printed at full volume, because "I configured tracking and got none" is the
        # exact surprise this line exists to prevent. The server env is dropped with it.
        plan.env.clear()
        plan.lines.append(
            f"tracking : {backend:8s} OFF      config is complete but NO CREDENTIALS ({where}); "
            f"{server}. Not tracking. Would have been: {run_name}"
        )
        return plan

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
        # allocation it is wall-clock spent not training. With the public files server as
        # the default, it would also be checkpoints leaving for files.clear.ml.
        #
        # setdefault semantics, not policy: an operator who exports CLEARML_LOG_MODEL means
        # it, and Plan.apply() does env.update(), which would otherwise clobber the
        # deliberate choice. Turning this on for real wants a config key rather than an
        # export -- deliberately not added until somebody wants the behaviour.
        if "CLEARML_LOG_MODEL" not in environ:
            plan.env["CLEARML_LOG_MODEL"] = "FALSE"
        plan.lines.append(
            f"tracking : clearml  ON       project={vals['clearml_project']} "
            f"task={run_name} ({origin}); {server}; key: {where}"
        )
    else:
        plan.env["WANDB_ENTITY"] = vals["wandb_entity"]
        plan.env["WANDB_PROJECT"] = vals["wandb_project"]
        plan.run_name = run_name
        plan.lines.append(
            f"tracking : wandb    ON       entity={vals['wandb_entity']} "
            f"project={vals['wandb_project']} name={run_name} ({origin}); {server}; key: {where}"
        )
    plan.enabled = (backend,)

    return plan


# --------------------------------------------------------------------------------------------
# Self-test and CLI.
#
# The CLI reports the decision for a rendered yaml, for a person checking a config; no step
# calls it (the trainers call resolve() themselves).


def load_yaml_fields(path: Path) -> tuple[dict, dict]:
    """(tracking fields, run-name hints) read straight from a gold yaml.

    NOT via TrlParser, and the difference is worth stating: TrlParser also applies command-line
    overrides, so this sees the FILE and not necessarily the final parsed values. That is the
    right trade for a check by hand, and gold.py/sft.py resolve from the parsed dataclass,
    which is the authoritative pass. If the two
    ever disagree, gold.py wins -- and it prints its own report, so the disagreement is visible.
    """
    import yaml

    doc = yaml.safe_load(path.read_text()) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"{path} is not a YAML mapping")
    fields = {k: doc.get(k) for k in ALL_CONFIG_FIELDS}
    hints = {
        k: doc.get(k)
        for k in (
            "model_name_or_path",
            "teacher_model_name_or_path",
            "dataset_name",
            "lmbda",
        )
    }
    return fields, hints


def _fake_clearml_probe(environ, home):
    """The self-test's stand-in for clearml_probe(): keys and token from the environment only."""
    return {
        "public": True,
        "token": bool(environ.get("CLEARML_AUTH_TOKEN")),
        "access": bool(
            environ.get("CLEARML_API_ACCESS_KEY")
            or environ.get("TRAINS_API_ACCESS_KEY")
        ),
        "secret": bool(
            environ.get("CLEARML_API_SECRET_KEY")
            or environ.get("TRAINS_API_SECRET_KEY")
        ),
    }


def self_test() -> int:
    """Run the checks below with a stand-in ClearML probe (real clearml: test_tracking.py)."""
    global clearml_probe, USE_WANDB_VALIDATOR
    real, validator = clearml_probe, USE_WANDB_VALIDATOR
    clearml_probe, USE_WANDB_VALIDATOR = _fake_clearml_probe, False
    try:
        return _self_test()
    finally:
        clearml_probe, USE_WANDB_VALIDATOR = real, validator


def _self_test() -> int:
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
    check(
        "nothing configured -> off, no fatal",
        p.enabled == () and not p.fatal and len(p.lines) == 1,
        f"{p.enabled} {p.fatal}",
    )

    # 2. ClearML complete, key in env. The default backend on its happy path.
    env = {"CLEARML_API_ACCESS_KEY": "x", "CLEARML_API_SECRET_KEY": "y"}
    p = resolve(
        {"tracking_tool": "clearml", "clearml_project": "distill"},
        environ=env,
        home=Path("/nonexistent"),
        config_path="/c/granite-4.1-3b_onpolicy_node2.yaml",
        is_installed=yes,
    )
    check(
        "clearml complete + key -> ON, env set, name from stem",
        p.enabled == ("clearml",)
        and p.env["CLEARML_PROJECT"] == "distill"
        and p.env["CLEARML_TASK"] == "granite-4.1-3b_onpolicy_node2"
        and not p.fatal,
        f"{p.enabled} {p.env} {p.fatal}",
    )

    # 3. Partial: a run name but no project. Must ABORT, because ClearML would silently use its
    #    own default project instead of failing.
    p = resolve(
        {"tracking_tool": "clearml", "clearml_run_name": "mine"},
        environ=env,
        home=Path("/nonexistent"),
        is_installed=yes,
    )
    check(
        "clearml run_name without project -> FATAL",
        not p.enabled
        and any(
            "partially configured" in f and "clearml_project" in f for f in p.fatal
        ),
        f"{p.fatal}",
    )

    # 4. Partial the other way round, on the backend where entity is real.
    p = resolve(
        {"tracking_tool": "wandb", "wandb_entity": "ibm"},
        environ={"WANDB_API_KEY": "k"},
        home=Path("/nonexistent"),
        is_installed=yes,
    )
    check(
        "wandb entity without project -> FATAL",
        not p.enabled and any("wandb_project" in f for f in p.fatal),
        f"{p.fatal}",
    )

    # 5. W&B complete. Note run_name lands on the PLAN (for training_args.run_name), not in env --
    #    the asymmetry the docstring describes.
    p = resolve(
        {
            "tracking_tool": "wandb",
            "wandb_entity": "ibm",
            "wandb_project": "distill",
            "wandb_run_name": "r1",
        },
        environ={"WANDB_API_KEY": "k"},
        home=Path("/nonexistent"),
        is_installed=yes,
    )
    check(
        "wandb complete -> ON, entity+project in env, name on the plan",
        p.enabled == ("wandb",)
        and p.env["WANDB_ENTITY"] == "ibm"
        and p.env["WANDB_PROJECT"] == "distill"
        and p.run_name == "r1"
        and "WANDB_NAME" not in p.env,
        f"{p.enabled} {p.env} {p.run_name}",
    )

    # 6. Complete but unkeyed -> off, NOT fatal, and loud. This is the specified behaviour and
    #    the one most likely to be quietly mis-implemented as either fatal or silent.
    p = resolve(
        {"tracking_tool": "clearml", "clearml_project": "distill"},
        environ={},
        home=Path("/nonexistent"),
        is_installed=yes,
    )
    check(
        "complete but no key -> off, not fatal, says NO CREDENTIALS",
        not p.enabled and not p.fatal and any("NO CREDENTIALS" in l for l in p.lines),
        f"{p.lines}",
    )

    # 7. Offline mode needs no key, so it must NOT be reported as unkeyed.
    p = resolve(
        {"tracking_tool": "clearml", "clearml_project": "d"},
        environ={"CLEARML_OFFLINE_MODE": "1"},
        home=Path("/nonexistent"),
        is_installed=yes,
    )
    check(
        "CLEARML_OFFLINE_MODE=1 satisfies the key requirement",
        p.enabled == ("clearml",) and any("offline mode" in l for l in p.lines),
        f"{p.lines}",
    )
    p = resolve(
        {"tracking_tool": "wandb", "wandb_entity": "e", "wandb_project": "p"},
        environ={"WANDB_MODE": "offline"},
        home=Path("/nonexistent"),
        is_installed=yes,
    )
    check(
        "WANDB_MODE=offline satisfies the key requirement",
        p.enabled == ("wandb",),
        f"{p.lines}",
    )

    # 8. A name in the config beats the generated one, and says so.
    p = resolve(
        {
            "tracking_tool": "clearml",
            "clearml_project": "d",
            "clearml_run_name": "explicit",
        },
        environ=env,
        home=Path("/nonexistent"),
        config_path="/c/some_config.yaml",
        is_installed=yes,
    )
    check(
        "config run name wins over auto",
        p.env["CLEARML_TASK"] == "explicit"
        and any("from config" in l for l in p.lines),
        f"{p.lines}",
    )

    # 9. The granite.build path: the rendered stem carries nothing, so compose from the config.
    name = auto_run_name(
        "/out/gold-config.rendered.yaml",
        {
            "model_name_or_path": "/m/granite-4.1-3b-base_retagged_v2",
            "teacher_model_name_or_path": "/hub/snapshots/a0057a78",
            "dataset_name": "/d/subsampled_0.2_shuffled.jsonl",
            "lmbda": 0.3,
        },
        {},
    )
    check(
        "generic rendered stem -> composed semantic name",
        name
        == "granite-4.1-3b-base_retagged_v2_from-a0057a78_subsampled_0.2_shuffled_onpolicy_lmbda0.3",
        name,
    )

    # 10. lmbda 0.0 is the off-policy arm and must not be called onpolicy -- the arm is the single
    #     most consequential key in this recipe, so a run name that lies about it is a real cost.
    off = auto_run_name("/out/gold-config.rendered.yaml", {"lmbda": 0.0}, {})
    check(
        "lmbda 0.0 -> offpolicy in the composed name",
        "offpolicy" in off and "onpolicy" not in off,
        off,
    )

    # 10b. ...and a run with NEITHER a teacher NOR an lmbda is a plain SFT control, which must
    #      not be named as if it had either. Sits beside 10 because both are the same class of
    #      bug: a run name that misreports which arm produced the numbers. student-sft
    #      renders to `sft-config.rendered.yaml`, a generic stem, so this branch is what its
    #      auto-generated names actually come from.
    sft = auto_run_name(
        "/out/sft-config.rendered.yaml",
        {
            "model_name_or_path": "/m/granite-4.1-3b-base_retagged_v2",
            "dataset_name": "/d/subsampled_0.2_shuffled.jsonl",
        },
        {},
    )
    check(
        "no teacher and no lmbda -> plain sft name",
        sft == "granite-4.1-3b-base_retagged_v2_subsampled_0.2_shuffled_sft",
        sft,
    )
    #      The KD variant of that same step DOES have a teacher (the precompute directory), and
    #      must read as a distillation run rather than as the control.
    kd = auto_run_name(
        "/out/sft-config.rendered.yaml",
        {
            "model_name_or_path": "/m/student",
            "teacher_model_name_or_path": "/d/logits_granite-4.2-30b_top256",
            "dataset_name": "/d/corpus.jsonl",
        },
        {},
    )
    check(
        "teacher present, lmbda absent -> named as a distillation run",
        kd == "student_from-logits_granite-4.2-30b_top256_corpus_offpolicy",
        kd,
    )

    # 11. The LSF job id is appended, because it survives preemption and requeue where a
    #     timestamp would split one run into several.
    check(
        "LSB_JOBID is appended",
        auto_run_name("/c/cfg_node2.yaml", {}, {"LSB_JOBID": "1152213"})
        == "cfg_node2-lsf1152213",
    )

    # 12. The chosen tool's package absent -> FATAL, credentials or not. Distinct from the no-key
    #     case above: a missing key is an unconfigured environment, a missing package a broken one.
    #     For ClearML this is also the path where no probe can run (clearml_probe() is None), so
    #     the case below passes no probe at all, as the real code has none.
    p = resolve(
        {"tracking_tool": "clearml", "clearml_project": "d"},
        environ=env,
        home=Path("/nonexistent"),
        is_installed=no,
        clearml_probe_fn=lambda e, h: None,
    )
    check(
        "clearml chosen, keys present, clearml not installed (no probe) -> FATAL not installed",
        not p.enabled and any("not installed" in f for f in p.fatal),
        f"{p.lines} {p.fatal}",
    )
    p = resolve(
        {"tracking_tool": "wandb", "wandb_entity": "e", "wandb_project": "p"},
        environ={},
        home=Path("/nonexistent"),
        is_installed=no,
    )
    check(
        "wandb chosen, no credentials, wandb not installed -> FATAL not installed (not off)",
        any("not installed" in f for f in p.fatal),
        f"{p.lines} {p.fatal}",
    )
    p = resolve(
        {"tracking_tool": "clearml", "clearml_project": "d"},
        environ=env,
        home=Path("/nonexistent"),
        is_installed=no,
    )
    check(
        "enabled but package missing -> FATAL",
        any("not installed" in f for f in p.fatal),
        f"{p.fatal}",
    )

    # 13. Credential detection off disk, both backends, in a real temp HOME.
    with tempfile.TemporaryDirectory() as td:
        h = Path(td)
        check("no wandb creds on a bare home", not wandb_credentials({}, h)[0])
        (h / ".netrc").write_text(
            "machine api.wandb.ai\n  login user\n  password abc123\n"
        )
        ok, where = wandb_credentials({}, h)
        check(
            "netrc entry for api.wandb.ai is detected", ok and ".netrc" in where, where
        )
        # wandb >= 0.31's `wandb login` writes its default server's host, not api.wandb.ai.
        # That entry is credentials only when the run talks to Forge...
        (h / ".netrc").write_text(
            "machine api.forge.coreweave.com\n  login user\n  password abc123\n"
        )
        ok, where = wandb_credentials(
            {"WANDB_BASE_URL": "https://api.forge.coreweave.com"}, h
        )
        check(
            "netrc entry for api.forge.coreweave.com is detected for a Forge server",
            ok and "api.forge.coreweave.com" in where,
            where,
        )
        # ...and not for api.wandb.ai, this module's default: wandb then reads only that entry.
        check(
            "forge netrc entry is not credentials for the default server, api.wandb.ai",
            not wandb_credentials({}, h)[0]
            and not wandb_credentials({"WANDB_BASE_URL": "https://api.wandb.ai"}, h)[0],
        )
        (h / ".netrc").write_text(
            "machine example.com\n  login user\n  password abc123\n"
        )
        check(
            "netrc for a DIFFERENT machine is not wandb credentials",
            not wandb_credentials({}, h)[0],
        )

    # 14. "" is not a value. step-template.yaml renders every unset string key as "", so if this
    #     were wrong every granite.build run would abort as half-configured.
    p = resolve(
        {"clearml_project": "", "clearml_run_name": "  "},
        environ=env,
        home=Path("/nonexistent"),
        is_installed=yes,
    )
    check(
        "empty-string yaml keys count as unset",
        p.enabled == () and not p.fatal,
        f"{p.enabled} {p.fatal}",
    )

    # 15. apply(): report_to is EXTENDED, not replaced, and run_name is only touched for wandb.
    class FakeArgs:
        report_to = ["tensorboard"]
        run_name = "/some/output_dir"

    a = FakeArgs()
    env2: dict[str, str] = {}
    resolve(
        {"tracking_tool": "wandb", "wandb_entity": "e", "wandb_project": "p"},
        environ={"WANDB_API_KEY": "k"},
        home=Path("/nonexistent"),
        config_path="/c/cfg.yaml",
        is_installed=yes,
    ).apply(a, env2)
    check(
        "apply extends report_to and sets run_name for wandb",
        a.report_to == ["tensorboard", "wandb"]
        and a.run_name == "cfg"
        and env2["WANDB_ENTITY"] == "e",
        f"{a.report_to} {a.run_name} {env2}",
    )

    b = FakeArgs()
    resolve(
        {"tracking_tool": "clearml", "clearml_project": "d"},
        environ=env,
        home=Path("/nonexistent"),
        config_path="/c/cfg.yaml",
        is_installed=yes,
    ).apply(b, {})
    check(
        "apply does NOT touch run_name for clearml (it reads CLEARML_TASK)",
        b.report_to == ["tensorboard", "clearml"] and b.run_name == "/some/output_dir",
        f"{b.report_to} {b.run_name}",
    )

    # 15b. The chosen tool is the only remote tracker in report_to; local tensorboard stays.
    class AllArgs:
        report_to = ["tensorboard", "wandb", "clearml", "mlflow"]
        run_name = "x"

    c = AllArgs()
    resolve({"tracking_tool": "none"}, environ={}, home=Path("/nonexistent")).apply(
        c, {}
    )
    check(
        "tool none: every remote tracker leaves report_to, tensorboard stays",
        c.report_to == ["tensorboard"],
        f"{c.report_to}",
    )
    c = AllArgs()
    resolve(
        {"tracking_tool": "wandb", "wandb_entity": "e", "wandb_project": "p"},
        environ={"WANDB_API_KEY": "k"},
        home=Path("/nonexistent"),
        is_installed=yes,
    ).apply(c, {})
    check(
        "tool wandb: clearml and mlflow leave report_to, wandb is there once",
        c.report_to == ["tensorboard", "wandb"],
        f"{c.report_to}",
    )

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
            if (
                self.boom
                and "reuse_last_task_id" in kw
                and isinstance(kw["reuse_last_task_id"], str)
            ):
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
        resolve(
            cfg
            or {"tracking_tool": "wandb", "wandb_entity": "e", "wandb_project": "p"},
            environ={"WANDB_API_KEY": "k", **(environ or {})},
            home=Path("/nonexistent"),
            config_path="/c/cfg.yaml",
            is_installed=yes,
        ).apply(ArgsWithOut(out), e, task_init=rec or Recorder())
        return e

    j1 = pin("/runs/mine", {"LSB_JOBID": "111"})
    j2 = pin("/runs/mine", {"LSB_JOBID": "222"})
    check(
        "wandb run id is the same for one output_dir under two different LSF job ids",
        j1.get("WANDB_RUN_ID") and j1["WANDB_RUN_ID"] == j2["WANDB_RUN_ID"],
        f"{j1.get('WANDB_RUN_ID')} vs {j2.get('WANDB_RUN_ID')}",
    )
    check(
        "wandb resume policy is allow, so the FIRST launch is not an error",
        j1.get("WANDB_RESUME") == "allow",
        f"{j1.get('WANDB_RESUME')}",
    )
    # A trailing slash and a relative path must land on the same run as the absolute form,
    # because a resubmit typed by hand is exactly where those differ.
    check(
        "wandb run id is path-normalised, not string-keyed",
        pin("/runs/mine/").get("WANDB_RUN_ID") == j1["WANDB_RUN_ID"],
        f"{pin('/runs/mine/').get('WANDB_RUN_ID')} vs {j1['WANDB_RUN_ID']}",
    )
    check(
        "a DIFFERENT output_dir gets a different run id",
        pin("/runs/other").get("WANDB_RUN_ID") != j1["WANDB_RUN_ID"],
    )
    # setdefault, for the reason CLEARML_LOG_MODEL uses it: an operator who set the id by hand
    # meant it, and this code must not be able to overrule them.
    check(
        "a caller-set WANDB_RUN_ID survives",
        pin("/runs/mine", {"WANDB_RUN_ID": "hand-picked"})["WANDB_RUN_ID"]
        == "hand-picked",
    )
    check("no output_dir -> no run id invented", "WANDB_RUN_ID" not in env2, f"{env2}")
    clearml_env = pin(
        "/runs/mine",
        {"CLEARML_API_ACCESS_KEY": "a", "CLEARML_API_SECRET_KEY": "b"},
        cfg={"tracking_tool": "clearml", "clearml_project": "d"},
    )
    check(
        "clearml-only run does not get a WANDB_RUN_ID",
        "WANDB_RUN_ID" not in clearml_env,
        f"{clearml_env}",
    )

    # 18. ClearML task continuation. Nine cases, and the reason there are nine rather than two is
    #     that every one of them corresponds to a way this can silently do the WRONG thing --
    #     delete a previous leg's logs, fold a resumed curve over itself, or create one task per
    #     rank. The kwargs are asserted individually, not as a blob, because `continue_last_task=0`
    #     versus `False` and `reuse_last_task_id=<id>` versus `True` are single-token differences
    #     with opposite consequences.
    cl_cfg = {"tracking_tool": "clearml", "clearml_project": "d"}
    cl_env = {"CLEARML_API_ACCESS_KEY": "a", "CLEARML_API_SECRET_KEY": "b"}
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ckpt-dir"
        rec = Recorder("task-first")
        pin(str(out), cl_env, cfg=cl_cfg, rec=rec)
        first = rec.calls[-1]
        check(
            "first launch creates a task with reuse_last_task_id=False (upstream's True DELETES "
            "the previous execution's logs)",
            len(rec.calls) == 1
            and first.get("reuse_last_task_id") is False
            and "continue_last_task" not in first,
            f"{rec.calls}",
        )
        check(
            "first launch records the task id inside output_dir",
            (out / Plan.CLEARML_ID_FILE).read_text().strip() == "task-first",
            f"{list(out.iterdir())}",
        )
        check(
            "the task carries the project and name the callback would have used",
            first.get("project_name") == "d" and first.get("task_name"),
            f"{first}",
        )
        check(
            "frameworks are NOT auto-connected, so clearml does not upload checkpoints",
            first.get("auto_connect_frameworks")
            == {"tensorboard": False, "pytorch": False},
            f"{first}",
        )
        check(
            "output_uri is left unset, so models do not default to the file server",
            "output_uri" not in first,
            f"{first}",
        )

        rec2 = Recorder("task-first")
        pin(str(out), cl_env, cfg=cl_cfg, rec=rec2)
        second = rec2.calls[-1]
        check(
            "a RELAUNCH into the same output_dir continues that task by id",
            len(rec2.calls) == 1 and second.get("reuse_last_task_id") == "task-first",
            f"{rec2.calls}",
        )
        check(
            "the relaunch passes continue_last_task=0 -- the int, which continues with NO "
            "iteration offset (True would replot resumed step 26 at 51)",
            second.get("continue_last_task") == 0
            and second["continue_last_task"] is not False,
            f"{second}",
        )

        # The failure path. What must NOT happen is a fall back to the callback's default, which
        # would reuse-and-delete; a fresh task loses continuity but never data.
        rec3 = Recorder("task-second", boom=True)
        pin(str(out), cl_env, cfg=cl_cfg, rec=rec3)
        check(
            "if continuation is refused, a NEW task is created with reuse_last_task_id=False "
            "rather than deleting the old one",
            len(rec3.calls) == 2 and rec3.calls[1].get("reuse_last_task_id") is False,
            f"{rec3.calls}",
        )
        check(
            "and the recorded id moves to the task that now exists",
            (out / Plan.CLEARML_ID_FILE).read_text().strip() == "task-second",
        )

        class Refuser:
            calls = 0

            def __call__(self, **kw):
                Refuser.calls += 1
                raise RuntimeError("simulated: server down")

        out2 = Path(td) / "ckpt-dir-2"
        a2 = ArgsWithOut(str(out2))
        a2.report_to = []
        resolve(
            cl_cfg,
            environ=dict(cl_env),
            home=Path("/nonexistent"),
            config_path="/c/cfg.yaml",
            is_installed=yes,
        ).apply(a2, dict(cl_env), task_init=Refuser())
        check(
            "Task.init failing -> clearml leaves report_to, so the callback cannot retry with "
            "upstream's reuse-and-delete defaults",
            "clearml" not in a2.report_to and Refuser.calls == 1,
            f"{a2.report_to} {Refuser.calls}",
        )

        rec4 = Recorder()
        pin(str(out), {**cl_env, "RANK": "3"}, cfg=cl_cfg, rec=rec4)
        check(
            "a non-zero RANK initialises nothing, so a 16-rank job makes 1 task and not 16",
            rec4.calls == [],
            f"{rec4.calls}",
        )

        rec5 = Recorder()
        pin(str(out), {**cl_env, "CLEARML_OFFLINE_MODE": "1"}, cfg=cl_cfg, rec=rec5)
        check(
            "offline mode does not try to continue a task that has no server-side id",
            rec5.calls == [],
            f"{rec5.calls}",
        )

        rec6 = Recorder()
        e6 = dict(cl_env)
        resolve(
            cl_cfg,
            environ={"WANDB_API_KEY": "k"},
            home=Path("/nonexistent"),
            config_path="/c/cfg.yaml",
            is_installed=yes,
        ).apply(FakeArgs(), e6, task_init=rec6)
        check(
            "no output_dir -> no task pinned, because there is no resume to continue across",
            rec6.calls == [],
            f"{rec6.calls}",
        )

    # 16. One tool per run, chosen explicitly. Every way of not choosing is refused, not guessed.
    both = {"clearml_project": "d", "wandb_entity": "e", "wandb_project": "p"}
    keyed_env = {**env, "WANDB_API_KEY": "k"}
    for tool in ("clearml", "wandb"):
        p = resolve(
            {**both, "tracking_tool": tool},
            environ=keyed_env,
            home=Path("/nonexistent"),
            is_installed=yes,
        )
        check(
            f"tracking_tool={tool} with the other tool's keys -> FATAL, nothing enabled",
            not p.enabled and any("One tool per run" in f for f in p.fatal),
            f"{p.enabled} {p.fatal}",
        )
    for tool in ("", "none", "NONE"):
        p = resolve(
            {"clearml_project": "d", "tracking_tool": tool},
            environ=keyed_env,
            home=Path("/nonexistent"),
            is_installed=yes,
        )
        check(
            f"tracking_tool={tool!r} with clearml keys set -> FATAL (not inferred)",
            not p.enabled and any("tracking_tool is not" in f for f in p.fatal),
            f"{p.fatal}",
        )
    p = resolve({"tracking_tool": "none"}, environ=keyed_env, home=Path("/nonexistent"))
    check(
        "tracking_tool=none alone -> off, no fatal",
        not p.enabled and not p.fatal,
        f"{p.fatal}",
    )
    p = resolve(
        {"tracking_tool": "mlflow"}, environ=keyed_env, home=Path("/nonexistent")
    )
    check(
        "an unknown tracking_tool -> FATAL",
        any("is not one of" in f for f in p.fatal),
        f"{p.fatal}",
    )
    p = resolve(
        {"tracking_tool": "WandB", "wandb_entity": "e", "wandb_project": "p"},
        environ=keyed_env,
        home=Path("/nonexistent"),
        is_installed=yes,
    )
    check("tracking_tool is case-insensitive", p.enabled == ("wandb",), f"{p.fatal}")

    # 19. Servers. Each tool gets its public server unless the user set another, and a user
    #     setting is never replaced -- the default goes into plan.env only when nothing set it.
    wb = {"tracking_tool": "wandb", "wandb_entity": "e", "wandb_project": "p"}
    with tempfile.TemporaryDirectory() as td:
        h = Path(td)
        p = resolve(wb, environ={"WANDB_API_KEY": "k"}, home=h, is_installed=yes)
        check(
            "wandb: no server set -> WANDB_BASE_URL=https://api.wandb.ai (not wandb's Forge default)",
            p.env.get("WANDB_BASE_URL") == "https://api.wandb.ai"
            and any("server=https://api.wandb.ai (default)" in l for l in p.lines),
            f"{p.env} {p.lines}",
        )
        p = resolve(
            wb,
            environ={
                "WANDB_API_KEY": "k",
                "WANDB_BASE_URL": "https://wandb.corp.example",
            },
            home=h,
            is_installed=yes,
        )
        check(
            "wandb: a user WANDB_BASE_URL is kept, not overwritten",
            "WANDB_BASE_URL" not in p.env
            and any("wandb.corp.example" in l for l in p.lines),
            f"{p.env} {p.lines}",
        )
        (h / ".config" / "wandb").mkdir(parents=True)
        (h / ".config" / "wandb" / "settings").write_text(
            "[default]\nbase_url = https://wandb.corp.example\n"
        )
        p = resolve(wb, environ={"WANDB_API_KEY": "k"}, home=h, is_installed=yes)
        check(
            "wandb: a base_url in wandb's settings file counts as set by the user",
            "WANDB_BASE_URL" not in p.env
            and any("wandb.corp.example" in l for l in p.lines),
            f"{p.env} {p.lines}",
        )
        (h / ".config" / "wandb" / "settings").unlink()
        for bad_url in (
            "https://forge.coreweave.com",
            "http://forge.coreweave.com/api/wandb",
            "https://forge.coreweave.com:443/api/wandb",
            "https://app.wandb.ai",
            "http://api.wandb.ai",
            "api.wandb.ai",
        ):
            p = resolve(
                wb,
                environ={"WANDB_API_KEY": "k", "WANDB_BASE_URL": bad_url},
                home=h,
                is_installed=yes,
            )
            check(
                f"wandb: an address wandb rejects is FATAL here, before training ({bad_url})",
                not p.enabled and any("wandb would refuse it" in f for f in p.fatal),
                f"{p.fatal}",
            )
        p = resolve(
            wb,
            environ={
                "WANDB_API_KEY": "k",
                "WANDB_BASE_URL": "https://forge.coreweave.com/api/wandb/",
            },
            home=h,
            is_installed=yes,
        )
        check(
            "wandb: the Forge proxy address is accepted as wandb normalises it",
            p.enabled == ("wandb",)
            and any("server=https://api.forge.coreweave.com" in l for l in p.lines),
            f"{p.lines} {p.fatal}",
        )
        p = resolve(
            wb,
            environ={"WANDB_API_KEY": "k", "WANDB_IDENTITY_TOKEN_FILE": "/t"},
            home=h,
            is_installed=yes,
        )
        check(
            "wandb: API key and identity token file together -> FATAL, as wandb refuses it",
            any("WANDB_IDENTITY_TOKEN_FILE" in f for f in p.fatal),
            f"{p.fatal}",
        )
        (h / "token.jwt").write_text("x")
        ok, where = wandb_credentials(
            {"WANDB_IDENTITY_TOKEN_FILE": str(h / "token.jwt")}, h
        )
        check(
            "wandb: an identity token file is credentials",
            ok and "IDENTITY_TOKEN" in where,
            where,
        )
        ok, _ = wandb_credentials(
            {"WANDB_IDENTITY_TOKEN_FILE": str(h / "missing.jwt")}, h
        )
        check("wandb: an identity token file that does not exist is not", not ok)

        cl = {"tracking_tool": "clearml", "clearml_project": "d"}
        cl_keys = {"CLEARML_API_ACCESS_KEY": "a", "CLEARML_API_SECRET_KEY": "b"}
        p = resolve(cl, environ=cl_keys, home=h, is_installed=yes)
        check(
            "clearml: this module sets no ClearML server, whatever clearml resolves",
            p.enabled == ("clearml",)
            and not any(
                k.startswith(("CLEARML_API_HOST", "CLEARML_WEB", "CLEARML_FILES"))
                for k in p.env
            ),
            f"{p.env}",
        )
        corp = {"public": False, "token": False, "access": True, "secret": True}
        p = resolve(
            cl,
            environ={},
            home=h,
            is_installed=yes,
            clearml_probe_fn=lambda e, hh: corp,
        )
        check(
            "clearml: a self-hosted server clearml resolves is reported and left alone",
            p.enabled == ("clearml",)
            and any(
                "server: self-hosted (from clearml's config or environment)" in l
                for l in p.lines
            )
            and "CLEARML_API_HOST" not in p.env,
            f"{p.lines} {p.env}",
        )
        for probe, want, name in (
            (
                {
                    "public": True,
                    "token": True,
                    "access": False,
                    "secret": False,
                },
                True,
                "CLEARML_AUTH_TOKEN alone",
            ),
            (
                {
                    "public": True,
                    "token": False,
                    "access": True,
                    "secret": True,
                },
                True,
                "both keys",
            ),
            (
                {
                    "public": True,
                    "token": False,
                    "access": True,
                    "secret": False,
                },
                False,
                "the access key alone",
            ),
            (
                {
                    "public": True,
                    "token": False,
                    "access": False,
                    "secret": True,
                },
                False,
                "the secret key alone",
            ),
            (
                {
                    "public": True,
                    "token": False,
                    "access": False,
                    "secret": False,
                },
                False,
                "nothing",
            ),
            ({"error": "boom"}, False, "an unreadable config"),
            (None, False, "no clearml"),
        ):
            got = clearml_verdict(probe)[0]
            check(f"clearml: {name} -> credentials {want}", got is want, f"{got}")
        p = resolve(
            cl,
            environ={},
            home=h,
            is_installed=yes,
            clearml_probe_fn=lambda e, hh: {
                "public": True,
                "token": False,
                "access": True,
                "secret": False,
            },
        )
        check(
            "clearml: one key -> off with NO CREDENTIALS, and no env left behind",
            not p.enabled and not p.env and any("only one of" in l for l in p.lines),
            f"{p.lines} {p.env}",
        )
        p = resolve(
            cl,
            environ={"CLEARML_LOG_MODEL": "TRUE"},
            home=h,
            is_installed=yes,
            clearml_probe_fn=lambda e, hh: corp,
        )
        check(
            "clearml: a user CLEARML_LOG_MODEL is not overridden",
            "CLEARML_LOG_MODEL" not in p.env,
            f"{p.env}",
        )
        p = resolve(
            cl,
            environ={},
            home=h,
            is_installed=yes,
            clearml_probe_fn=lambda e, hh: corp,
        )
        check(
            "clearml: CLEARML_LOG_MODEL defaults to FALSE",
            p.env.get("CLEARML_LOG_MODEL") == "FALSE",
            f"{p.env}",
        )

        # Offline: clearml's own variable and its TRAINS_ alias, with clearml's bool parsing.
        check(
            "clearml offline: TRAINS_OFFLINE_MODE counts",
            is_offline("clearml", {"TRAINS_OFFLINE_MODE": "1"}),
        )
        check(
            "clearml offline: CLEARML_ wins over TRAINS_",
            not is_offline(
                "clearml", {"CLEARML_OFFLINE_MODE": "0", "TRAINS_OFFLINE_MODE": "1"}
            ),
        )
        truthy = ("1", "true", "True", "yes", "y", "t", "on", "anything")
        falsy = ("", "0", "false", "no", "n", "f", "off")
        check(
            "clearml offline: strtobool's values, and any other non-empty string, are true",
            all(_truthy(v) for v in truthy) and not any(_truthy(v) for v in falsy),
            f"{[(v, _truthy(v)) for v in truthy + falsy]}",
        )

        # W&B details the first review's mutation pass found unpinned.
        (h / "cfgdir").mkdir()
        (h / "cfgdir" / "settings").write_text(
            "[default]\nbase_url = https://wandb.corp.example\n"
        )
        check(
            "wandb: $WANDB_CONFIG_DIR/settings is read for base_url",
            wandb_base_url({"WANDB_CONFIG_DIR": str(h / "cfgdir")}, h)
            == ("https://wandb.corp.example", str(h / "cfgdir" / "settings")),
            f"{wandb_base_url({'WANDB_CONFIG_DIR': str(h / 'cfgdir')}, h)}",
        )
        p = resolve(
            wb,
            environ={"WANDB_API_KEY": "k", "WANDB_BASE_URL": "https://api.wandb.ai/"},
            home=h,
            is_installed=yes,
        )
        check(
            "wandb: a trailing slash is normalised away, as wandb does",
            any(
                "server=https://api.wandb.ai (set by WANDB_BASE_URL)" in l
                for l in p.lines
            ),
            f"{p.lines}",
        )
        for bad_url in (
            " https://wandb.corp.example ",
            "https://wandb corp.example",
            "https://wandb.corp.example:99999",
        ):
            p = resolve(
                wb,
                environ={"WANDB_API_KEY": "k", "WANDB_BASE_URL": bad_url},
                home=h,
                is_installed=yes,
            )
            check(
                f"wandb: {bad_url!r} is refused, as wandb refuses it",
                bool(p.fatal) and not p.enabled,
                f"{p.fatal}",
            )
        p = resolve(wb, environ={}, home=h, is_installed=yes)
        check(
            "wandb: unkeyed -> off, and the default WANDB_BASE_URL is not left in plan.env",
            not p.enabled and not p.env,
            f"{p.env}",
        )
        p = resolve(wb, environ={"WANDB_MODE": "disabled"}, home=h, is_installed=yes)
        check(
            "wandb: WANDB_MODE=disabled -> off, said as disabled, not as NO CREDENTIALS",
            not p.enabled
            and not p.fatal
            and any("WANDB_MODE=disabled" in l for l in p.lines)
            and not any("NO CREDENTIALS" in l for l in p.lines),
            f"{p.lines}",
        )
        check(
            "clearml offline: an empty CLEARML_OFFLINE_MODE is unset, so TRAINS_OFFLINE_MODE counts",
            is_offline(
                "clearml", {"CLEARML_OFFLINE_MODE": "", "TRAINS_OFFLINE_MODE": "1"}
            )
            and is_offline(
                "clearml", {"CLEARML_OFFLINE_MODE": " ", "TRAINS_OFFLINE_MODE": "1"}
            ),
        )
        check(
            "wandb offline: WANDB_MODE is case-sensitive, as in wandb",
            not is_offline("wandb", {"WANDB_MODE": "Offline"})
            and is_offline("wandb", {"WANDB_MODE": "offline"}),
        )
        for bad_mode in ("Offline", "OFFLINE", " offline", "nope"):
            p = resolve(
                wb,
                environ={"WANDB_API_KEY": "k", "WANDB_MODE": bad_mode},
                home=h,
                is_installed=yes,
            )
            check(
                f"wandb: WANDB_MODE={bad_mode!r} is refused, as wandb refuses it",
                not p.enabled and any("not a mode wandb accepts" in f for f in p.fatal),
                f"{p.fatal}",
            )
        ws = h / "workspace"
        (ws / "wandb").mkdir(parents=True)
        (ws / "wandb" / "settings").write_text(
            "[default]\nbase_url = https://wandb.workspace.example\n"
        )
        (h / ".config" / "wandb").mkdir(parents=True, exist_ok=True)
        (h / ".config" / "wandb" / "settings").write_text(
            "[default]\nbase_url = https://wandb.global.example\n"
        )
        check(
            "wandb: the workspace settings file (<cwd>/wandb/settings) wins over the global one",
            wandb_base_url({}, h, cwd=ws)[0] == "https://wandb.workspace.example"
            and wandb_base_url({}, h, cwd=h / "elsewhere")[0]
            == "https://wandb.global.example",
            f"{wandb_base_url({}, h, cwd=ws)} {wandb_base_url({}, h, cwd=h / 'elsewhere')}",
        )
        (h / ".config" / "wandb" / "settings").write_text(
            "[default]\nbase_url = https://forge.coreweave.com\n"
        )
        p = resolve(wb, environ={"WANDB_API_KEY": "k"}, home=h, is_installed=yes)
        check(
            "wandb: an invalid base_url in a settings file is refused, saying wandb would ignore it",
            any("ignore the file" in f for f in p.fatal),
            f"{p.fatal}",
        )
        (h / ".config" / "wandb" / "settings").unlink()

        nh = h / "netrc-order"
        nh.mkdir()
        (nh / "_netrc").write_text("machine api.wandb.ai login u password p\n")
        check(
            "wandb: ~/_netrc is used when ~/.netrc is absent",
            wandb_netrc_path({}, nh) == nh / "_netrc",
        )
        (nh / ".netrc").write_text("machine other login u password p\n")
        check(
            "wandb: ~/.netrc wins over ~/_netrc",
            wandb_netrc_path({}, nh) == nh / ".netrc"
            and not wandb_credentials({}, nh)[0],
        )
        (nh / ".netrc").write_text("nonsense\n")
        ok, where = wandb_credentials({}, nh)
        check(
            "wandb: an unparsable netrc is not credentials, and says why",
            not ok and "cannot read" in where,
            where,
        )

    # 20. Netrc parity with wandb 0.31.2's own lookup (wbnetrc.read_netrc_auth_with_source),
    #     case by case. The netrc texts and server addresses are the #752 review's
    #     (netrc_diff.py); the expected letters were produced by running wandb 0.31.2 on each
    #     case with the same server this module uses (an unset WANDB_BASE_URL meaning
    #     https://api.wandb.ai): T credentials, F none, E wandb rejects the address.
    pw = "0123456789abcdef0123456789abcdef01234567"
    parity_netrcs = {
        "A api.wandb.ai": f"machine api.wandb.ai\n  login user\n  password {pw}\n",
        "B forge": f"machine api.forge.coreweave.com\n  login user\n  password {pw}\n",
        "C corp": f"machine wandb.corp.example\n  login user\n  password {pw}\n",
        "D default-only": f"default login user password {pw}\n",
        "E forge-no-pw + other-pw": "machine api.forge.coreweave.com\n  login user\nmachine github.com login x password y\n",
        "F suffix host": f"machine api.wandb.ai.evil.com login user password {pw}\n",
        "G quoted host": f'machine "api.wandb.ai" login user password {pw}\n',
        "H qa api": f"machine api.qa.wandb.ai login user password {pw}\n",
        "I forge with port": f"machine api.forge.coreweave.com:443 login user password {pw}\n",
        "J one-line forge": f"machine api.forge.coreweave.com login user password {pw}",
        "K both": f"machine api.wandb.ai login u password {pw}\nmachine api.forge.coreweave.com login u password {pw[:-1]}X\n",
        "L api.wandb.ai empty pw": 'machine api.wandb.ai\n  login user\n  password ""\n',
        "M forge no pw, wandb.ai pw": f"machine api.forge.coreweave.com login u\nmachine api.wandb.ai login u password {pw[:-1]}Y\n",
    }
    parity_bases = [
        None,
        "https://api.wandb.ai",
        "https://api.wandb.ai/",
        "https://api.forge.coreweave.com",
        "https://forge.coreweave.com/api/wandb",
        "https://forge.coreweave.com/api/wandb/",
        "https://qa.forge.coreweave.com/api/wandb",
        "https://api.qa.forge.coreweave.com",
        "https://api.qa.wandb.ai",
        "https://wandb.corp.example",
        "https://api.forge.coreweave.com:443",
        "http://forge.coreweave.com/api/wandb",
        "https://forge.coreweave.com",
        "https://forge.coreweave.com:443/api/wandb",
    ]
    parity_expected = {
        "A api.wandb.ai": "TTTTTTFFFFTEEE",
        "B forge": "FFFTTTFFFFFEEE",
        "C corp": "FFFFFFFFFTFEEE",
        "D default-only": "TTTTTTTTTTTEEE",
        "E forge-no-pw + other-pw": "FFFFFFFFFFFEEE",
        "F suffix host": "FFFFFFFFFFFEEE",
        "G quoted host": "TTTTTTFFFFTEEE",
        "H qa api": "FFFFFFTTTFFEEE",
        "I forge with port": "FFFFFFFFFFTEEE",
        "J one-line forge": "FFFTTTFFFFFEEE",
        "K both": "TTTTTTFFFFTEEE",
        "L api.wandb.ai empty pw": "FFFFFFFFFFFEEE",
        "M forge no pw, wandb.ai pw": "TTTFFFFFFFTEEE",
    }
    with tempfile.TemporaryDirectory() as td:
        h = Path(td)
        wrong = []
        for name, text in parity_netrcs.items():
            (h / "netrc").write_text(text)
            got = ""
            for base in parity_bases:
                e = {"NETRC": str(h / "netrc")}
                if base is not None:
                    e["WANDB_BASE_URL"] = base
                try:
                    wandb_normalize_base_url(base or WANDB_DEFAULT_BASE_URL)
                    got += "T" if wandb_credentials(e, h)[0] else "F"
                except ValueError:
                    got += "E"
            if got != parity_expected[name]:
                wrong.append(f"{name}: got {got}, wandb {parity_expected[name]}")
        check(
            f"netrc lookup agrees with wandb 0.31.2 on all {len(parity_netrcs) * len(parity_bases)} cases",
            not wrong,
            "; ".join(wrong),
        )

    if not bad:
        print(
            "\nTRACKING RESOLUTION EXERCISED: the tool choice, on, off, partial-fatal, unkeyed, "
            "offline, the default servers and user-set ones, auto-name (both sources), "
            "credential detection on disk, netrc parity with wandb 0.31.2, apply(), the "
            "output_dir-pinned W&B run id, and output_dir-keyed ClearML task continuation"
        )
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
