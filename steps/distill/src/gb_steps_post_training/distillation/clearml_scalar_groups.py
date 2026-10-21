"""Route training scalars into SEPARATE ClearML plots instead of one unreadable "train" plot.

THE PROBLEM, measured directly (the bucket100 arm). transformers'
`ClearMLCallback.on_log` sends every key that is not `eval_*`/`test_*` to a single plot titled
"train", one series per key. On this trainer that is 11 series in one set of axes:

    grad_norm       [0.171682,  3.12516]     loss            [0.0402896, 0.211795]
    learning_rate   [0,         1e-05]       off_policy_loss [0.0403,    0.2118]
    n_sync_params   [0,         0]           t_train         [9.4,       43.2]
    t_gen           [0,         0]           time            [9.55,      3004.4]
    t_sync/_ag/_bc  [0,         0]

Five orders of magnitude between `learning_rate` (1e-5) and `time` (3004). Autoscaling makes the
plot a flat line at zero with one spike, and NOTHING in it is readable -- not the loss curve that
says whether the run is learning, not the grad-norm that says whether it is stable.

WHAT THIS DOES. Classifies each key by what it MEASURES and titles the plot accordingly, so each
plot holds series that share a unit and a scale:

    loss        loss, off_policy_loss, on_policy_loss, matched_loss, unmatched_loss,
                hidden_loss, sampled_opd_loss           -- all a divergence, all ~0.0-1.0
    lr          learning_rate                            -- alone, because 1e-5 shares no axis
    grad_norm   grad_norm                                -- alone, same reason
    time        time, t_train, t_gen, t_sync, t_sync_ag, t_sync_bc, t_gen_actual,
                t_gen_wait, t_train_pure                 -- all SECONDS, deliberately together
    counters    n_sync_params, num_tokens, n_gen_calls, n_gen_batch, n_gen_tokens,
                n_gen_truncated                          -- integer counts
    gen         gen_truncated_pct, gen_tokens_per_row, gen_tokens_per_s
                                                         -- the SHAPE of what generation produced,
                                                            as opposed to what it cost in seconds.
                                                            ~10^1-10^3 across the three, the same
                                                            two-order spread `time` already carries
                                                            on purpose, and they are read together:
                                                            a per-row length pinned at the cap with
                                                            a high truncated_pct is the finding,
                                                            and tokens_per_s is what a bigger batch
                                                            is supposed to move. Truncation is a
                                                            PERCENT, not a 0-1 fraction, so it does
                                                            not vanish against the other two
    train       anything unrecognised                    -- today's behaviour, so a new key is
                                                            never silently dropped

WHY `time` KEEPS THE WHOLE ALONGSIDE ITS PARTS. `time` is wall-clock since the previous log
(custom_gold_trainer.log:3707) and the `t_*` are phases within it, so `time - (t_train + t_gen)` is
unaccounted overhead -- which is the single most useful quantity this project measures, and it is
only visible if the whole and the parts share an axis. They share the unit, so the scale argument
above does not apply. The first step is a real outlier (3004 s: transformers' length-grouped sampler
deliberately puts the longest megabatch first) and it will compress the y-axis until you deselect it
or switch to log; that is a property of the data, not of the grouping.

WHAT IS DELIBERATELY LEFT TO UPSTREAM. `eval_*`, `test_*` and the six "single value" keys
(train_runtime, train_samples_per_second, train_steps_per_second, train_loss, total_flos, epoch --
reported once via `report_single_value`, not as a curve) are handed to `ClearMLCallback.on_log`
unchanged. Those are the branches most likely to change between transformers releases, so this
module does not reimplement them; it partitions the dict and delegates. The only thing it owns is
the title of a train-time curve. A companion check asserts the single-value list
here still matches the one in the installed transformers, so the partition cannot drift silently.

READERS OF THESE SERIES MUST NOT HARD-CODE "train". A companion analysis tool
did (`sc["train"]["off_policy_loss"]`) and would have returned zeros -- reporting
`lmbda_realized: None` for every run -- rather than failing. It now searches across titles and
falls back to "train" so the six tasks logged BEFORE this change keep reading correctly. Any new
consumer should use `find_series()` below for the same reason.

Stdlib only at module scope, on purpose: the routing is a pure function of a string, and the check
script exercises it on the login node where transformers/clearml are not importable.
"""
from __future__ import annotations

# Mirrors the local list in transformers.integrations.integration_utils.ClearMLCallback.on_log.
# Duplicated because it is a local variable there and cannot be imported; pinned by
# a companion check that parses the installed source and fails on drift.
SINGLE_VALUE_SCALARS = (
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
    "train_loss",
    "total_flos",
    "epoch",
)

#: Plot titles this module can produce, for the check script and for consumers to iterate.
TITLES = ("loss", "lr", "grad_norm", "time", "counters", "gen", "uld", "train")

_TIME_SUFFIXES = ("_time", "_seconds", "_secs", "_latency")
_COUNTER_SUFFIXES = ("_count", "_params", "_tokens")


def scalar_destination(key: str) -> tuple[str, str]:
    """Map a log key to the (plot title, series name) it should be reported under.

    Pure, total, and order-sensitive: the first matching rule wins. Only ever called for keys the
    caller has already decided are train-time curves -- `eval_*`, `test_*` and the single-value
    keys never reach here.

    A leading "train/" is stripped before classifying. The trainer emits `train/is_weight_mean` and
    friends (custom_gold_trainer.log:3775) with the namespace baked into the key, which upstream
    would render as a series literally named "train/is_weight_mean" inside a plot named "train".
    Stripping it lets those keys be classified like any other and keeps the series name readable.
    """
    series = key[len("train/"):] if key.startswith("train/") else key

    # FIRST, deliberately, and the ordering is the whole decision. The ULD diagnostics
    # (`uld/span_mismatch_rows`, `uld/regions_per_row`, `uld/relaxed_boundaries`,
    # `uld/scaffold_{tokens,regions}_trimmed`, `uld/region_chunks`) are read together or not at all:
    # a scaffold-trim count that drops to zero only means something next to the regions-per-row it
    # should have moved with. Scattering them across `counters` and `train` by suffix is what put
    # them nowhere anyone looked. Matching before the loss rule means a future `uld/*_loss` key
    # would land here rather than on the `loss` plot; that is the accepted cost, and no such key
    # exists today (the ULD loss components are emitted as bare `matched_loss`/`unmatched_loss`).
    #
    # The prefix is KEPT in the series name, unlike `train/` above, and that asymmetry is forced
    # rather than chosen. This function must be idempotent -- `scalar_destination(series)` has to
    # return the same destination as `scalar_destination(key)` -- because find_series() classifies
    # whatever name a caller hands it, which is usually the series name it read off the plot.
    # `train/` survives stripping only by luck: the bare name falls through to the `train` fallback,
    # which is where it already was. `uld/` has no such luck -- a bare `region_chunks` matches no
    # rule and lands on `train`, so stripping would make the plot unreachable by its own labels.
    # a companion check's stage 5 rejected the stripped version for exactly this.
    if series.startswith("uld/"):
        return "uld", series

    if series == "loss" or series.endswith("_loss") or series.startswith("loss_"):
        return "loss", series
    if series in ("learning_rate", "lr"):
        return "lr", series
    if series == "grad_norm" or series.endswith("_grad_norm"):
        return "grad_norm", series
    # `time` exactly, or a `t_`-prefixed phase timer, or an explicit time-unit suffix.
    if series == "time" or series.startswith("t_") or series.endswith(_TIME_SUFFIXES):
        return "time", series
    # Before the counters rule only for readability -- the two cannot collide, because a `gen_`
    # prefix is not an `n_`/`num_` prefix and none of the gen keys carry a counter suffix. The
    # generation COUNTS are deliberately `n_gen_*` so they land in `counters` with the other
    # integers, and only the derived ratios land here.
    if series.startswith("gen_"):
        return "gen", series
    if series.startswith("n_") or series.startswith("num_") or series.endswith(_COUNTER_SUFFIXES):
        return "counters", series
    return "train", series


def partition(logs: dict) -> tuple[dict, dict]:
    """Split a `logs` dict into (ours, upstream's).

    Ours: numeric train-time curves we re-title. Upstream's: everything else -- non-numeric values
    (so upstream emits its own warning rather than us swallowing it), the `eval_`/`test_` prefixes,
    and the single-value keys.
    """
    ours, theirs = {}, {}
    for k, v in (logs or {}).items():
        numeric = isinstance(v, (int, float)) and not isinstance(v, bool)
        if numeric and k not in SINGLE_VALUE_SCALARS and not k.startswith(("eval_", "test_")):
            ours[k] = v
        else:
            theirs[k] = v
    return ours, theirs


def find_series(scalars: dict, name: str) -> dict:
    """Look up one series by name across plot titles, newest layout first, then legacy "train".

    For consumers of `Task.get_reported_scalars()`. Exists because the title a series lives under
    is now a function of this module's rules and changed once already: tasks logged before it kept
    everything under "train". A consumer that hard-codes either layout reads zeros on tasks from
    the other one, silently.
    """
    title, series = scalar_destination(name)
    # Series-name candidates, not just title candidates. The title fallback covers tasks logged
    # before this module existed; the bare-name candidate covers a narrower, dated window: jobs
    # 1706331 and 1706464 (the two ULD arms) started minutes before the `uld/` rule stopped
    # stripping its prefix, so their `uld` plot carries series named `region_chunks` rather than
    # `uld/region_chunks`. Both spellings are tried so a consumer asking the current way still
    # finds them. Cheap, total, and it removes the only reason to restart two 27-hour runs over a
    # label.
    names = [series]
    if "/" in series:
        names.append(series.rsplit("/", 1)[1])
    for t in (title, "train"):
        for s in names:
            got = (scalars.get(t) or {}).get(s)
            if got:
                return got
    return {}


def install(trainer, emit=print) -> bool:
    """Swap the auto-added ClearMLCallback for the grouping subclass. Returns True if swapped.

    Called after the Trainer is constructed (it adds the callback itself from `report_to`) and
    before `train()`, because `ClearMLCallback.setup` runs lazily on the first log or train_begin.

    ORDER MATTERS: remove before add. `remove_callback` matches by class, and the replacement IS a
    ClearMLCallback subclass, so adding first would remove both.
    """
    try:
        from transformers.integrations.integration_utils import ClearMLCallback
    except Exception as exc:                                    # transformers without the extra
        emit(f"[scalar-groups] not installed: cannot import ClearMLCallback ({exc})")
        return False

    present = [cb for cb in trainer.callback_handler.callbacks
               if isinstance(cb, ClearMLCallback)]
    if not present:
        # report_to has no "clearml" -- nothing to regroup, and saying so beats silence when
        # someone is looking for grouped plots that never appear.
        emit("[scalar-groups] no ClearMLCallback on the trainer (report_to has no 'clearml'); "
             "nothing to regroup")
        return False

    grouped = _grouped_callback_class(ClearMLCallback)
    for cb in present:
        trainer.remove_callback(cb)
    trainer.add_callback(grouped())
    emit(f"[scalar-groups] ClearML train scalars will be split across plots: "
         f"{', '.join(TITLES)}")
    return True


def _grouped_callback_class(base):
    """Build the subclass against the installed base class (kept out of module scope so importing
    this module needs no transformers)."""

    class GroupedClearMLCallback(base):
        """Re-titles train-time scalars; delegates everything else to the base implementation."""

        def on_log(self, args, state, control, model=None, processing_class=None, logs=None,
                   **kwargs):
            ours, theirs = partition(logs)

            # Always delegate, even with an empty dict: the base method owns the `self._clearml is
            # None` guard, the lazy `setup()` call and the world-process-zero check, and we need all
            # three to have happened before touching the task below.
            super().on_log(args, state, control, model=model, processing_class=processing_class,
                           logs=theirs, **kwargs)

            if getattr(self, "_clearml", None) is None or not ours:
                return
            if not state.is_world_process_zero:
                return
            task = getattr(self, "_clearml_task", None)
            if task is None:
                return

            logger = task.get_logger()
            suffix = getattr(base, "log_suffix", "")
            for key, value in ours.items():
                title, series = scalar_destination(key)
                logger.report_scalar(title=title + suffix, series=series, value=value,
                                     iteration=state.global_step)

    return GroupedClearMLCallback
