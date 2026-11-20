# SPDX-License-Identifier: Apache-2.0

"""Turn one lm-engine checkpoint rung into a verified Hugging Face model directory.

``recipes/pretrain`` runs this once per rung, on a CPU pod, on the shared lm-engine
image::

    train --> export-N --> eval-<name>-ckN --> [eval-combined]

The exported directory is then served by vLLM and scored by the eval bridge, so it is
the one artifact in the recipe that every downstream number depends on.

STAGES

Both conversion stages are *upstream* code, run here rather than reimplemented:

1. ``lm_engine.training.unshard`` reads the sharded ``global_step<N>/`` tree and writes
   an unsharded lm-engine ``gpt_base`` checkpoint. It runs **single-process** -- no
   ``torchrun``, no process group (see :func:`run_unshard`).
2. ``tools.model_conversion.export_to_huggingface`` rewrites that as a ``granite`` /
   ``llama`` / ``granitemoehybrid`` model transformers and vLLM can load. ``tools`` is
   not part of the installed ``lm_engine`` package: it lives in the source tree at
   ``/app/src`` on ``PYTHONPATH``, which is why the image keeps the clone.
3. The tokenizer is copied in, byte for byte, *after* the conversion.
4. :mod:`export_checks` compares the written ``config.json`` against the values the run
   trained with, and proves the tokenizer files are byte-identical.
5. :func:`check_parity` loads both models on the CPU in fp32 and compares logits.
6. ``export.json`` records what was exported and how it verified.
7. The unsharded checkpoint is deleted -- *only* once everything above passed.

WHY THE VERIFICATION IS THIS PARANOID

Each stage above guards an incident this project has already paid for; the long version
is in :mod:`export_checks`. In short: a granite-4 model trained at RoPE theta=1e7 was
exported without ``rope_theta``, served at the 1e4 default, and scored 0.000 on a
benchmark whose baseline was 0.8075 -- while producing fluent text. And a tokenizer
re-saved through ``save_pretrained`` leaked the trainer's runtime truncation setting
into ``tokenizer.json`` and dropped a trained ``Split`` pre-tokenizer, understating
every eval recorded against it.

The parity check is the backstop for whatever the config checks cannot see. Upstream's
own ``tests/model_conversion_test.py`` holds the converted and the original model to
``atol=3e-7`` in fp32 (``2.5e-5`` for ``granitemoehybrid``), so the step's default
``parity_tolerance`` of 0.05 is loose by four orders of magnitude on purpose: it is
there to catch a *wrong* model, not to police float arithmetic.

ORDERING, AND WHY STAGE 3 COMES AFTER STAGE 2

``export_to_huggingface`` ends with ``tokenizer.save_pretrained(save_path,
legacy_format=False)`` inside a bare ``try``/``except``, so by the time this step's own
copy runs, upstream has already written either a re-saved tokenizer or (if it could not
load one from the unsharded dir, which is the normal case -- lm-engine's
``save_pretrained`` writes only ``config.json`` and safetensors) none at all. The copy
therefore runs last and overwrites, and :func:`export_checks.check_tokenizer` proves
byte-identity afterwards rather than taking it on trust.

THE ARTIFACT LINE IS LOAD-BEARING

gbserver binds the target's ``hf_model`` output by scraping::

    GB_ARTIFACT_ID:hf_model GB_ARTIFACT_PATH:<dir>

from this step's stdout. It is printed exactly once, from one place
(:func:`artifact_line`), and only after every check and the parity comparison have
passed. Printing it for an unverified export would publish a broken model to every
downstream eval, which is precisely the failure mode above.

IDEMPOTENCE

A re-run whose ``export.json`` records the same ``iteration``, lm-engine sha and
``model_type`` *and* a passing parity check prints the artifact line and exits. Any
other state -- a mismatch on any of the three, a recorded parity failure, a manifest
that will not parse -- re-exports. The bias is deliberate: re-running the export costs
minutes on a CPU pod, and trusting a manifest that does not describe the model on disk
costs a whole eval sweep.

STRUCTURE

One function per stage, a thin :func:`main` that sequences them, and every heavy import
(``torch``, ``transformers``, ``lm_engine``, ``tools``) *inside* the function that needs
it. The module therefore imports on a bare interpreter, which keeps the orchestration
tests fast and honest -- they monkeypatch the stages and assert the sequencing.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import traceback
from typing import Sequence

import export_checks

__all__ = [
    "ARTIFACT_ID",
    "MANIFEST_NAME",
    "artifact_line",
    "check_parity",
    "convert",
    "copy_tokenizer",
    "lm_engine_sha",
    "main",
    "run_checks",
    "run_unshard",
    "unsharded_dir",
    "work_dir",
]

#: The output name the step declares and gbserver binds.
ARTIFACT_ID = "hf_model"

#: Provenance record, written into the exported directory.
MANIFEST_NAME = "export.json"

#: The lm-engine source tree in the family image. ``tools/`` is not installed, so the
#: image keeps the clone here and puts it on ``PYTHONPATH``; it is also where the
#: pinned sha comes from.
LM_ENGINE_SOURCE = "/app/src"

#: The architectures ``tools.model_conversion`` can export to
#: (``_MODEL_EXPORT_FUNCTIONS``). Anything else raises ``NotImplementedError`` deep in
#: upstream, so it is refused by the argument parser instead.
MODEL_TYPES = ("granite", "granitemoehybrid", "llama")

#: ``global_step<N>`` -- lm-engine's ``_get_checkpoint_tag``.
CHECKPOINT_TAG = "global_step"

_LATEST = "latest_checkpointed_iteration.json"

#: Shape of the parity batch. Small enough to run on a CPU pod in seconds, wide enough
#: that a transposed or mis-permuted weight cannot hide in it.
PARITY_ROWS = 4
PARITY_COLUMNS = 128

#: Seed for the parity ids, fixed so two runs of the same export compare the same
#: tokens and the recorded max delta is reproducible.
PARITY_SEED = 0


# ----------------------------------------------------------------- paths, names


def work_dir(hf_dir: pathlib.Path) -> pathlib.Path:
    """Scratch directory for the export, a sibling of the output directory.

    A sibling rather than a subdirectory: the output directory *is* the artifact, and a
    half-converted lm-engine checkpoint inside it would be served to vLLM along with
    the model. It is removed on success and kept on any failure, because an unsharded
    checkpoint is the only thing left to debug a conversion with.
    """
    return hf_dir.parent / f"{hf_dir.name}.export-work"


def unsharded_dir(hf_dir: pathlib.Path) -> pathlib.Path:
    """Where ``lm_engine.training.unshard`` writes its output."""
    return work_dir(hf_dir) / "unsharded"


def artifact_line(hf_dir: pathlib.Path) -> str:
    """The one line gbserver scrapes to bind the ``hf_model`` output."""
    return f"GB_ARTIFACT_ID:{ARTIFACT_ID} GB_ARTIFACT_PATH:{hf_dir}"


def _log(message: str) -> None:
    """Stage logging, on stderr.

    stdout carries the artifact line, and a marker-shaped string in a log message is a
    binding gbserver would honour, so progress never goes there.
    """
    print(f"export: {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- stage 0: input


def resolve_iteration(train_dir: pathlib.Path, iteration: str | int | None) -> int:
    """Which rung to export, and prove the train_dir actually holds it.

    An empty ``iteration`` means "the latest rung present", read from
    ``latest_checkpointed_iteration.json`` -- the marker lm-engine's ``save_checkpoint``
    writes *last*, so it means the rung is complete rather than merely started.

    A rung the train_dir does not hold is refused here rather than three minutes later
    inside upstream: a recipe whose ladder outran its save interval would otherwise
    produce an export target that fails with a stack trace from ``load_yaml``.
    """
    if iteration in (None, ""):
        marker = train_dir / _LATEST
        try:
            resolved = json.loads(marker.read_text())["latest_checkpointed_iteration"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(
                f"no iteration given and {marker} could not be read ({error}); "
                "either the training run has written no complete checkpoint yet or "
                "train_dir is wrong"
            ) from error
    else:
        resolved = iteration
    try:
        resolved = int(resolved)
    except (TypeError, ValueError) as error:
        raise ValueError(f"not a checkpoint iteration: {resolved!r}") from error

    checkpoint = train_dir / f"{CHECKPOINT_TAG}{resolved}"
    if not checkpoint.is_dir():
        raise ValueError(
            f"{train_dir} holds no {CHECKPOINT_TAG}{resolved}/ "
            f"(looked for {checkpoint})"
        )
    return resolved


def load_expected(path: str | None) -> dict:
    """The training-side values the exported config must still carry.

    The step's ``config.expected`` reaches here as a JSON file written by the ``run:``
    block. Absent means an empty expectation, which :func:`export_checks.check_config`
    treats as "check nothing"; present but unreadable is an error, never silently
    empty -- a typo'd path would otherwise turn the whole config check off.
    """
    if not path:
        return {}
    text = pathlib.Path(path).read_text()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(
            f"{path} does not hold a JSON object (got {type(value).__name__})"
        )
    return value


def lm_engine_sha(source_dir: str = LM_ENGINE_SOURCE) -> str | None:
    """The pinned lm-engine commit, or ``None`` if it cannot be determined.

    The family image keeps the clone at ``/app/src`` (``tools/`` is not installed, so it
    has to), which makes ``git rev-parse HEAD`` the cheapest honest answer. Outside the
    image -- a laptop, or an image stripped of its ``.git`` -- there is no answer, and
    ``None`` is recorded rather than guessed. It is part of the idempotence key, so a
    manifest written with a sha and re-read without one re-exports, which is the safe
    direction.
    """
    if not pathlib.Path(source_dir).is_dir():
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", source_dir, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


# -------------------------------------------------------------- stage 1: unshard


def write_unshard_config(
    path: pathlib.Path,
    train_dir: pathlib.Path,
    iteration: int,
    unsharded: pathlib.Path,
    dtype: str,
) -> pathlib.Path:
    """Write the ``UnshardingArgs`` config ``lm_engine.training.unshard`` reads.

    The schema is ``lm_engine/training/arguments/__init__.py::UnshardingArgs``:
    ``load_args`` (``LoadArgs``: ``load_path`` required, ``iteration`` optional),
    ``unsharded_path`` (required), and an optional ``mixed_precision_args.dtype`` that
    overrides the dtype recorded in the training config. Only those are set; every other
    field has a default, and ``extra="forbid"`` on ``BaseArgs`` means an invented key is
    a hard error.

    Written as JSON into a ``.yml`` file on purpose. Upstream reads it with
    ``yaml.safe_load``, for which JSON is a valid subset, and emitting it with
    :mod:`json` keeps this module free of a PyYAML import -- it has to load on a bare
    interpreter for the orchestration tests.
    """
    config = {
        "load_args": {"load_path": str(train_dir), "iteration": iteration},
        "mixed_precision_args": {"dtype": dtype},
        "unsharded_path": str(unsharded),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))
    return path


def run_unshard(config_path: pathlib.Path) -> None:
    """Run ``python -m lm_engine.training.unshard --config <config_path>``.

    **Single process, no torchrun.** Verified in the pinned tree
    (a9c6e2b81ec33143419f82309706929195a740c0):

    * ``lm_engine/training/unshard.py::main`` passes
      ``barrier=ProcessGroupManager.is_initialized()`` -- i.e.
      ``torch.distributed.is_initialized()``, ``False`` here -- so it asks for no
      barrier of its own.
    * ``run_rank_n`` (``training/parallel/manager.py:495``) returns the function's
      result directly when ``ProcessGroupManager.get_global_rank()`` is ``None``, and it
      *is* ``None``: the module-level ``_DENSE_MESH = _Mesh()`` has ``rank = None``
      until ``ProcessGroupManager(...)`` is constructed, which only the training entry
      points do.
    * ``load_checkpoint_and_unshard`` reads the shards with ``no_dist=True`` under
      ``set_dummy_tensor_parallel_rank(0)`` / ``..._world_size(1)`` and the pipeline
      equivalents, so it is explicitly written for the no-process-group case.
    * upstream's own launcher, ``scripts/common/unshard.sh``, is exactly
      ``python -m lm_engine.training.unshard --config ${1}``.

    A subprocess rather than an in-process call because upstream's ``get_args`` parses
    ``sys.argv`` and ``set_logger`` reconfigures logging globally; neither belongs in
    this process. Its output is inherited, so unshard's own logs land in the step log.
    """
    command = [
        sys.executable,
        "-m",
        "lm_engine.training.unshard",
        "--config",
        str(config_path),
    ]
    _log(f"stage 1/5 unshard: {' '.join(command)}")
    subprocess.run(command, check=True)


# -------------------------------------------------------------- stage 2: convert


def convert(unsharded: pathlib.Path, model_type: str, hf_dir: pathlib.Path) -> None:
    """Run ``export_to_huggingface(unsharded, model_type, save_path=hf_dir)``.

    Signature confirmed at ``tools/model_conversion/__init__.py:67``::

        def export_to_huggingface(
            pretrained_model_name_or_path: str, model_type: str, save_path: str | None = None
        ) -> tuple[GPTBaseConfig, GenerationConfig, AutoTokenizer, dict]

    (``tools/export_to_hf.py`` in the same tree shows a different order and a module
    that no longer exists -- it is stale; the function above is what
    ``tests/model_conversion_test.py`` calls.) It writes ``config.json``,
    ``generation_config.json`` and the safetensors into ``save_path``, and the return
    value is not needed: the step verifies what landed on disk, not what the exporter
    says it wrote.

    It also *tries* to write a tokenizer, from the unsharded directory, in a bare
    ``try``/``except``. lm-engine's own ``save_pretrained`` puts no tokenizer there, so
    in practice none is written -- but stage 3 overwrites whatever it did either way.
    """
    from tools.model_conversion import export_to_huggingface

    _log(f"stage 2/5 convert: {unsharded} -> {hf_dir} as {model_type}")
    hf_dir.mkdir(parents=True, exist_ok=True)
    export_to_huggingface(str(unsharded), model_type, save_path=str(hf_dir))


# ------------------------------------------------------------- stage 3: tokenizer


def copy_tokenizer(src: pathlib.Path, dst: pathlib.Path) -> list[str]:
    """Copy the trained tokenizer files into the exported model, byte for byte.

    ``shutil.copy2`` and never ``save_pretrained``: re-saving a loaded tokenizer leaked
    the trainer's runtime ``truncation.max_length`` into ``tokenizer.json`` and dropped
    a trained ``Split`` pre-tokenizer, and every eval recorded against that model was
    understated.

    Any tokenizer file already in ``dst`` that ``src`` does not have is removed first.
    Those can only have come from upstream's own ``save_pretrained`` call (or from an
    earlier export of a different tokenizer), and leaving one behind would ship a
    synthesised file next to the copied ones -- which
    :func:`export_checks.check_tokenizer` would then, correctly, fail the export over.

    Returns the names copied, for the manifest.
    """
    dst.mkdir(parents=True, exist_ok=True)
    available = {
        name for name in export_checks.TOKENIZER_FILES if (src / name).is_file()
    }
    for name in export_checks.TOKENIZER_FILES:
        if name in available:
            continue
        stale = dst / name
        if stale.is_file():
            _log(f"stage 3/5 tokenizer: removing {name}, which {src} does not have")
            stale.unlink()
    copied = []
    for name in sorted(available):
        shutil.copy2(src / name, dst / name)
        copied.append(name)
    _log(
        f"stage 3/5 tokenizer: copied {len(copied)} file(s) from {src}: {', '.join(copied)}"
    )
    return copied


# ---------------------------------------------------------------- stage 4: checks


def run_checks(
    hf_dir: pathlib.Path, tokenizer_dir: pathlib.Path, expected: dict
) -> list[str]:
    """Run every :mod:`export_checks` entry point and return all problems found.

    Both halves run even when the first finds something: one failed export should tell
    the user everything that is wrong with it, not one thing per re-run.
    """
    config_path = hf_dir / "config.json"
    try:
        cfg = json.loads(config_path.read_text())
    except (OSError, ValueError) as error:
        return [
            f"the exported {config_path} could not be read ({error}); "
            "export_to_huggingface wrote no usable config"
        ]
    if not isinstance(cfg, dict):
        return [f"the exported {config_path} does not hold a JSON object"]

    errors = export_checks.check_config(cfg, expected)
    errors += export_checks.check_tokenizer(tokenizer_dir, hf_dir, cfg)
    return errors


# ---------------------------------------------------------------- stage 5: parity


def check_parity(unsharded: pathlib.Path, hf_dir: pathlib.Path) -> float:
    """Return ``max |logits_lm_engine - logits_hf|`` over a fixed batch, on CPU in fp32.

    The config checks can only see what the config *states*; this sees the weights. It
    mirrors upstream's ``tests/utils.py::model_conversion_test``, which builds both
    models, runs one forward pass through each and compares logits -- the difference
    being that the models here come off disk rather than out of a fixture.

    The lm-engine side loads through the API upstream documents for exactly this
    (``lm_engine/training/hf_adapter.py:120``)::

        hf_model = LLMAdapter_HF.from_pretrained(path, config=config)

    ``config`` is required and must be an lm-engine ``CommonConfig``.
    ``AutoConfig.from_pretrained`` returns one for an unsharded directory because
    ``import lm_engine.training`` runs ``register_model_classes()``, which registers
    ``GPTBaseConfig`` and friends under their ``model_type``; upstream's own
    ``export_to_huggingface`` depends on that same behaviour (it reaches the config
    through ``download_repo`` -> ``AutoConfig.from_pretrained``).

    Both sides are loaded in fp32 even though the unsharded checkpoint is written in
    bf16: comparing in the training dtype would fold a ~1e-2 rounding floor into the
    measurement and leave no room between "bit-level noise" and "wrong model".
    """
    import lm_engine.training  # noqa: F401  # registers the lm_engine model classes
    import torch
    from lm_engine.training.hf_adapter import LLMAdapter_HF
    from transformers import AutoConfig, AutoModelForCausalLM

    _log(f"stage 5/5 parity: loading {unsharded} and {hf_dir} on cpu in fp32")
    lm_engine_config = AutoConfig.from_pretrained(str(unsharded))
    lm_engine_model = LLMAdapter_HF.from_pretrained(
        str(unsharded), config=lm_engine_config, dtype=torch.float32
    )
    lm_engine_model.eval()

    hf_model = AutoModelForCausalLM.from_pretrained(str(hf_dir), dtype=torch.float32)
    hf_model.eval()

    vocab_size = int(lm_engine_config.vocab_size)
    torch.manual_seed(PARITY_SEED)
    input_ids = torch.randint(0, vocab_size, (PARITY_ROWS, PARITY_COLUMNS))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        lm_engine_logits = lm_engine_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits.float()
        hf_logits = hf_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits.float()

    return float((lm_engine_logits - hf_logits).abs().max().item())


# -------------------------------------------------------------- stage 6: manifest


def read_manifest(hf_dir: pathlib.Path) -> dict | None:
    """The previous ``export.json``, or ``None`` if there is not a usable one.

    Every unreadable shape -- absent, unparseable, not an object -- collapses to
    ``None`` and therefore to "export again". A manifest is a claim about the files
    next to it; a claim that cannot be read is worth nothing, and crashing on one would
    make a half-written directory unrecoverable without manual deletion.
    """
    try:
        value = json.loads((hf_dir / MANIFEST_NAME).read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def already_exported(
    hf_dir: pathlib.Path, iteration: int, sha: str | None, model_type: str
) -> bool:
    """Whether the directory already holds this exact verified export.

    All four must agree: the three identity fields *and* a recorded parity pass. A
    manifest that records a parity failure, or that predates the parity check and so has
    no flag, is not a verified export.
    """
    record = read_manifest(hf_dir)
    if record is None:
        return False
    return (
        record.get("iteration") == iteration
        and record.get("lm_engine_sha") == sha
        and record.get("model_type") == model_type
        and record.get("parity_passed") is True
    )


def write_manifest(hf_dir: pathlib.Path, record: dict) -> None:
    """Write ``export.json`` into the exported directory."""
    hf_dir.mkdir(parents=True, exist_ok=True)
    (hf_dir / MANIFEST_NAME).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )


# ------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-dir", required=True, help="lm-engine save_path")
    parser.add_argument(
        "--output-dir", required=True, help="where to write the HF model"
    )
    parser.add_argument(
        "--tokenizer-path",
        required=True,
        help="directory holding the trained tokenizer",
    )
    parser.add_argument(
        "--model-type",
        required=True,
        choices=MODEL_TYPES,
        help="Hugging Face architecture to export as",
    )
    parser.add_argument(
        "--iteration", default="", help="rung to export (default: the latest present)"
    )
    parser.add_argument(
        "--expected", default="", help="JSON file of training-side values to verify"
    )
    parser.add_argument(
        "--parity-tolerance",
        type=float,
        default=0.05,
        help="max |delta logits| allowed between the two models (default: %(default)s)",
    )
    parser.add_argument(
        "--dtype", default="bf16", help="dtype to unshard into (default: %(default)s)"
    )
    parser.add_argument(
        "--lm-engine-source",
        default=LM_ENGINE_SOURCE,
        help="lm-engine clone to read the pinned sha from (default: %(default)s)",
    )
    parser.add_argument(
        "--image", default="", help="image tag, recorded in the manifest"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    train_dir = pathlib.Path(args.train_dir)
    hf_dir = pathlib.Path(args.output_dir)
    tokenizer_dir = pathlib.Path(args.tokenizer_path)

    # Everything that can be known before any work is checked before any work: these
    # are the mistakes a recipe makes, and each one is cheaper as an error message than
    # as a stack trace from inside upstream ten minutes later.
    if not train_dir.is_dir():
        _log(f"error: --train-dir is not a directory: {train_dir}")
        return 2
    if not tokenizer_dir.is_dir():
        _log(f"error: --tokenizer-path is not a directory: {tokenizer_dir}")
        return 2
    try:
        iteration = resolve_iteration(train_dir, args.iteration)
    except ValueError as error:
        _log(f"error: {error}")
        return 2
    try:
        expected = load_expected(args.expected)
    except (OSError, ValueError) as error:
        _log(f"error: --expected {args.expected}: {error}")
        return 2

    sha = lm_engine_sha(args.lm_engine_source)

    if already_exported(
        hf_dir, iteration=iteration, sha=sha, model_type=args.model_type
    ):
        _log(
            f"{hf_dir / MANIFEST_NAME} already records a verified {args.model_type} "
            f"export of iteration {iteration} at lm-engine {sha}; nothing to do"
        )
        print(artifact_line(hf_dir), flush=True)
        return 0

    work = work_dir(hf_dir)
    unsharded = unsharded_dir(hf_dir)
    work.mkdir(parents=True, exist_ok=True)
    hf_dir.mkdir(parents=True, exist_ok=True)

    try:
        config_path = write_unshard_config(
            work / "unshard.yml", train_dir, iteration, unsharded, args.dtype
        )
        run_unshard(config_path)
        convert(unsharded, args.model_type, hf_dir)
        tokenizer_files = copy_tokenizer(tokenizer_dir, hf_dir)

        _log("stage 4/5 checks: config and tokenizer")
        errors = run_checks(hf_dir, tokenizer_dir, expected)
        if errors:
            _log(
                f"the export is not faithful to the run that produced it ({len(errors)} problem(s)):"
            )
            for problem in errors:
                _log(f"  - {problem}")
            _log(
                f"refusing to publish {hf_dir}; the unsharded checkpoint is kept at "
                f"{unsharded} for debugging"
            )
            return 1

        delta = check_parity(unsharded, hf_dir)
    except Exception as error:  # noqa: BLE001 - any stage failure must fail the step
        traceback.print_exc()
        _log(f"error: the export failed: {error}")
        _log(f"the work directory is kept at {work} for debugging")
        return 1

    passed = delta < args.parity_tolerance
    _log(
        f"stage 5/5 parity: max |delta logits| = {delta:.3e} "
        f"(tolerance {args.parity_tolerance:g}) -> {'pass' if passed else 'FAIL'}"
    )

    write_manifest(
        hf_dir,
        {
            "iteration": iteration,
            "train_dir": str(train_dir),
            "model_type": args.model_type,
            "dtype": args.dtype,
            "lm_engine_sha": sha,
            "image": args.image or None,
            "tokenizer_source": str(tokenizer_dir),
            "tokenizer_files": tokenizer_files,
            "parity_max_delta": delta,
            "parity_tolerance": args.parity_tolerance,
            "parity_passed": passed,
        },
    )

    if not passed:
        _log(
            f"the exported model does not reproduce the trained one: max |delta logits| "
            f"{delta:.3e} exceeds the tolerance {args.parity_tolerance:g}. Refusing to "
            f"publish {hf_dir}; the unsharded checkpoint is kept at {unsharded}"
        )
        return 1

    shutil.rmtree(work, ignore_errors=True)
    _log(f"done: {hf_dir}")
    print(artifact_line(hf_dir), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
