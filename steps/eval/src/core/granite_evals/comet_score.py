"""COMET scoring for wmt24pp, run by the image's COMET env (``/opt/comet``).

``python comet_score.py <spec.json>``. The spec (written by
``nemo_skills_extended.WMT24pp``) names ns's own scorer module
(``nemo_skills/evaluation/evaluator/comet.py``, loaded by file path: the COMET
env has no nemo_skills), the local COMET model dir, the precision, the batch
size and the ``[input, output]`` files to score. Each file goes through ns's
``process_file`` (all visible GPUs; adds ``comet`` to every row and writes
``<output>.done``).

One departure from ns's ``load_comet_model``: the checkpoint is loaded with
``reload_hparams=True, local_files_only=True`` so the encoder's config and
tokenizer come from the pinned local snapshot the run downloaded (the model's
``hparams.yaml`` there points at it), not from the hub at its head. The weights
are the checkpoint's, as in ns.

Only the standard library is imported at the top: this file lives in the
granite_evals package but runs in an env that has none of its dependencies.
Lightning's multi-GPU launcher reruns this script once per extra GPU; every rank
reads the same spec, so all ranks score the same files.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text())
    mod_spec = importlib.util.spec_from_file_location("ns_comet", spec["ns_comet"])
    ns_comet = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(ns_comet)  # also applies ns's torch.load compat patch

    from comet import load_from_checkpoint

    model = load_from_checkpoint(spec["checkpoint"], reload_hparams=True, local_files_only=True)
    dtype = ns_comet._PRECISION_TO_DTYPE[spec["precision"]]
    if dtype is not None:
        model = model.to(dtype=dtype)
    for inp, out in spec["files"]:
        ns_comet.process_file(Path(inp), Path(out), model, int(spec["batch_size"]))


if __name__ == "__main__":
    main()
