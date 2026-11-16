# SPDX-License-Identifier: Apache-2.0

"""Build-time patch for upstream open-instruct finetune.py (pinned commit).

finetune.py pre-fetches the model on rank 0 with
``snapshot_download(model_name_or_path)``. ``snapshot_download()`` validates its
argument as a Hub repo id, so a local model directory (the step's staged
``base_model``) raises ``HFValidationError``. Skip the pre-fetch for a local
directory; Hub ids are unchanged. Exits non-zero if the target line is not found
exactly once, so an upstream change fails the image build instead of silently
shipping an unpatched trainer.
"""

import pathlib
import sys

OLD = (
    "    if model_path and accelerator.is_main_process:\n" "        snapshot_download("
)
NEW = (
    "    if model_path and not os.path.isdir(model_path) and accelerator.is_main_process:\n"
    "        snapshot_download("
)


def patch(path: pathlib.Path) -> None:
    """Apply the local-model patch to ``path`` in place.

    :param path: open_instruct/finetune.py.
    :raises SystemExit: If the target is not found exactly once.
    """
    text = path.read_text()
    if text.count(OLD) != 1:
        sys.exit(f"openinstruct-sft: patch target not found exactly once in {path}")
    path.write_text(text.replace(OLD, NEW))


if __name__ == "__main__":
    patch(pathlib.Path(sys.argv[1]))
