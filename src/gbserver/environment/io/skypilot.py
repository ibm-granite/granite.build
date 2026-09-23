"""SkyPilot renderings of the inline prologue (hf download) and epilogue
(marker-capture hf upload).

See docs/plans/2026-09-18-inline-hfpush-envio-design.md §4-§6.

The prologue mirrors the existing inline-hfpull shell in
``gbserver.environment.skypilot`` (factored here). The epilogue resolves each
output's source path *at runtime* from the step's own ``GB_ARTIFACT_PATH``
marker (captured to the file named by ``capture_var``), sets every ``HF_*`` /
``BINDING_ID`` env var that ``HFPUSH_UPLOAD_BODY`` reads, and then runs that
shared upload body (which begins at ``hf_mocked() {`` and ends with the
``Pushed HF URI: ...`` marker). ``HFPUSH_UPLOAD_BODY`` is derived from the
drift-guarded ``HFPUSH_RUN_SHELL`` (hf_shell.py), so it stays in lock-step with
the skypilot hfpush step.yaml run block.
"""

from collections.abc import Sequence

from gbserver.environment.io.base import EnvironmentIO
from gbserver.environment.io.descriptors import (
    HfInputIO,
    HfOutputIO,
    InputIO,
    OutputIO,
)
from gbserver.environment.io.hf_shell import HFPUSH_UPLOAD_BODY


def _shq(value: str) -> str:
    """Single-quote ``value`` for POSIX shell (safe against injection)."""
    return "'" + value.replace("'", "'\\''") + "'"


class SkypilotIO(EnvironmentIO):
    def render_prologue(self, inputs: Sequence[InputIO]) -> str:
        hf_inputs = [i for i in inputs if isinstance(i, HfInputIO)]
        if not hf_inputs:
            return ""
        lines = [
            "# -- gbserver: inline hfpull for inputs --",
            "pip install --no-cache-dir 'huggingface_hub[cli]' 2>/dev/null || true",
        ]
        for i in hf_inputs:
            cmd = f'hf download "{i.repo}" --local-dir "{i.dest}"'
            if i.revision:
                cmd += f' --revision "{i.revision}"'
            if i.type:
                cmd += f" --repo-type {i.type}"
            lines.append(cmd)
        lines.append("# -- end inline hfpull --")
        return "\n".join(lines) + "\n"

    def render_epilogue(self, outputs: Sequence[OutputIO], capture_var: str) -> str:
        hf_outputs = [o for o in outputs if isinstance(o, HfOutputIO)]
        if not hf_outputs:
            return ""
        blocks = []
        for o in hf_outputs:
            owner = o.repo.split("/", 1)[0]
            repo_name = o.repo.split("/", 1)[1] if "/" in o.repo else o.repo
            resource_group_id = (
                "" if o.resource_group_id is None else o.resource_group_id
            )
            # huggingface_hub reads HF_ENDPOINT from the environment; an
            # empty-but-SET value overrides its https://huggingface.co default
            # with "" and breaks every API call ("Request URL is missing an
            # 'http://' or 'https://' protocol"). So export HF_ENDPOINT only when
            # non-empty; otherwise `unset` it so hf falls back to its default.
            endpoint = getattr(o, "endpoint", "") or ""
            endpoint_line = (
                f"export HF_ENDPOINT={_shq(endpoint)}"
                if endpoint
                else "unset HF_ENDPOINT"
            )
            # Resolve the source path at runtime from the step's own
            # GB_ARTIFACT_PATH marker (marker-capture, spec §5): read the
            # capture file, pull the artifact path for THIS binding_id,
            # matching either a GB_ or LLMB_ prefix, and take the last match.
            resolve = "\n".join(
                [
                    "# -- gbserver: inline hfpush (marker-capture) --",
                    f"BINDING_ID={_shq(o.binding_id)}",
                    'HF_SOURCE="$(sed -n '
                    '"s/.*\\(GB_\\|LLMB_\\)ARTIFACT_ID:${BINDING_ID} '
                    '.*\\(GB_\\|LLMB_\\)ARTIFACT_PATH:\\([^ ]*\\).*/\\3/p" '
                    f'"${{{capture_var}}}" | tail -n 1)"',
                    'if [ -z "${HF_SOURCE:-}" ]; then',
                    '  echo "inline hfpush: no GB_ARTIFACT_PATH for binding'
                    ' ${BINDING_ID}" >&2',
                    "  exit 1",
                    "fi",
                    "export HF_SOURCE",
                    f"HF_URI={_shq(o.uri)}",
                    f"HF_OWNER={_shq(owner)}",
                    f"HF_REPO_NAME={_shq(repo_name)}",
                    'export HF_REPO="${HF_OWNER}/${HF_REPO_NAME}"',
                    endpoint_line,
                    f"export HF_REVISION={_shq(o.revision)}",
                    f"export HF_PATH_IN_REPO={_shq(o.path_in_repo)}",
                    f"export HF_PRIVATE={_shq(str(o.private))}",
                    f"export HF_TYPE={_shq(o.hf_type)}",
                    f"export HF_RESOURCE_GROUP_ID={_shq(resource_group_id)}",
                    "export HF_URI HF_OWNER HF_REPO_NAME BINDING_ID",
                ]
            )
            blocks.append(resolve + "\n" + HFPUSH_UPLOAD_BODY)
        return "\n".join(blocks) + "\n"
