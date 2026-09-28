"""SkyPilot rendering of the inline prologue (hf download).

See docs/plans/2026-09-18-inline-hfpush-envio-design.md §4-§6.

The prologue mirrors the existing inline-hfpull shell in
``gbserver.environment.skypilot`` (factored here). The push-side epilogue was
removed: SkyPilot/AWS artifact push is now a dispatched step over a shared
filesystem whose destination resolves at push time (#390), so no upload shell
is folded into the producing step's run script.
"""

from collections.abc import Sequence

from gbserver.environment.io.base import EnvironmentIO
from gbserver.environment.io.descriptors import HfInputIO, InputIO


class SkypilotIO(EnvironmentIO):
    def render_prologue(self, inputs: Sequence[InputIO]) -> str:
        hf_inputs = [i for i in inputs if isinstance(i, HfInputIO)]
        if not hf_inputs:
            return ""
        lines = [
            "# -- gbserver: inline hfpull for inputs --",
            # Pin <2.0: hf 2.x pulls httpx2, whose BrotliDecoder calls
            # brotli.Decompressor.process(output_buffer_limit=...), a kwarg only
            # in brotli>=1.2.0; the worker's ambient conda brotli (1.0.9) rejects
            # it (TypeError). Not a Python-version issue. See the separate pin PR
            # (fix/pin-hfhub-lt2-skypilot) and the hf-2.x/brotli follow-up issue.
            "pip install --no-cache-dir 'huggingface_hub[cli]<2.0' "
            "2>/dev/null || true",
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
