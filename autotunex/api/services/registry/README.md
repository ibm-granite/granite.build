# Model Registry backends

A `ModelRegistry` (see `base.py`) abstracts where trained/base models live. Selection
is by the `autotunex.model_registries` entry-point name, resolved via
`services.plugins.resolve(Seam.REGISTRY, ...)`:

- explicit `AUTOTUNEX_REGISTRY=<name>` wins;
- else fallback: `dmf` if `lakehouse` is importable (IBM deployments), otherwise `local`.

Built-ins:
- `local` (`local_backend.LocalRegistry`) — disk-backed, the zero-config default.
- `dmf` (`dmf_backend.DmfRegistry`) — IBM Lakehouse/DMF; `lakehouse` imported lazily.
- `hf` (`hf_backend.HuggingFaceRegistry`) — HuggingFace Hub, read-only (writes raise 501).

## Add a backend
1. Subclass `ModelRegistry` and implement all abstract methods. Match the return and
   `HTTPException` shapes of the existing backends so frontend/MCP consumers are unaffected.
2. Register it: add `"<name> = services.registry.<module>:<Class>"` under
   `autotunex.model_registries` in `setup.py`, then `pip install -e .`.
3. Select it with `AUTOTUNEX_REGISTRY=<name>`.
