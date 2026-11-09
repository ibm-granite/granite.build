# Granite 4 GOLD distillation — SkyPilot/LSF recipes

Recipes for GOLD knowledge distillation of Granite 4 models on an LSF cluster
via the SkyPilot LSF backend. Each recipe directory carries its own
`build.yaml` and `parameters.yaml`, and most carry a `README.md`; read it
before the first run.

| Recipe | What it runs |
|---|---|
| [`gold-smoke`](gold-smoke/README.md) | GOLD distillation smoke test |
| [`gold-blend-smoke`](gold-blend-smoke/README.md) | On-policy-blended GOLD smoke test |
| [`gold-onpolicy-smoke`](gold-onpolicy-smoke/README.md) | On-policy GOLD smoke test, vLLM as a separate target |
| [`gold-onpolicy-full-smoke`](gold-onpolicy-full-smoke/README.md) | Fully on-policy GOLD smoke test |
| [`gold-sweep-smoke`](gold-sweep-smoke/README.md) | The short gate for `gold-sweep-100` |
| [`gold-sweep-100`](gold-sweep-100/README.md) | One 100-step off-policy sweep arm |
| [`distill-pipeline-smoke`](distill-pipeline-smoke/README.md) | Distillation pipeline smoke test |
| [`distill-gold`](distill-gold/build.yaml) | Real-scale GOLD distillation |
| [`distill-sft-gold`](distill-sft-gold/build.yaml) | SFT warm-up feeding real-scale GOLD distillation |

## Placeholders

`<shared-fs>` in these docs and in the `parameters.yaml` comments stands for a
directory on the cluster's shared filesystem, one that every compute node and
container sees at the same path. It is not the environment's `shared_workdir`:
that holds gbserver's per-run directories
(`${shared_workdir}/builds/<build_id>/runs/<targetrun_id>/`), while the
`<shared-fs>` paths hold models, overlays and datasets you manage yourself. See
[`file_mounts` inside enroot containers](../../../docs/environments/skypilot-lsf.md#file_mounts-inside-enroot-containers)
for which shared paths LSF containers can see.

The `parameters.yaml` defaults (paths, queues, images) are specific to the
cluster the recipes were developed on. To run elsewhere, override them with
`--param KEY=VALUE` on `gb build start`.
