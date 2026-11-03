---
name: list-steps
description: Use when asked for an inventory, table, list, or overview of the step implementations in the granite.build repo's steps/ directory — which steps exist, their category, supported or tested environments (skypilot subtypes, slurm, lsf, aws, kubernetes, bash), container image, README, or sample build.yaml.
---

# List steps (inventory table)

Produces one Markdown table per category (a `## <Category>` heading over each, categories alphabetical) covering every git-tracked step under `steps/<group>/<name>/<env>/`, one row per step (implementations for several envs, e.g. `tune/fm-tune` bash + skypilot, share a row).

## Run

From the repo root:

```bash
python3 .claude/skills/list-steps/scripts/steps_table.py
```

stdout is the tables; stderr carries `NOTE:` lines that need judgment. Show the user the tables and relay every NOTE. Add them to a file (e.g. `steps/README.md`) only if asked.

## Columns (what the script derives)

| Column | Source |
|---|---|
| Category (table heading, not a column) | `STEP_CATEGORY` override, else `GROUP_CATEGORY` by top-level group (`tune` → Train); unknown → Other |
| Step | `<group>/<name>` |
| Environments | `subtypes:` under each `environment_configs` entry in `step-template.yaml` (so `lsf`, `slurm` are skypilot subtypes); when absent, the env type lowercased (`skypilot`, `bash`) |
| Tested Environments | each `test/<case>/test_*.py` paired with `test-data/<case>/build.yaml`'s `environment_uri`; labels are sorted; a skypilot subtype tested on >1 cluster shows the cluster, e.g. `slurm (bluevela)`, with the base env (`skypilot/slurm`) shown as `slurm (local)` |
| README | `configurations/assets/environments/<env>/steps/<step>/README.md`; unpublished steps fall back to `steps/.../README.md` and get `‡`. Link text is `README.md`, or the impl dir name (`bash`, `skypilot`) when a step has several impls |
| Image | blank only if the impl has a `Dockerfile` (custom image); else the launcher `image_id` (following one `config.X.Y` indirection); `none (bare node)` when empty; `?` plus a NOTE when unresolvable |
| Sample build | one `test-data/<case>/build.yaml` per impl: local SLURM first, then AWS, then any other; ties go to the shortest case name, then alphabetical |

Within each table rows sort by the full Step path; a group can be split across tables (e.g. `distill/hf-export` under Misc). Unit/render tests (`test/test_*.py`) do not count as tested environments.

## When the script needs updating

- **NOTE "no category mapping"**: a new group or a new mixed-category step (like `distill/*`). Ask the user for the category, then add it to `STEP_CATEGORY` / `GROUP_CATEGORY`.
- **NOTE "could not resolve image_id"**: the template builds the image ref some new way; fill the cell by hand and extend `resolve_image_id`.
- **NOTE "no ... environment_uri"**: a test case without a matching build, or one using `file:` envs; check it by hand.

## Common mistakes

- Listing untracked step directories: the script reads `git ls-files`, so local scratch steps are skipped on purpose.
- Reporting the platforms test builds use as the Environments column: that column is the template's declared `subtypes`, not what's tested.
- Calling a registry image "public": the Image column shows any non-Dockerfile image; say so if it's from a private registry (e.g. `us.icr.io/...`).
