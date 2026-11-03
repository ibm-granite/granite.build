---
name: list-steps
description: Use when asked for an inventory, table, list, or overview of the step implementations in the granite.build repo's steps/ directory — which steps exist, their category, supported or tested environments (skypilot subtypes, slurm, lsf, aws, kubernetes, bash), container image, documentation (README), or sample build.yaml.
---

# List steps (inventory table)

Produces one table per category (categories alphabetical) of the framework steps — every git-tracked step under `steps/<group>/<name>/<env>/` — followed by one final **Non-Framework Steps** table (uncategorised) of every git-tracked `configurations/**/step.yaml` that no framework step publishes. Both are always produced. One row per step: implementations for several envs share a row (e.g. `tune/fm-tune` bash + skypilot; non-framework `hello` in bash/docker/runpod/skypilot-aws). Output as Markdown, an Excel workbook, and/or a PowerPoint deck.

## Run

From anywhere inside the repo (the script switches to the repo root itself; `-o` is resolved against your current directory):

```bash
python3 .claude/skills/list-steps/scripts/steps_table.py                    # Markdown to stdout
python3 .claude/skills/list-steps/scripts/steps_table.py -f xlsx -f pptx -o /tmp/steps-inventory
```

| Format (`-f`, repeatable) | Output | Layout |
|---|---|---|
| `markdown` (default) | stdout, or `<-o>.md` | `## <Category>` heading over each table, `## Non-Framework Steps` last, then footnotes |
| `xlsx` | `<-o>.xlsx` | one sheet per category plus `Non-Framework Steps`; frozen header + filters; single links clickable, multi-link cells list URLs in a comment |
| `pptx` | `<-o>.pptx` | 16:9, one table slide per category titled with it, then `Non-Framework Steps` (rows that would overflow, estimated from wrapped text, continue on a `(cont.)` slide); links clickable |

`-o` is a path *without* extension (default `./steps-inventory`); ask where to write files rather than dropping them in the repo. Links are repo-relative in Markdown; xlsx/pptx default to the GitHub blob URL of the `upstream` (else `origin`) remote's `main`, since relative links don't work in Office. Override with `--link-base URL` (`--link-base ''` for relative; xlsx then writes link cells as text with the paths in a cell comment, since Excel can't hyperlink a scheme-less path).

`xlsx` needs XlsxWriter (`pptx` and `markdown` are stdlib only). If the script exits with "missing dependency", install the pinned, hash-checked version, preferably in a throwaway venv: `pip install --require-hashes -r .claude/skills/list-steps/scripts/requirements.txt`.

stdout is the Markdown tables; stderr carries `NOTE:` lines that need judgment and `wrote <path>` per file. Show the user the tables (or the file paths) and relay every NOTE. Add tables to a repo file (e.g. `steps/README.md`) only if asked.

## Columns (what the script derives)

| Column | Source |
|---|---|
| Category (table heading, not a column) | framework steps only: `STEP_CATEGORY` override, else `GROUP_CATEGORY` by top-level group (`tune` → Train); unknown → Other |
| Step | `<group>/<name>` for framework steps; for non-framework steps, the path below `.../steps/` (e.g. `bfcl-eval`) |
| Environments | framework steps: one label per `environment_configs` entry in `step-template.yaml`: the environment class lowercased, plus ` - <subtypes>` (comma-separated) when it declares `subtypes:`, e.g. `skypilot - lsf`, `skypilot` (no restriction: any subtype), `bash`; a step with several impls lists each, separated by `; `. Non-framework steps under `configurations/assets/environments/<class>/<sub>/[<cluster>/]steps/` take the subtype from the directory (e.g. `skypilot - lsf (ibm-bluevela)`), since it is the most specific; directories without a subtype use the step.yaml's declared environments, falling back to the directory's class (`configurations/assets/steps/<class>/` gives `<class>`) |
| Tested Environments | framework steps only (non-framework: `—`): each `test/<case>/test_*.py` paired with every `environment_uri` (quoted or not) in `test-data/<case>/build.yaml`; labels are sorted; a skypilot subtype tested on >1 cluster shows the cluster, e.g. `slurm (bluevela)`, with the base env (`skypilot/slurm`) shown as `slurm (local)` |
| Documentation | README at `configurations/assets/environments/<env>/steps/<step>/README.md`; unpublished steps fall back to `steps/.../README.md` and get `‡`; `—` when neither exists; non-framework steps link the README beside their step.yaml, or `—` if none. Link text is `README.md`, or the impl dir name (`bash`, `skypilot`) when a step has several impls |
| Image | blank only if the impl has a `Dockerfile` (custom image), or `custom (Dockerfile)` when it shares a row with impls that don't; else the launcher `image_id`, or `image` for docker/runpod (following one `config.X.Y` indirection); `Helm chart` for a k8s chart launcher; `none (bare node)` when empty; `?` plus a NOTE when unresolvable |
| Sample build | framework steps only (non-framework: `—`): one `test-data/<case>/build.yaml` per impl: local SLURM first, then AWS, then any other; ties go to the shortest case name, then alphabetical |

Within each table rows sort by the full Step path; a group can be split across tables (e.g. `distill/hf-export` under Misc). Unit/render tests (`test/test_*.py`) do not count as tested environments.

## When the script needs updating

- **NOTE "no category mapping"**: a new group, or a new mixed-category step (like `distill/*`). Ask the user for the category, then add it to `STEP_CATEGORY` / `GROUP_CATEGORY`.
- **NOTE "could not resolve image_id"**: the template builds the image ref some new way; fill the cell by hand and extend `resolve_image_id`.
- **NOTE "no ... environment_uri"**: a test case without a matching build, or one using `file:` envs; check it by hand.

## Common mistakes

- Listing untracked step directories: the script reads `git ls-files`, so local scratch steps are skipped on purpose.
- Reporting the platforms test builds use as the Environments column: that column is the template's declared `subtypes`, not what's tested.
- Calling a registry image "public": the Image column shows any non-Dockerfile image; say so if it's from a private registry (e.g. `us.icr.io/...`).

## Code layout

`scripts/steps_table.py` collects data into a format-neutral model (`scripts/cells.py`: cells are lists of `str` / `Code` / `Link` parts); `render_md.py`, `render_xlsx.py`, `render_pptx.py` each turn that model into one format. A new column or rule changes `steps_table.py` (and `COLUMNS` in `cells.py`; for pptx, `COL_WEIGHTS`) — not the renderers.
