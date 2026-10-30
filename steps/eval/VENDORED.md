# Vendored source: granite-evals

The benchmark runtime is vendored from
[laminair/granite-evals](https://github.com/laminair/granite-evals) at commit
`dce96b3` (Apache-2.0), so the eval steps do not depend on an external repository.

Layout (paths relative to this directory). The images assemble one
`granite_evals` package by overlaying a family's `src/` onto the core's
`granite_evals/benchmarks/` package directory; each family's vendored unit
tests sit beside its modules at `<family>/tests/`:

| Directory                    | Contents                                                                      |
|------------------------------|-------------------------------------------------------------------------------|
| `src/core/`                  | CLI, registry, data, meter, serving, sandbox, `benchmarks/nemo_skills.py`, `pyproject.toml`, `uv.lock` |
| `bfcl/src/`                  | `bfcl`                                                                        |
| `bird/src/`                  | `birdbench`                                                                   |
| `ifbench/src/`               | `ifbench` (split out of upstream `chat_if.py`)                                |
| `judged/src/`                | `judge_general` (profbench, gdpval), `mcpatlas`, `multi_challenge`, `safety`  |
| `lmeval/src/`                | `chat_if` (mmlu-prox-lite)                                                    |
| `nemoskills/src/`            | `nemo_skills_code`, `nemo_skills_extended`, `nemo_skills_ruler`, `scicode_prefill`, `tools_agentic` |
| `swebench/src/`              | `swebench`                                                                    |
| `tau/src/`                   | `tau`                                                                         |
| `tbench/src/`                | `tbench`                                                                      |

Changes from upstream:

- Internal service endpoints (judge gateway, MCP search) are no longer defaulted;
  pass `--option judge_base_url=...` / `--option search_mcp_url=...`.
- `languages=ibm` is now `languages=extended`.
- Site-specific scripts, docs, registry defaults and author metadata were dropped.
- `chat_if.py` is split so IFBench ships in its own family image.

Build the images with `make -C steps/eval images REGISTRY=<registry>`.
