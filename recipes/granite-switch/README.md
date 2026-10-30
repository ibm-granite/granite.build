# Granite Switch

Granite Switch lets you compose adapter functions into a single deployable
model. It brings the rigor and modularity of software engineering to LLMs.

## Available recipes

| Recipe | What it builds |
|---|---|
| [`lsf/answerability-sr`](lsf/answerability-sr/README.md) | A Shadow Residual answerability adapter on `granite-4.2-8b`, trained and then scored against the answerability eval set. |

More adapter-library recipes will be published here. Pick the `build.yaml` of
the recipe you want and run:

```bash
gb build start -f <path to the recipe>/build.yaml
```

Each recipe directory carries its own `README.md` and `parameters.yaml`; read
the README first, since a recipe may have a prerequisite to satisfy before the
first run — data to stage, or a credential to register.

Alternatively, access Granite.build through an agent session and simply ask
it to start a new build using that recipe.

For more details, see the Granite.build
[Getting started guide](../../docs/getting-started.md).

