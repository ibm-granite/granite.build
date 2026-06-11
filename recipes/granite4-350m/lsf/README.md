# Granite 4.0 350M — LSF (BlueVela) recipes

Build recipes for SFT training and evaluation of the Granite 4.0 350M
model on the BlueVela LSF cluster via the SkyPilot LSF backend.

## Recipes

| Recipe              | Purpose                                                     |
| ------------------- | ----------------------------------------------------------- |
| `sft-10k-test`      | Open Instruct SFT training on a 10k sample                  |
| `bfcl-eval`         | BFCLv3 function-calling evaluation                          |
| `bcb-server`        | BigCodeBench evaluation server                              |
| `general-eval`      | General-domain Sage eval suite (5 targets)                  |
| `math-eval`         | Math-domain Sage eval suite (5 targets)                     |
| `code-eval`         | Code-domain Sage eval suite (9 targets)                     |
| `multilingual-eval` | Multilingual Sage eval suite (5 targets)                    |
| `safety-eval`       | Safety Sage eval suite (2 targets)                          |
| `full-eval`         | Combined 27-target suite (26 Sage + 1 BFCL)                 |
| `sft-10k-eval-test` | SFT (2 epochs) chained to the 27-target eval suite via output binding |
| `export-results`    | Copy results from shared FS to the configured output store  |

## Defaults are BlueVela-specific

The `parameters.yaml` files in this directory carry default values for
paths, queues, and resources that are specific to the BlueVela cluster
and the granite-build project layout — for example:

```yaml
MODEL_PATH: "/proj/granite-build/g4os/granite-4.0-350m-base/r251014a"
OUTPUT_DIR: "/proj/granite-build/g4os/sft/checkpoints"
QUEUE: "normal"
ACCELERATORS: "H100:1"
```

These are placeholders for a working BlueVela deployment. To run on a
different LSF cluster, override every infrastructure-specific parameter
on the command line:

```shell
gb build start -f recipes/granite4-350m/lsf/sft-10k-test/build.yaml \
  --parameters-path recipes/granite4-350m/lsf/sft-10k-test/parameters.yaml \
  --param MODEL_PATH=/your/cluster/path/to/model \
  --param TOKENIZED_DATA_PATH=/your/cluster/path/to/data \
  --param OUTPUT_DIR=/your/cluster/path/to/output \
  --param QUEUE=your-lsf-queue \
  --param ACCELERATORS=H100:8
```

`QUEUE` maps to SkyPilot's `zone` field, which the LSF cloud backend
interprets as the LSF queue name (e.g. `normal`, `preemptable`).
