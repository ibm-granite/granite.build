# hfpull/hfpush LSF Support

Add LSF environment support for the existing hfpull and hfpush builtin steps, enabling Hugging Face Hub asset pull/push on LSF clusters using the `huggingface-cli` directly.

## Context

hfpull/hfpush currently only support K8s (Helm-based). LSF support is needed to match lhpull/lhpush which already work on both K8s and LSF. The LSF implementation uses Jinja2-templated `command.sh` scripts and injects secrets via the step config.

## Files to Create

### `src/gbserver/builtins/steps/hfpull/lsf_scripts/hfpull/command.sh`

Minimal Jinja — only used at the top to extract config values into shell variables, then pure bash.

```bash
#!/usr/bin/env bash
echo 'hfpull start'

{%- set hfp = config.hfpull_config %}

HF_DEST='{{ hfp.path }}'
HF_URI='{{ hfp.uri }}'
HF_REPO='{{ hfp.owner }}/{{ hfp.repo }}'
HF_REVISION='{{ hfp.revision }}'

if [[ -z "$HF_TOKEN" ]]; then
    echo 'HF_TOKEN is not set'
    exit 1
fi

echo "LLMB_LSF_LAUNCH_ID ${LLMB_LSF_LAUNCH_ID}"
echo "LLMB_LSF_ASSET_DIR ${LLMB_LSF_ASSET_DIR}"
echo "LLMB_LSF_QUEUE ${LLMB_LSF_QUEUE}"
echo "LLMB_LSF_JOB_NAME ${LLMB_LSF_JOB_NAME}"
echo "LLMB_LSF_WORKSPACE_DIR ${LLMB_LSF_WORKSPACE_DIR}"
echo "LLMB_LSF_OUTPUT_DIR ${LLMB_LSF_OUTPUT_DIR}"

echo "Pulling HF URI: ${HF_URI} to path ${HF_DEST}"

REVISION_FLAG=""
if [[ -n "$HF_REVISION" ]]; then
    REVISION_FLAG="--revision ${HF_REVISION}"
fi

echo huggingface-cli download "${HF_REPO}" --local-dir "${HF_DEST}" ${REVISION_FLAG}
huggingface-cli download "${HF_REPO}" --local-dir "${HF_DEST}" ${REVISION_FLAG}

MY_EXIT_CODE=$?
if [[ "${MY_EXIT_CODE}" != '0' ]]; then
    echo "${LLMB_LSF_JOB_NAME}: hfpull failed, exit code: ${MY_EXIT_CODE}"
    exit 1
fi

echo "Pulled HF URI: ${HF_URI} to path ${HF_DEST}"
echo 'hfpull end'
```

### `src/gbserver/builtins/steps/hfpush/lsf_scripts/hfpush/command.sh`

```bash
#!/usr/bin/env bash
echo 'hfpush start'

{%- set hfp = config.hfpush_config %}

HF_SOURCE='{{ hfp.path }}'
HF_URI='{{ hfp.uri }}'
HF_REPO='{{ hfp.owner }}/{{ hfp.repo }}'
HF_REVISION='{{ hfp.revision }}'
HF_PRIVATE='{{ hfp.private }}'
BINDING_ID='{{ hfp.binding_id }}'

if [[ -z "$HF_TOKEN" ]]; then
    echo 'HF_TOKEN is not set'
    exit 1
fi

echo "LLMB_LSF_LAUNCH_ID ${LLMB_LSF_LAUNCH_ID}"
echo "LLMB_LSF_ASSET_DIR ${LLMB_LSF_ASSET_DIR}"
echo "LLMB_LSF_QUEUE ${LLMB_LSF_QUEUE}"
echo "LLMB_LSF_JOB_NAME ${LLMB_LSF_JOB_NAME}"
echo "LLMB_LSF_WORKSPACE_DIR ${LLMB_LSF_WORKSPACE_DIR}"
echo "LLMB_LSF_OUTPUT_DIR ${LLMB_LSF_OUTPUT_DIR}"

echo "Pushing HF URI: ${HF_URI} from path ${HF_SOURCE}"

REVISION_FLAG=""
if [[ -n "$HF_REVISION" ]]; then
    REVISION_FLAG="--revision ${HF_REVISION}"
fi

PRIVATE_FLAG=""
if [[ "${HF_PRIVATE}" == "True" ]]; then
    PRIVATE_FLAG="--private"
fi

echo huggingface-cli upload "${HF_REPO}" "${HF_SOURCE}" ${REVISION_FLAG} ${PRIVATE_FLAG}
huggingface-cli upload "${HF_REPO}" "${HF_SOURCE}" ${REVISION_FLAG} ${PRIVATE_FLAG}

MY_EXIT_CODE=$?
if [[ "${MY_EXIT_CODE}" != '0' ]]; then
    echo "${LLMB_LSF_JOB_NAME}: hfpush failed, exit code: ${MY_EXIT_CODE}"
    exit 1
fi

echo "Pushed HF URI: ${HF_URI} for binding ${BINDING_ID}"
echo 'hfpush end'
```

The `Pushed HF URI:` line matches the existing `line_regex: "Pushed HF URI:\\s.+"` pattern in the K8s step.yaml event_configs, enabling ARTIFACT_PUSHED_EVENT detection.

## Files to Modify

### `src/gbserver/builtins/steps/hfpull/step.yaml`

Add `Lsf` section to `environment_configs`:

```yaml
  Lsf:
    launchers:
      tuning:
        type: bsub
        monitors:
        - bsub_monitor
    monitors:
      bsub_monitor:
        type: bsub_monitor
```

### `src/gbserver/builtins/steps/hfpush/step.yaml`

Add `Lsf` section to `environment_configs` with event monitoring:

```yaml
  Lsf:
    launchers:
      tuning:
        type: bsub
        monitors:
        - bsub_monitor
    monitors:
      bsub_monitor:
        type: bsub_monitor
        config:
          event_configs:
          - event_type: ARTIFACT_PUSHED_EVENT
            line_regex: "Pushed HF URI:\\s.+"
            is_json: False
            event_fields:
              - field_name: uri
                field_regex: "hf://[^\\s]+"
              - field_name: binding_id
                field_regex: "(?<=binding\\s)[^\\s]+"
```

### `src/gbserver/environment/lsf.py`

**New imports:**
```python
from gbcommon.uri.hf import HfURI
from gbserver.asset.hfstore import Hfstore
from gbserver.types.constants import (
    ...
    CODE_GBSERVER_BUILTINS_STEPS_HFPULL_DIR,
    CODE_GBSERVER_BUILTINS_STEPS_HFPULL_URI,
    CODE_GBSERVER_BUILTINS_STEPS_HFPUSH_DIR,
    CODE_GBSERVER_BUILTINS_STEPS_HFPUSH_URI,
)
```

**New helper method: `_load_builtin_hf_lsf_section()`**

Reads the hfpull or hfpush step.yaml, extracts the `lsf` and `workload` sections, injects HF_TOKEN from `hf_metadata["token_secretname"]`, and sets `skip_finding_output_artifacts: True`.

```python
def _load_builtin_hf_lsf_section(
    self, step_dir: Path, hf_metadata: dict
) -> Tuple[dict, dict]:
    step_path = step_dir / STEP_FILE_NAME
    assert step_path.is_file(), f"step yaml is missing: {step_path}"
    step_config = StepConfig.from_yaml(path=step_path)
    step_config_section = StepConfigSection.model_validate(step_config.config)
    lsf_dict = step_config_section.lsf.model_dump(exclude_unset=True)
    workload_dict = step_config_section.workload.model_dump(exclude_unset=True)
    assert "token_secretname" in hf_metadata
    lsf_dict["secrets"] = {
        "secret_names_to_use_as_env_variable": [
            {"env_name": "HF_TOKEN", "secret_name": hf_metadata["token_secretname"]}
        ]
    }
    lsf_dict["skip_finding_output_artifacts"] = True
    return lsf_dict, workload_dict
```

**New method: `pullasset_hfstore()`**

Mirrors `pullasset_lhstore` structure:
1. Validates assetstore is `Hfstore` and storeload_config mode is `"hf_pull"`
2. Builds binding_path from cache_path / owner / repo / hash
3. Gets `hf_metadata` from `Hfstore.get_metadata(uri)` (returns `{"token_secretname": "HF_TOKEN"}`)
4. Builds `hfpull_config` dict: `path`, `uri`, `owner`, `repo`, `revision`
5. Calls `_load_builtin_hf_lsf_section()` to get lsf/workload dicts with HF_TOKEN injected
6. Returns `(binding_config, BuildTargetStepConfig)`

**New method: `pushasset_hfstore()`**

Mirrors `pushasset_lhstore` structure:
1. Validates binding dict has `"path"`
2. Parses HfURI, gets hf_metadata
3. Builds `hfpush_config`: `path`, `uri`, `binding_id`, `owner`, `repo`, `revision`, `private` (flat structure for simple template access)
4. Includes `hf` sub-dict for any additional fields from storepush_config/output_config
5. Calls `_load_builtin_hf_lsf_section()` for secret injection
6. Returns `BuildTargetStepConfig`

## Testing

Existing test patterns for lhpull/lhpush LSF should be mirrored for hfpull/hfpush. Unit tests should validate:
- `pullasset_hfstore` returns correct config structure with HF_TOKEN injected
- `pushasset_hfstore` returns correct config with event monitoring configured
- command.sh templates render correctly with sample configs

## Non-goals

- No changes to the K8s flow (already works)
- No changes to `gbcommon.uri.hf` (not needed since we use CLI directly)
- No venv setup (huggingface-cli assumed available in the LSF environment)
