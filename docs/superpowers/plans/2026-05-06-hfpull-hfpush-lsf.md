# hfpull/hfpush LSF Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LSF environment support for hfpull/hfpush builtin steps so Hugging Face Hub assets can be pulled/pushed on LSF clusters using `huggingface-cli`.

**Architecture:** Create `lsf_scripts/` command.sh templates for both steps (minimal Jinja at top, pure bash below), add `Lsf` environment_configs to each step.yaml, and implement `pullasset_hfstore()`/`pushasset_hfstore()` methods in the `Lsf` class following the same pattern as `pullasset_lhstore()`/`pushasset_lhstore()`.

**Tech Stack:** Python 3.11+, Jinja2 templates, bash, `huggingface-cli`

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/gbserver/builtins/steps/hfpull/lsf_scripts/hfpull/command.sh` | LSF job script: downloads HF repo via CLI |
| Create | `src/gbserver/builtins/steps/hfpush/lsf_scripts/hfpush/command.sh` | LSF job script: uploads to HF repo via CLI |
| Modify | `src/gbserver/builtins/steps/hfpull/step.yaml` | Add `Lsf` environment_configs section |
| Modify | `src/gbserver/builtins/steps/hfpush/step.yaml` | Add `Lsf` environment_configs with event monitoring |
| Modify | `src/gbserver/environment/lsf.py` | Add `pullasset_hfstore`, `pushasset_hfstore`, `_load_builtin_hf_lsf_section` |
| Create | `test/unit/environment/test_lsf_hfstore.py` | Unit tests for the new LSF hfstore methods |

---

### Task 1: Create hfpull LSF command.sh

**Files:**
- Create: `src/gbserver/builtins/steps/hfpull/lsf_scripts/hfpull/command.sh`

- [ ] **Step 1: Create the lsf_scripts directory and command.sh**

```bash
mkdir -p src/gbserver/builtins/steps/hfpull/lsf_scripts/hfpull
```

Write `src/gbserver/builtins/steps/hfpull/lsf_scripts/hfpull/command.sh`:

```bash
#!/usr/bin/env bash

# ===============================================
echo 'hfpull start'

# --------------------------------------------------------------------------

{%- set hfp = config.hfpull_config %}

HF_DEST='{{ hfp.path }}'
HF_URI='{{ hfp.uri }}'
HF_REPO='{{ hfp.owner }}/{{ hfp.repo }}'
HF_REVISION='{{ hfp.revision }}'

if [[ -z "$HF_TOKEN" ]]; then
    echo 'HF_TOKEN is not set'
    exit 1
fi

# --------------------------------------------------------------------------
# Environment variables

echo "LLMB_LSF_LAUNCH_ID ${LLMB_LSF_LAUNCH_ID}"
echo "LLMB_LSF_ASSET_DIR ${LLMB_LSF_ASSET_DIR}"
echo "LLMB_LSF_QUEUE ${LLMB_LSF_QUEUE}"
echo "LLMB_LSF_JOB_NAME ${LLMB_LSF_JOB_NAME}"
echo "LLMB_LSF_WORKSPACE_DIR ${LLMB_LSF_WORKSPACE_DIR}"
echo "LLMB_LSF_OUTPUT_DIR ${LLMB_LSF_OUTPUT_DIR}"
echo "LLMB_LSF_LOG_FILE_STDOUT ${LLMB_LSF_LOG_FILE_STDOUT}"
echo "LLMB_LSF_LOG_FILE_STDERR ${LLMB_LSF_LOG_FILE_STDERR}"
echo "LLMB_LSF_SCRIPT_PATH ${LLMB_LSF_SCRIPT_PATH}"
echo "LLMB_LSF_NUM_NODES ${LLMB_LSF_NUM_NODES}"
echo "LLMB_LSF_NUM_CPUS ${LLMB_LSF_NUM_CPUS}"
echo "LLMB_LSF_NUM_GPUS ${LLMB_LSF_NUM_GPUS}"
echo "LLMB_LSF_MEMORY_SIZE ${LLMB_LSF_MEMORY_SIZE}"
echo "LLMB_LSF_BUILD_ID ${LLMB_LSF_BUILD_ID}"
echo "LLMB_LSF_TARGET_RUN_ID ${LLMB_LSF_TARGET_RUN_ID}"
echo "LLMB_LSF_TARGET_STEP_RUN_ID ${LLMB_LSF_TARGET_STEP_RUN_ID}"
echo "LLMB_LSF_TARGET_NAME ${LLMB_LSF_TARGET_NAME}"

# --------------------------------------------------------------------------

echo "Pulling HF URI: ${HF_URI} to path ${HF_DEST}"

REVISION_FLAG=""
if [[ -n "${HF_REVISION}" ]]; then
    REVISION_FLAG="--revision ${HF_REVISION}"
fi

echo huggingface-cli download "${HF_REPO}" --local-dir "${HF_DEST}" ${REVISION_FLAG}
huggingface-cli download "${HF_REPO}" --local-dir "${HF_DEST}" ${REVISION_FLAG}

# --------------------------------------------------------------------------

MY_EXIT_CODE=$?
if [[ "${MY_EXIT_CODE}" != '0' ]]; then
    echo "${LLMB_LSF_JOB_NAME}: hfpull failed, exit code: ${MY_EXIT_CODE}"
    exit 1
fi

echo "Pulled HF URI: ${HF_URI} to path ${HF_DEST}"

echo 'hfpull end'
# ===============================================
```

- [ ] **Step 2: Commit**

```bash
git add src/gbserver/builtins/steps/hfpull/lsf_scripts/hfpull/command.sh
git commit -m "feat: add hfpull LSF command.sh script"
```

---

### Task 2: Create hfpush LSF command.sh

**Files:**
- Create: `src/gbserver/builtins/steps/hfpush/lsf_scripts/hfpush/command.sh`

- [ ] **Step 1: Create the lsf_scripts directory and command.sh**

```bash
mkdir -p src/gbserver/builtins/steps/hfpush/lsf_scripts/hfpush
```

Write `src/gbserver/builtins/steps/hfpush/lsf_scripts/hfpush/command.sh`:

```bash
#!/usr/bin/env bash

# ===============================================
echo 'hfpush start'

# --------------------------------------------------------------------------

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

# --------------------------------------------------------------------------
# Environment variables

echo "LLMB_LSF_LAUNCH_ID ${LLMB_LSF_LAUNCH_ID}"
echo "LLMB_LSF_ASSET_DIR ${LLMB_LSF_ASSET_DIR}"
echo "LLMB_LSF_QUEUE ${LLMB_LSF_QUEUE}"
echo "LLMB_LSF_JOB_NAME ${LLMB_LSF_JOB_NAME}"
echo "LLMB_LSF_WORKSPACE_DIR ${LLMB_LSF_WORKSPACE_DIR}"
echo "LLMB_LSF_OUTPUT_DIR ${LLMB_LSF_OUTPUT_DIR}"
echo "LLMB_LSF_LOG_FILE_STDOUT ${LLMB_LSF_LOG_FILE_STDOUT}"
echo "LLMB_LSF_LOG_FILE_STDERR ${LLMB_LSF_LOG_FILE_STDERR}"
echo "LLMB_LSF_SCRIPT_PATH ${LLMB_LSF_SCRIPT_PATH}"
echo "LLMB_LSF_NUM_NODES ${LLMB_LSF_NUM_NODES}"
echo "LLMB_LSF_NUM_CPUS ${LLMB_LSF_NUM_CPUS}"
echo "LLMB_LSF_NUM_GPUS ${LLMB_LSF_NUM_GPUS}"
echo "LLMB_LSF_MEMORY_SIZE ${LLMB_LSF_MEMORY_SIZE}"
echo "LLMB_LSF_BUILD_ID ${LLMB_LSF_BUILD_ID}"
echo "LLMB_LSF_TARGET_RUN_ID ${LLMB_LSF_TARGET_RUN_ID}"
echo "LLMB_LSF_TARGET_STEP_RUN_ID ${LLMB_LSF_TARGET_STEP_RUN_ID}"
echo "LLMB_LSF_TARGET_NAME ${LLMB_LSF_TARGET_NAME}"

# --------------------------------------------------------------------------

echo "Pushing HF URI: ${HF_URI} from path ${HF_SOURCE}"

REVISION_FLAG=""
if [[ -n "${HF_REVISION}" ]]; then
    REVISION_FLAG="--revision ${HF_REVISION}"
fi

PRIVATE_FLAG=""
if [[ "${HF_PRIVATE}" == "True" ]]; then
    PRIVATE_FLAG="--private"
fi

echo huggingface-cli upload "${HF_REPO}" "${HF_SOURCE}" ${REVISION_FLAG} ${PRIVATE_FLAG}
huggingface-cli upload "${HF_REPO}" "${HF_SOURCE}" ${REVISION_FLAG} ${PRIVATE_FLAG}

# --------------------------------------------------------------------------

MY_EXIT_CODE=$?
if [[ "${MY_EXIT_CODE}" != '0' ]]; then
    echo "${LLMB_LSF_JOB_NAME}: hfpush failed, exit code: ${MY_EXIT_CODE}"
    exit 1
fi

echo "Pushed HF URI: ${HF_URI} for binding ${BINDING_ID}"

echo 'hfpush end'
# ===============================================
```

- [ ] **Step 2: Commit**

```bash
git add src/gbserver/builtins/steps/hfpush/lsf_scripts/hfpush/command.sh
git commit -m "feat: add hfpush LSF command.sh script"
```

---

### Task 3: Update step.yaml files with Lsf environment_configs

**Files:**
- Modify: `src/gbserver/builtins/steps/hfpull/step.yaml`
- Modify: `src/gbserver/builtins/steps/hfpush/step.yaml`

- [ ] **Step 1: Add Lsf section to hfpull/step.yaml**

Append after the existing `K8s` environment_configs block (after line 36):

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

The full file should end:

```yaml
environment_configs:
  K8s:
    launchers:
      hfpull:
        type: helm
        monitors:
        - log_monitor
        config:
          chart: helm-charts/hfpull   
    monitors:
      log_monitor:
        type: sidecar_monitor
        config:
          event_configs: {}
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

- [ ] **Step 2: Add Lsf section to hfpush/step.yaml**

Append after the existing `K8s` environment_configs block (after line 52):

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

- [ ] **Step 3: Commit**

```bash
git add src/gbserver/builtins/steps/hfpull/step.yaml src/gbserver/builtins/steps/hfpush/step.yaml
git commit -m "feat: add Lsf environment_configs to hfpull and hfpush step.yaml"
```

---

### Task 4: Add pullasset_hfstore and pushasset_hfstore to lsf.py

**Files:**
- Modify: `src/gbserver/environment/lsf.py`

- [ ] **Step 1: Add imports**

Add `HfURI` import alongside existing URI imports (after line 34 `from gbcommon.uri.lh import LhURI`):

```python
from gbcommon.uri.hf import HfURI
```

Add `Hfstore` import alongside existing store imports (after line 39 `from gbserver.asset.lhstore import Lhstore`):

```python
from gbserver.asset.hfstore import Hfstore
```

Add HF constants to the existing constants import block (after line 57, within the `from gbserver.types.constants import (...)` block):

```python
    CODE_GBSERVER_BUILTINS_STEPS_HFPULL_DIR,
    CODE_GBSERVER_BUILTINS_STEPS_HFPULL_URI,
    CODE_GBSERVER_BUILTINS_STEPS_HFPUSH_DIR,
    CODE_GBSERVER_BUILTINS_STEPS_HFPUSH_URI,
```

- [ ] **Step 2: Add `_load_builtin_hf_lsf_section` helper method**

Insert after `_load_builtin_lh_lsf_section` (after line 1027). This method is nearly identical to `_load_builtin_lh_lsf_section` but injects `HF_TOKEN` instead of `LAKEHOUSE_TOKEN`:

```python
    def _load_builtin_hf_lsf_section(
        self: Self, step_dir: Path, hf_metadata: dict
    ) -> Tuple[dict, dict]:
        """Read a builtin HF step YAML and return (lsf_section_dict, workload_section_dict)
        with the HF_TOKEN secret injected into the LSF section.

        Args:
            step_dir: Directory containing the builtin step YAML (e.g. hfpull or hfpush dir).
            hf_metadata: HF metadata dict; must contain 'token_secretname'.
        Returns:
            Tuple of (lsf_dict, workload_dict) ready for BuildTargetStepConfig config.
        Raises:
            AssertionError: if the step YAML is missing or 'token_secretname' is absent.
        """
        step_path = step_dir / STEP_FILE_NAME
        assert step_path.is_file(), f"step yaml is missing: {step_path}"
        step_config = StepConfig.from_yaml(path=step_path)
        hfpc = step_config.config
        assert isinstance(
            hfpc, dict
        ), f"invalid step config type: {type(hfpc).__name__}"
        step_config_section = StepConfigSection.model_validate(hfpc)
        logger.info("step_config_section: %s", step_config_section)
        lsf_dict = step_config_section.lsf.model_dump(exclude_unset=True)
        workload_dict = step_config_section.workload.model_dump(exclude_unset=True)
        assert isinstance(
            hf_metadata, dict
        ), f"invalid hf_metadata type: {type(hf_metadata).__name__}"
        assert (
            "token_secretname" in hf_metadata
        ), "token_secretname is missing in hf_metadata"
        lsf_dict["secrets"] = {
            "secret_names_to_use_as_env_variable": [
                {
                    "env_name": "HF_TOKEN",
                    "secret_name": hf_metadata["token_secretname"],
                }
            ]
        }
        lsf_dict["skip_finding_output_artifacts"] = True
        return lsf_dict, workload_dict
```

- [ ] **Step 3: Add `pullasset_hfstore` method**

Insert after `_load_builtin_hf_lsf_section`. This follows the same structure as `pullasset_lhstore`:

```python
    async def pullasset_hfstore(
        self: Self,
        uri: URI,
        binding: Optional[Any] = None,
        storeload_config: Optional[StoreLoad] = None,
        assetstore: Optional[Assetstore] = None,
        secrets: Optional[dict] = None,
        **kwargs: Dict,
    ) -> Tuple[Dict, Optional[BuildTargetStepConfig]]:
        """Pull an asset from Hugging Face Hub to LSF cluster storage.

        Args:
            uri: HF URI to pull (e.g. hf://models/org/repo).
            binding: Unused for hfpull.
            storeload_config: Must have mode 'hf_pull' and config with 'cache_path'.
            assetstore: Hfstore instance.
            secrets: Optional secrets dict.
        Returns:
            Tuple of (binding_config, BuildTargetStepConfig).
        Raises:
            AssertionError: If assetstore type or mode is invalid.
        """
        assert isinstance(
            assetstore, Hfstore
        ), f"invalid type assetstore: {type(assetstore).__name__} (expected 'Hfstore')"
        assert storeload_config is not None, "storeload_config is None"
        assert (
            storeload_config.mode == "hf_pull"
        ), f"Only 'hf_pull' mode is supported for Lsf, mode: {storeload_config.mode} uri: {uri}"
        cache_path = storeload_config.config.get("cache_path", None)
        assert isinstance(cache_path, str), f"invalid cache_path: {cache_path}"
        assert cache_path != "", f"invalid cache_path: {cache_path}"
        hfuri = uri if isinstance(uri, HfURI) else HfURI.parse(uri)
        binding_path = (
            Path(cache_path) / hfuri.get_owner() / hfuri.get_repo() / hfuri.hash()
        )
        hf_metadata = Asset(uri=hfuri).get_metadata()
        logger.info("hf_metadata: %s", hf_metadata)
        hfpull_config = {
            "path": str(binding_path),
            "uri": str(hfuri),
            "owner": hfuri.get_owner(),
            "repo": hfuri.get_repo(),
            "revision": hfuri.get_revision(),
        }
        logger.info("hfpull_config: %s", hfpull_config)
        hfpull_stepuri = CODE_GBSERVER_BUILTINS_STEPS_HFPULL_URI
        if (
            storeload_config is not None
            and storeload_config.config is not None
            and "step_uri" in storeload_config.config
        ):
            hfpull_stepuri = storeload_config.config["step_uri"]
            assert isinstance(
                hfpull_stepuri, str
            ), f"invalid hfpull_stepuri: {hfpull_stepuri}"
        binding_config = {BINDING_KEY: {"path": str(binding_path)}}
        lsf_dict, workload_dict = self._load_builtin_hf_lsf_section(
            CODE_GBSERVER_BUILTINS_STEPS_HFPULL_DIR, hf_metadata
        )
        return binding_config, BuildTargetStepConfig(
            step_uri=hfpull_stepuri,
            config={
                "lsf": lsf_dict,
                "workload": workload_dict,
                "hfpull_config": hfpull_config,
            },
        )
```

- [ ] **Step 4: Add `pushasset_hfstore` method**

Insert after `pullasset_hfstore`:

```python
    async def pushasset_hfstore(
        self: Self,
        binding: Any,
        binding_id: Optional[str] = "",
        storepush_config: Optional[StorePush] = None,
        uri: Optional[Union[str, URI]] = None,
        assetstore: Optional[Assetstore] = None,
        **kwargs: Dict,
    ) -> BuildTargetStepConfig:
        """Push an artifact from LSF cluster storage to Hugging Face Hub.

        Args:
            binding: Dict with a 'path' key pointing to the artifact on cluster.
            binding_id: Output binding name for artifact tracking.
            storepush_config: Environment-level push configuration.
            uri: Target HF URI string or object.
            assetstore: Hfstore instance.
        Returns:
            BuildTargetStepConfig for the hfpush step.
        Raises:
            ValueError: If uri is empty.
            AssertionError: If binding has no 'path'.
        """
        if uri is None or uri == "":
            raise ValueError(f"Empty uri received to pushasset {binding}")
        hfuri = uri if isinstance(uri, HfURI) else HfURI.parse(uri)
        logger.info("binding type %s value %s", type(binding), binding)
        assert isinstance(
            binding, dict
        ), f"expected binding to be a dict, actual: {type(binding).__name__} {binding}"
        assert (
            "path" in binding
        ), f"expected 'path' to be in the binding, actual: {binding}"
        binding_path = binding["path"]
        logger.info("binding_path: %s", binding_path)
        hf_metadata = Asset(uri=hfuri).get_metadata()
        logger.info("hf_metadata: %s", hf_metadata)

        hf_private = True
        if (
            storepush_config is not None
            and storepush_config.config is not None
            and "hf" in storepush_config.config
        ):
            hf_private = storepush_config.config["hf"].get("private", hf_private)

        hfpush_config = {
            "path": binding_path,
            "uri": str(hfuri),
            "binding_id": binding_id,
            "owner": hfuri.get_owner(),
            "repo": hfuri.get_repo(),
            "revision": hfuri.get_revision(),
            "private": hf_private,
        }
        logger.info("hfpush_config: %s", hfpush_config)
        hfpush_stepuri = CODE_GBSERVER_BUILTINS_STEPS_HFPUSH_URI
        if (
            storepush_config is not None
            and storepush_config.config is not None
            and "step_uri" in storepush_config.config
        ):
            hfpush_stepuri = storepush_config.config["step_uri"]
            assert isinstance(
                hfpush_stepuri, str
            ), f"invalid hfpush_stepuri: {hfpush_stepuri}"
        lsf_dict, workload_dict = self._load_builtin_hf_lsf_section(
            CODE_GBSERVER_BUILTINS_STEPS_HFPUSH_DIR, hf_metadata
        )
        return BuildTargetStepConfig(
            step_uri=hfpush_stepuri,
            config={
                "lsf": lsf_dict,
                "workload": workload_dict,
                "hfpush_config": hfpush_config,
            },
        )
```

- [ ] **Step 5: Commit**

```bash
git add src/gbserver/environment/lsf.py
git commit -m "feat: add pullasset_hfstore and pushasset_hfstore to LSF environment"
```

---

### Task 5: Write unit tests

**Files:**
- Create: `test/unit/environment/test_lsf_hfstore.py`

- [ ] **Step 1: Create test directory if needed**

```bash
mkdir -p test/unit/environment
touch test/unit/environment/__init__.py
```

- [ ] **Step 2: Write tests**

Write `test/unit/environment/test_lsf_hfstore.py`:

```python
"""Unit tests for Lsf.pullasset_hfstore and Lsf.pushasset_hfstore."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gbserver.environment.environment import BINDING_KEY
from gbserver.types.buildconfig import BuildTargetStepConfig


@pytest.fixture
def lsf_env():
    """Create a minimal Lsf environment instance for testing asset methods."""
    from gbserver.environment.lsf import Lsf

    event_q = asyncio.Queue()
    env_config = MagicMock()
    env_config.config = {
        "workspace": {"local_dir": "/tmp/lsf_test", "remote_dir": "/remote/test"},
        "authentication": {"use_ssh": False, "login_nodes": []},
    }
    env_config.type = "Lsf"
    with patch(
        "gbserver.environment.environment.Environment.__init__", return_value=None
    ):
        lsf = Lsf(event_q=event_q, environment_config=env_config)
    return lsf


@pytest.fixture
def mock_hf_metadata():
    """Return metadata as Asset(uri=hfuri).get_metadata() would."""
    return {
        "uri": "hf://models/myorg/myrepo",
        "host": "huggingface.co",
        "owner": "myorg",
        "repo": "myrepo",
        "revision": "main",
        "hf_type": "model",
        "token_secretname": "HF_TOKEN",
    }


@pytest.fixture
def mock_hfuri():
    """Return a mock HfURI."""
    uri = MagicMock()
    uri.get_owner.return_value = "myorg"
    uri.get_repo.return_value = "myrepo"
    uri.get_revision.return_value = "main"
    uri.hash.return_value = "abc123hash"
    uri.__str__ = lambda self: "hf://models/myorg/myrepo"
    return uri


class TestPullassetHfstore:
    @pytest.mark.asyncio
    async def test_returns_binding_config_with_path(
        self, lsf_env, mock_hfuri, mock_hf_metadata
    ):
        """pullasset_hfstore returns a binding_config with the expected cache path."""
        from gbserver.asset.hfstore import Hfstore

        assetstore = MagicMock(spec=Hfstore)
        storeload_config = MagicMock()
        storeload_config.mode = "hf_pull"
        storeload_config.config = {"cache_path": "/data/cache"}

        with (
            patch("gbserver.environment.lsf.HfURI.parse", return_value=mock_hfuri),
            patch(
                "gbserver.environment.lsf.Asset") as mock_asset_cls,
            patch.object(lsf_env, "_load_builtin_hf_lsf_section", return_value=({}, {})),
        ):
            mock_asset_cls.return_value.get_metadata.return_value = mock_hf_metadata
            binding_config, step_config = await lsf_env.pullasset_hfstore(
                uri=mock_hfuri,
                assetstore=assetstore,
                storeload_config=storeload_config,
            )

        assert BINDING_KEY in binding_config
        expected_path = str(Path("/data/cache/myorg/myrepo/abc123hash"))
        assert binding_config[BINDING_KEY]["path"] == expected_path

    @pytest.mark.asyncio
    async def test_returns_build_target_step_config(
        self, lsf_env, mock_hfuri, mock_hf_metadata
    ):
        """pullasset_hfstore returns a BuildTargetStepConfig with hfpull_config."""
        from gbserver.asset.hfstore import Hfstore

        assetstore = MagicMock(spec=Hfstore)
        storeload_config = MagicMock()
        storeload_config.mode = "hf_pull"
        storeload_config.config = {"cache_path": "/data/cache"}

        with (
            patch("gbserver.environment.lsf.HfURI.parse", return_value=mock_hfuri),
            patch("gbserver.environment.lsf.Asset") as mock_asset_cls,
            patch.object(
                lsf_env,
                "_load_builtin_hf_lsf_section",
                return_value=({"secrets": {}}, {"cwd": "."}),
            ),
        ):
            mock_asset_cls.return_value.get_metadata.return_value = mock_hf_metadata
            _, step_config = await lsf_env.pullasset_hfstore(
                uri=mock_hfuri,
                assetstore=assetstore,
                storeload_config=storeload_config,
            )

        assert isinstance(step_config, BuildTargetStepConfig)
        assert "hfpull_config" in step_config.config
        hfpull_config = step_config.config["hfpull_config"]
        assert hfpull_config["owner"] == "myorg"
        assert hfpull_config["repo"] == "myrepo"
        assert hfpull_config["revision"] == "main"
        assert "lsf" in step_config.config
        assert "workload" in step_config.config

    @pytest.mark.asyncio
    async def test_rejects_wrong_assetstore_type(self, lsf_env, mock_hfuri):
        """pullasset_hfstore raises AssertionError if assetstore is not Hfstore."""
        storeload_config = MagicMock()
        storeload_config.mode = "hf_pull"
        storeload_config.config = {"cache_path": "/data/cache"}

        with pytest.raises(AssertionError, match="expected 'Hfstore'"):
            await lsf_env.pullasset_hfstore(
                uri=mock_hfuri,
                assetstore=MagicMock(),  # not spec=Hfstore
                storeload_config=storeload_config,
            )

    @pytest.mark.asyncio
    async def test_rejects_wrong_mode(self, lsf_env, mock_hfuri):
        """pullasset_hfstore raises AssertionError for non-hf_pull mode."""
        from gbserver.asset.hfstore import Hfstore

        assetstore = MagicMock(spec=Hfstore)
        storeload_config = MagicMock()
        storeload_config.mode = "dmf_pull"
        storeload_config.config = {"cache_path": "/data/cache"}

        with pytest.raises(AssertionError, match="Only 'hf_pull' mode"):
            await lsf_env.pullasset_hfstore(
                uri=mock_hfuri,
                assetstore=assetstore,
                storeload_config=storeload_config,
            )


class TestPushassetHfstore:
    @pytest.mark.asyncio
    async def test_returns_build_target_step_config(
        self, lsf_env, mock_hfuri, mock_hf_metadata
    ):
        """pushasset_hfstore returns BuildTargetStepConfig with hfpush_config."""
        with (
            patch("gbserver.environment.lsf.HfURI.parse", return_value=mock_hfuri),
            patch("gbserver.environment.lsf.Asset") as mock_asset_cls,
            patch.object(
                lsf_env,
                "_load_builtin_hf_lsf_section",
                return_value=({"secrets": {}}, {"cwd": "."}),
            ),
        ):
            mock_asset_cls.return_value.get_metadata.return_value = mock_hf_metadata
            step_config = await lsf_env.pushasset_hfstore(
                binding={"path": "/workspace/output/model"},
                binding_id="output_model",
                uri=mock_hfuri,
            )

        assert isinstance(step_config, BuildTargetStepConfig)
        assert "hfpush_config" in step_config.config
        hfpush_config = step_config.config["hfpush_config"]
        assert hfpush_config["path"] == "/workspace/output/model"
        assert hfpush_config["binding_id"] == "output_model"
        assert hfpush_config["owner"] == "myorg"
        assert hfpush_config["repo"] == "myrepo"
        assert hfpush_config["private"] is True
        assert "lsf" in step_config.config
        assert "workload" in step_config.config

    @pytest.mark.asyncio
    async def test_raises_on_empty_uri(self, lsf_env):
        """pushasset_hfstore raises ValueError for empty uri."""
        with pytest.raises(ValueError, match="Empty uri"):
            await lsf_env.pushasset_hfstore(
                binding={"path": "/workspace/output"},
                uri=None,
            )

    @pytest.mark.asyncio
    async def test_raises_on_missing_path_in_binding(
        self, lsf_env, mock_hfuri, mock_hf_metadata
    ):
        """pushasset_hfstore raises AssertionError if binding lacks 'path'."""
        with (
            patch("gbserver.environment.lsf.HfURI.parse", return_value=mock_hfuri),
            patch("gbserver.environment.lsf.Asset") as mock_asset_cls,
        ):
            mock_asset_cls.return_value.get_metadata.return_value = mock_hf_metadata
            with pytest.raises(AssertionError, match="expected 'path'"):
                await lsf_env.pushasset_hfstore(
                    binding={},
                    uri=mock_hfuri,
                )

    @pytest.mark.asyncio
    async def test_private_flag_from_storepush_config(
        self, lsf_env, mock_hfuri, mock_hf_metadata
    ):
        """pushasset_hfstore picks up private=False from storepush_config."""
        storepush_config = MagicMock()
        storepush_config.config = {"hf": {"private": False}}

        with (
            patch("gbserver.environment.lsf.HfURI.parse", return_value=mock_hfuri),
            patch("gbserver.environment.lsf.Asset") as mock_asset_cls,
            patch.object(
                lsf_env,
                "_load_builtin_hf_lsf_section",
                return_value=({"secrets": {}}, {"cwd": "."}),
            ),
        ):
            mock_asset_cls.return_value.get_metadata.return_value = mock_hf_metadata
            step_config = await lsf_env.pushasset_hfstore(
                binding={"path": "/workspace/output/model"},
                binding_id="output_model",
                uri=mock_hfuri,
                storepush_config=storepush_config,
            )

        assert step_config.config["hfpush_config"]["private"] is False


class TestLoadBuiltinHfLsfSection:
    def test_injects_hf_token_secret(self, lsf_env, tmp_path):
        """_load_builtin_hf_lsf_section injects HF_TOKEN into lsf_dict secrets."""
        step_yaml_content = """
name: hfpull
version: 1.0.0
type: upload
config:
  workload:
    cwd: "."
  compute_config:
    total_memory_per_node: 10Gi
environment_configs:
  Lsf:
    launchers:
      tuning:
        type: bsub
        monitors:
        - bsub_monitor
    monitors:
      bsub_monitor:
        type: bsub_monitor
"""
        step_file = tmp_path / "step.yaml"
        step_file.write_text(step_yaml_content)

        hf_metadata = {"token_secretname": "MY_HF_SECRET"}
        with patch("gbserver.environment.lsf.STEP_FILE_NAME", "step.yaml"):
            lsf_dict, workload_dict = lsf_env._load_builtin_hf_lsf_section(
                tmp_path, hf_metadata
            )

        assert lsf_dict["skip_finding_output_artifacts"] is True
        secrets = lsf_dict["secrets"]["secret_names_to_use_as_env_variable"]
        assert len(secrets) == 1
        assert secrets[0]["env_name"] == "HF_TOKEN"
        assert secrets[0]["secret_name"] == "MY_HF_SECRET"

    def test_raises_on_missing_token_secretname(self, lsf_env, tmp_path):
        """_load_builtin_hf_lsf_section raises if token_secretname missing."""
        step_yaml_content = """
name: hfpull
version: 1.0.0
type: upload
config:
  workload:
    cwd: "."
  compute_config:
    total_memory_per_node: 10Gi
environment_configs:
  Lsf:
    launchers:
      tuning:
        type: bsub
        monitors:
        - bsub_monitor
    monitors:
      bsub_monitor:
        type: bsub_monitor
"""
        step_file = tmp_path / "step.yaml"
        step_file.write_text(step_yaml_content)

        with patch("gbserver.environment.lsf.STEP_FILE_NAME", "step.yaml"):
            with pytest.raises(AssertionError, match="token_secretname is missing"):
                lsf_env._load_builtin_hf_lsf_section(tmp_path, {})
```

- [ ] **Step 3: Run the tests**

Run: `pytest test/unit/environment/test_lsf_hfstore.py -v`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add test/unit/environment/
git commit -m "test: add unit tests for LSF hfstore pull/push methods"
```

---

### Task 6: Verify type-checking passes

- [ ] **Step 1: Run mypy on the modified file**

```bash
mypy --disable-error-code=import-untyped src/gbserver/environment/lsf.py
```

Expected: No new errors introduced.

- [ ] **Step 2: Run formatting**

```bash
make xformat
```

- [ ] **Step 3: Commit any formatting fixes**

```bash
git add -u
git commit -m "style: apply formatting to lsf hfstore changes"
```
