"""Unit tests for K8s.pullasset_hfstore and the shared step-chart permission settings.

Two things are covered here:

1. ``K8s.pullasset_hfstore`` — the binding path layout and the hfpull step config.
   The bash/docker/lsf/skypilot environments all had hfstore tests; k8s did not.
2. The chart-level permission settings that make the shared PVC usable from a pod
   running as an arbitrary OpenShift UID: the ``umask`` applied in every step
   container, and ``runAsGroup: 0``. These are asserted against the template
   sources because ``helm`` is not available in every test environment.
"""

import asyncio
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gbcommon.uri.hf import HfURI
from gbserver.environment.environment import BINDING_KEY
from gbserver.types.buildconfig import BuildTargetStepConfig

# The step chart's directory name is itself a Jinja template.
CHART_DIR = (
    Path(__file__).resolve().parents[3]
    / "src/gbserver/builtins/steps/gbstep/helm-charts"
    / "{{ step.name | default(run_metadata.target_name) }}"
)
CONTAINER_TEMPLATES = (
    CHART_DIR / "charts/gbstepbase/templates/_single_container.tpl",
    CHART_DIR / "charts/gbstepbase/templates/_multi_containers.tpl",
    CHART_DIR / "charts/gbraystepbase/templates/_helpers.tpl",
)


@pytest.fixture
def k8s_env():
    """Create a minimal K8s environment instance for testing asset methods."""
    from gbserver.environment.k8s import K8s

    event_q = asyncio.Queue()
    env_config = MagicMock()
    env_config.config = {"namespace": "gb-test"}
    env_config.type = "K8s"

    # K8s.__init__ reads self.config/self.secrets (normally set by
    # Environment.__init__, patched out here to avoid touching storage).
    def fake_init(self, *_args, **_kwargs):
        self.config = env_config
        self.secrets = {}

    with patch("gbserver.environment.environment.Environment.__init__", new=fake_init):
        k8s = K8s(event_q=event_q, environment_config=env_config)
    return k8s


@pytest.fixture
def mock_hfuri():
    """Return a mock HfURI that passes isinstance checks."""
    uri = MagicMock(spec=HfURI)
    uri.get_owner.return_value = "myorg"
    uri.get_repo.return_value = "myrepo"
    uri.hash.return_value = "abc123hash"
    uri.__str__ = lambda self: "hf://models/myorg/myrepo"
    return uri


class TestPullassetHfstore:
    @pytest.mark.asyncio
    async def test_returns_binding_config_with_path(self, k8s_env, mock_hfuri):
        """The binding path is <cache_path>/<owner>/<repo>/<hash>."""
        from gbserver.asset.hfstore import Hfstore

        storeload_config = MagicMock()
        storeload_config.mode = "default"
        storeload_config.config = {"cache_path": "/gb-read-write/hfcache"}

        binding_config, step_config = await k8s_env.pullasset_hfstore(
            uri=mock_hfuri,
            assetstore=MagicMock(spec=Hfstore),
            storeload_config=storeload_config,
        )

        assert BINDING_KEY in binding_config
        expected = str(Path("/gb-read-write/hfcache/myorg/myrepo/abc123hash"))
        assert binding_config[BINDING_KEY]["path"] == expected

    @pytest.mark.asyncio
    async def test_returns_build_target_step_config(self, k8s_env, mock_hfuri):
        """A BuildTargetStepConfig carrying hfpull_config is returned."""
        from gbserver.asset.hfstore import Hfstore

        storeload_config = MagicMock()
        storeload_config.mode = "default"
        storeload_config.config = {"cache_path": "/gb-read-write/hfcache"}

        _, step_config = await k8s_env.pullasset_hfstore(
            uri=mock_hfuri,
            assetstore=MagicMock(spec=Hfstore),
            storeload_config=storeload_config,
        )

        assert isinstance(step_config, BuildTargetStepConfig)
        assert step_config.step_uri == "space://steps/hfpull"
        assert "hfpull_config" in step_config.config
        assert step_config.config["hfpull_config"]["path"] == str(
            Path("/gb-read-write/hfcache/myorg/myrepo/abc123hash")
        )

    @pytest.mark.asyncio
    async def test_honours_custom_step_uri(self, k8s_env, mock_hfuri):
        """A storeload-configured step_uri overrides the default hfpull step."""
        from gbserver.asset.hfstore import Hfstore

        storeload_config = MagicMock()
        storeload_config.mode = "default"
        storeload_config.config = {
            "cache_path": "/gb-read-write/hfcache",
            "step_uri": "space://steps/myhfpull",
        }

        _, step_config = await k8s_env.pullasset_hfstore(
            uri=mock_hfuri,
            assetstore=MagicMock(spec=Hfstore),
            storeload_config=storeload_config,
        )

        assert step_config.step_uri == "space://steps/myhfpull"

    @pytest.mark.asyncio
    async def test_raises_on_missing_cache_path(self, k8s_env, mock_hfuri):
        """cache_path is required for k8s — there is no default."""
        from gbserver.asset.hfstore import Hfstore

        storeload_config = MagicMock()
        storeload_config.mode = "default"
        storeload_config.config = {}

        with pytest.raises(ValueError, match="cache_path"):
            await k8s_env.pullasset_hfstore(
                uri=mock_hfuri,
                assetstore=MagicMock(spec=Hfstore),
                storeload_config=storeload_config,
            )

    @pytest.mark.asyncio
    async def test_rejects_wrong_assetstore_type(self, k8s_env, mock_hfuri):
        """A non-Hfstore assetstore is rejected."""
        storeload_config = MagicMock()
        storeload_config.mode = "default"
        storeload_config.config = {"cache_path": "/gb-read-write/hfcache"}

        with pytest.raises(AssertionError, match="invalid assetstore"):
            await k8s_env.pullasset_hfstore(
                uri=mock_hfuri,
                assetstore=MagicMock(),
                storeload_config=storeload_config,
            )


class TestSharedPvcPermissions:
    """Guards for the group-writable shared PVC settings.

    Pods run as an arbitrary OpenShift UID but share GID 0, so shared state is
    only reusable if the permission bits allow group writes. ``runAsGroup: 0``
    supplies the group; the umask supplies the bits.
    """

    @pytest.mark.parametrize("template", CONTAINER_TEMPLATES, ids=lambda p: p.name)
    def test_umask_applied_before_workload(self, template):
        """Every step container sets the umask, immediately after `set -o pipefail`.

        Ordering matters: the umask has to be in effect before anything creates
        files, including the heredoc that writes command.sh.
        """
        lines = template.read_text(encoding="utf-8").splitlines()
        idx = [i for i, ln in enumerate(lines) if ln.strip() == "set -o pipefail"]
        assert len(idx) == 1, f"expected one `set -o pipefail` in {template.name}"
        nxt = lines[idx[0] + 1].strip()
        assert nxt.startswith("umask "), f"{template.name}: umask must follow pipefail"
        # Must fall back to a group-writable default for charts whose values predate this.
        assert '| default "0002"' in nxt, f"{template.name}: missing 0002 default"

    def test_umask_default_is_a_quoted_string(self):
        """An unquoted 0002 would be parsed by YAML as the integer 2.

        Guards the octal trap: `umask 0022` written unquoted becomes `umask 18`.
        """
        text = (CHART_DIR / "values-default.yaml").read_text(encoding="utf-8")
        match = re.search(r"^\s*umask:\s*(.+)$", text, re.MULTILINE)
        assert match, "no umask key in values-default.yaml"
        assert match.group(1).startswith('"'), "umask value must be quoted"
        assert "'0002'" in match.group(1), "default umask should be 0002"

    def test_both_container_templates_set_run_as_root_group(self):
        """Single- and multi-container paths must both honour run_as_root_group.

        The multi-container template previously only added IPC_LOCK, so
        multi-container steps never got GID 0 and a umask alone would not have
        made their files reusable.
        """
        for template in CONTAINER_TEMPLATES[:2]:
            text = template.read_text(encoding="utf-8")
            assert "run_as_root_group" in text, f"{template.name}: no gate"
            assert "runAsGroup: 0" in text, f"{template.name}: no runAsGroup"
