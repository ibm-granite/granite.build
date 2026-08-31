"""Unit tests for K8s.pullasset_hfstore and the shared step-chart permission settings.

Two things are covered here:

1. ``K8s.pullasset_hfstore`` — the binding path layout and the hfpull step config.
   The bash/docker/lsf/skypilot environments all had hfstore tests; k8s did not.
2. The chart-level permission settings that make the shared PVC usable from a pod
   running as an arbitrary OpenShift UID: the ``umask`` applied in the step
   containers, and ``runAsGroup: 0``. Asserted against the template sources (and,
   for the umask guard, by running the rendered bash) so the checks need no helm
   binary; a full ``helm template`` render is a separate manual/CI step.
"""

import re
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# gbserver.environment.k8s imports kubernetes_asyncio (via its retry strategies),
# which lives in the optional 'ibm' extra and is absent from the lightweight
# quick-test CI venv. Only the pullasset tests import it; the template checks
# below read files and need no skip.
from libgbtest.constants import requires_k8s

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
    """A stand-in exposing K8s.pullasset_hfstore, without constructing a K8s.

    ``pullasset_hfstore`` never touches ``self`` -- it derives everything from its
    arguments -- so binding the unbound method to a bare object exercises the real
    code while avoiding ``Environment.__init__``, which pulls in storage, plugin
    type discovery and the node-health singleton (all unavailable or unreliable
    under xdist in CI).
    """
    from gbserver.environment.k8s import K8s

    class _K8sStandin:
        pullasset_hfstore = K8s.pullasset_hfstore

    return _K8sStandin()


@pytest.fixture
def mock_hfuri():
    """Return a mock HfURI that passes isinstance checks."""
    uri = MagicMock(spec=HfURI)
    uri.get_owner.return_value = "myorg"
    uri.get_repo.return_value = "myrepo"
    uri.hash.return_value = "abc123hash"
    uri.__str__ = lambda self: "hf://models/myorg/myrepo"
    return uri


@requires_k8s
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

    @staticmethod
    def _umask_guard(template: Path) -> list[str]:
        """Return the umask guard block from a container template, de-indented."""
        lines = template.read_text(encoding="utf-8").splitlines()
        starts = [i for i, ln in enumerate(lines) if ln.strip().startswith("GB_UMASK=")]
        assert len(starts) == 1, f"expected one GB_UMASK block in {template.name}"
        i = starts[0]
        end = next(n for n in range(i, len(lines)) if lines[n].strip() == "fi")
        indent = len(lines[i]) - len(lines[i].lstrip())
        return [ln[indent:] for ln in lines[i : end + 1]]

    @pytest.mark.parametrize("template", CONTAINER_TEMPLATES, ids=lambda p: p.name)
    def test_umask_set_before_anything_creates_files(self, template):
        """The umask guard runs before the workload, and after `set -o pipefail`.

        Ordering matters: the umask must be in effect before anything creates
        files, including the heredoc that writes command.sh.
        """
        lines = template.read_text(encoding="utf-8").splitlines()
        pipefail = [i for i, ln in enumerate(lines) if ln.strip() == "set -o pipefail"]
        gb_umask = [
            i for i, ln in enumerate(lines) if ln.strip().startswith("GB_UMASK=")
        ]
        assert gb_umask, f"{template.name}: no umask guard"
        assert pipefail, f"{template.name}: no `set -o pipefail`"
        # The guard must come after pipefail and before the command.sh heredoc.
        heredoc = [i for i, ln in enumerate(lines) if "command.sh" in ln]
        assert min(pipefail) < gb_umask[0], f"{template.name}: umask precedes pipefail"
        if heredoc:
            assert gb_umask[0] < min(
                heredoc
            ), f"{template.name}: umask after command.sh"

    @pytest.mark.parametrize("template", CONTAINER_TEMPLATES, ids=lambda p: p.name)
    def test_umask_falls_back_to_group_writable_default(self, template):
        """Charts whose values predate `k8s.umask` still get 0002."""
        guard = "\n".join(self._umask_guard(template))
        assert '| default "0002"' in guard, f"{template.name}: missing 0002 default"
        assert "umask 0002" in guard, f"{template.name}: no safe fallback"

    @pytest.mark.parametrize(
        "value,expected,warns",
        [
            ("0002", "0002", False),
            ("0027", "0027", False),
            ("002", "0002", False),
            # YAML coerces an unquoted 0027 to the int 23, which bash would read
            # as octal 0023 — looser, not tighter. Must be rejected, not applied.
            ("23", "0002", True),
            # 0022 unquoted becomes 18, which is not valid octal at all.
            ("18", "0002", True),
            ("2", "0002", True),
            ("abc", "0002", True),
            ("", "0002", True),
        ],
    )
    def test_umask_guard_rejects_malformed_values(self, value, expected, warns):
        """A misconfigured umask must be visible, never silently wrong.

        `set -e` is not in effect in the prologue, so a bare `umask 18` would fail
        and be swallowed, leaving the pod at the image default 0022.
        """
        guard = self._umask_guard(CONTAINER_TEMPLATES[0])
        # Emulate helm substituting the value into the first line.
        script = "\n".join([f'GB_UMASK="{value}"'] + guard[1:])
        proc = subprocess.run(
            ["bash", "-c", f"set -o pipefail\n{script}\necho EFFECTIVE=$(umask)"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert f"EFFECTIVE={expected}" in proc.stdout, proc.stdout
        assert ("WARNING" in proc.stderr) is warns, proc.stderr

    def test_umask_default_is_a_quoted_string(self):
        """An unquoted 0002 would be parsed by YAML as the integer 2."""
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
