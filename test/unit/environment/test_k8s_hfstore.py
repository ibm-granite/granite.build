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


class TestArtifactPermissionNormalization:
    """Guards for the post-workload chmod that makes produced artifacts readable.

    A umask only masks bits off the mode a writer asks for, so a workload that
    explicitly creates a file ``0600`` (safetensors' ``mkstemp`` does) still
    leaves it unreadable to a later pod on a different UID. BYOI steps can run
    anything and set any mode, so the only reliable point of control is after
    the workload exits: normalize the tree where it becomes shared state.
    """

    WORKLOAD_TEMPLATES = CONTAINER_TEMPLATES[:2]
    # The block itself lives in one place; both container templates `include` it.
    UTILS_TEMPLATE = CHART_DIR / "charts/gbstepbase/templates/_utils.tpl"
    INCLUDE_NAME = "gbstepbase.normalizeOutputPermissions"

    @classmethod
    def _epilogue(cls) -> list[str]:
        """Return the shared chmod epilogue's bash body, de-indented.

        Read from the single ``define`` rather than from each container template:
        the block is emitted at column 0 there and the call sites re-indent it,
        which is what makes one copy serve both contexts.
        """
        lines = cls.UTILS_TEMPLATE.read_text(encoding="utf-8").splitlines()
        starts = [
            i
            for i, ln in enumerate(lines)
            if ln.strip() == f'{{{{- define "{cls.INCLUDE_NAME}" }}}}'
        ]
        assert len(starts) == 1, f"expected exactly one {cls.INCLUDE_NAME} define"
        i = starts[0]
        end = next(
            n for n in range(i + 1, len(lines)) if lines[n].strip() == "{{- end }}"
        )
        return lines[i + 1 : end]

    @pytest.mark.parametrize("template", WORKLOAD_TEMPLATES, ids=lambda p: p.name)
    def test_epilogue_is_included_not_duplicated(self, template):
        """Each container template calls the shared define -- it does not inline it.

        The block is ~30 lines of bash; two hand-maintained copies would drift.
        """
        text = template.read_text(encoding="utf-8")
        assert self.INCLUDE_NAME in text, f"{template.name}: does not include the block"
        assert "g+rwX" not in text, f"{template.name}: still inlines the chmod"

    def test_epilogue_defined_exactly_once(self):
        """Exactly one copy of the bash exists across the whole chart."""
        copies = [
            p
            for p in CHART_DIR.rglob("*.tpl")
            if "g+rwX" in p.read_text(encoding="utf-8")
        ]
        assert copies == [self.UTILS_TEMPLATE], f"chmod bash duplicated in {copies}"

    @pytest.mark.parametrize("template", WORKLOAD_TEMPLATES, ids=lambda p: p.name)
    def test_epilogue_runs_after_workload_but_before_exit_check(self, template):
        """Ordering: after the workload's exit code is captured, before it is acted on.

        It must follow the workload (there is nothing to fix before that) and
        precede the ``exit 1``, so a failed run's partial output is normalized
        too -- later pods still read and retry against it.
        """
        lines = template.read_text(encoding="utf-8").splitlines()
        capture = [
            i
            for i, ln in enumerate(lines)
            if ln.strip().startswith("COMMAND_SH_EXIT_CODE=")
        ]
        include = [i for i, ln in enumerate(lines) if self.INCLUDE_NAME in ln]
        check = [
            i
            for i, ln in enumerate(lines)
            if ln.strip().startswith('if [[ "${COMMAND_SH_EXIT_CODE}" != "0" ]]')
        ]
        assert include, f"{template.name}: no epilogue include"
        assert capture, f"{template.name}: no exit-code capture"
        assert check, f"{template.name}: no exit-code check"
        assert (
            min(capture) < include[0]
        ), f"{template.name}: epilogue precedes the workload"
        assert include[0] < min(
            check
        ), f"{template.name}: epilogue after the exit check"

    @staticmethod
    def _stub_chmod(directory: Path, lines: int, exit_code: int) -> Path:
        """Put a ``chmod`` on PATH that emits ``lines`` errors and exits ``exit_code``.

        Stands in for a mount that refuses ``chmod`` -- a read-only or
        root-squashed PVC, or files owned by another UID -- which cannot be
        reproduced as the owning user, since the owner may always chmod its own
        files regardless of the mode bits.
        """
        stub = directory / "chmod"
        body = ["#!/bin/sh"]
        for n in range(lines):
            body.append(
                f'echo "chmod: Unable to change file mode on /out/f{n}.bin:'
                ' Operation not permitted" >&2'
            )
        body.append(f"exit {exit_code}")
        stub.write_text("\n".join(body) + "\n")
        stub.chmod(0o755)
        return stub

    @pytest.mark.parametrize("exit_code", (0, 1), ids=("chmod-exit-0", "chmod-exit-1"))
    def test_epilogue_never_fails_the_step(self, exit_code, tmp_path):
        """A refused ``chmod`` must not fail the step -- the guard is not a gate.

        Some environments forbid ``chmod`` outright while the artifact is already
        perfectly readable, so a non-zero ``chmod`` here says nothing about
        whether the step succeeded.
        """
        block = "\n".join(self._epilogue())
        out = tmp_path / "out"
        out.mkdir()
        (out / "f.bin").write_bytes(b"x")
        stub_dir = tmp_path / f"bin{exit_code}"
        stub_dir.mkdir()
        self._stub_chmod(stub_dir, lines=3, exit_code=exit_code)

        result = subprocess.run(
            ["bash", "-c", f"set -o pipefail\nOUTPUT_PATH={out}\n{block}"],
            capture_output=True,
            text=True,
            check=False,
            env={"PATH": f"{stub_dir}:/usr/bin:/bin", "HOME": str(tmp_path)},
        )
        assert result.returncode == 0, result.stderr

    def test_epilogue_reports_failures_where_the_monitor_can_see_them(self, tmp_path):
        """A refused ``chmod`` must be visible, and capped.

        Only ``command.sh`` is tee'd to ``/logs/output.log`` -- the file the
        sidecar monitor tails -- and this block runs after that pipeline, so the
        diagnostics must go to stdout or they never reach the log an operator
        reads. The listing is capped because a wholly root-squashed tree emits
        one line per file.
        """
        block = "\n".join(self._epilogue())
        out = tmp_path / "out"
        out.mkdir()
        (out / "f.bin").write_bytes(b"x")
        stub_dir = tmp_path / "bin"
        stub_dir.mkdir()
        self._stub_chmod(stub_dir, lines=25, exit_code=1)

        result = subprocess.run(
            ["bash", "-c", f"set -o pipefail\nOUTPUT_PATH={out}\n{block}"],
            capture_output=True,
            text=True,
            check=False,
            env={"PATH": f"{stub_dir}:/usr/bin:/bin", "HOME": str(tmp_path)},
        )
        assert result.returncode == 0, result.stderr
        # On stdout, not stderr: stderr would bypass the monitored log.
        assert "WARNING" in result.stdout, "failure not reported on stdout"
        assert "25 path(s)" in result.stdout, "true failure count not reported"
        assert "and 15 more" in result.stdout, "listing not capped"
        assert result.stdout.count("Operation not permitted") == 10, "cap not applied"

    def test_epilogue_uses_capital_x(self):
        """``g+rwX`` must not become ``g+rwx`` -- that would mark data executable."""
        block = "\n".join(self._epilogue())
        assert "g+rwX" in block, "expected g+rwX"
        assert "g+rwx" not in block, "g+rwx would chmod data +x"

    def test_rendered_epilogue_makes_a_0600_artifact_group_readable(self, tmp_path):
        """Run the real block: the reported failure mode must be fixed.

        Reproduces the reported tree -- a 0600 ``adapter_model.safetensors``
        beside 0644 siblings -- and asserts the epilogue makes it group-readable
        while leaving the group-execute bit off the data files.
        """
        block = "\n".join(self._epilogue())
        out = tmp_path / "custom_output"
        sub = out / "granite-4.1-3b_finance"
        sub.mkdir(parents=True)
        data = sub / "adapter_model.safetensors"
        data.write_bytes(b"x")
        data.chmod(0o600)
        readme = sub / "README.md"
        readme.write_text("x")
        readme.chmod(0o644)
        script = sub / "run.sh"
        script.write_text("x")
        script.chmod(0o744)
        sub.chmod(0o755)

        result = subprocess.run(
            ["bash", "-c", f"set -o pipefail\nOUTPUT_PATH={out}\n{block}"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

        mode = lambda f: f.stat().st_mode & 0o777
        assert mode(data) & 0o040, "data file still not group-readable"
        assert not mode(data) & 0o010, "data file must not become group-executable"
        assert mode(readme) & 0o040, "sibling file not group-readable"
        assert mode(script) & 0o010, "already-executable file should keep group-execute"
        assert mode(sub) & 0o050, "directory must stay group-readable/traversable"

    def test_rendered_epilogue_is_a_noop_without_output_path(self, tmp_path):
        """Built-in steps set no OUTPUT_PATH; the block must exit cleanly anyway.

        ``OUTPUT_PATH`` comes from the gbstep chart's values.yaml, so hfpush and
        the other built-in k8s steps run this block with the variable unset.
        """
        block = "\n".join(self._epilogue())
        for script in (
            f"set -o pipefail\nunset OUTPUT_PATH\n{block}",
            f"set -o pipefail\nOUTPUT_PATH={tmp_path}/missing\n{block}",
        ):
            result = subprocess.run(
                ["bash", "-c", script], capture_output=True, text=True, check=False
            )
            assert result.returncode == 0, result.stderr
            assert "Normalizing" not in result.stdout
