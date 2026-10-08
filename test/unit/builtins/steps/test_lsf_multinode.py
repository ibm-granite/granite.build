"""Unit tests for multi-node LSF launch in the gbstep builtin.

Covers the two template files that turn a `compute_config.num_nodes > 1`
request into a real multi-host allocation:

* ``llmb_lsf_jobsub.sh``  -- must add ``span[ptile=1]`` so that LSF's ``-n``
  (a SLOT count, not a host count) actually spreads across hosts, and so that
  ``blaunch`` (one task per allocated slot) spawns one task per node.
* ``llmb_lsf_wrapper.sh`` -- runs once per node under blaunch and must derive
  this node's rank / master address from the LSF host list.

The rendering tests are load-bearing, not cosmetic. Jinja2's comment opener is
a left brace followed by a hash, which is also the start of a bash length
expansion. A length expansion anywhere in these files makes Jinja either abort
or silently swallow everything up to the next hash + right brace -- which is
how the topology function got deleted from the rendered script once already.
``test_no_bash_length_expansion_in_templates`` and
``test_topology_function_survives_rendering`` exist to keep that from
regressing.
"""

import re
import subprocess
from pathlib import Path

import pytest

from gbserver.utils.template import fill_template

REPO_ROOT = Path(__file__).resolve().parents[4]
LSF_SCRIPTS_DIR = (
    REPO_ROOT
    / "src/gbserver/builtins/steps/gbstep/lsf_scripts"
    / "{{ step.name | default(run_metadata.target_name) }}"
)
JOBSUB = LSF_SCRIPTS_DIR / "llmb_lsf_jobsub.sh"
WRAPPER = LSF_SCRIPTS_DIR / "llmb_lsf_wrapper.sh"


def _context(num_nodes: int) -> dict:
    return {
        "config": {
            "compute_config": {
                "num_nodes": num_nodes,
                "num_gpus_per_node": 8,
                "num_cpus_per_node": 64,
                "total_memory_per_node": "512Gi",
            },
            "lsf": {"bsub": {"queue": "preemptable"}},
            "gb": {},
            "workload": {},
        },
        "environment_config": {
            "lsf": {"bsub": {}},
            "authentication": {},
            "workload": {},
        },
        "run_metadata": {
            "build_id": "b1",
            "targetrun_id": "tr1",
            "targetsteprun_id": "tsr1",
            "target_name": "tgt",
            "username": "u1",
        },
        "launcher_config": {},
        "bindings": {},
        "step": {"name": "custom_code_lsf"},
        "setup_config": {"space": {}},
    }


def _render(path: Path, num_nodes: int) -> str:
    return fill_template(path.read_text(), _context(num_nodes), strict=True)


class TestTemplateRenderingSafety:
    @pytest.mark.parametrize("path", [JOBSUB, WRAPPER], ids=["jobsub", "wrapper"])
    def test_no_bash_length_expansion_in_templates(self, path: Path):
        """A bash length expansion collides with Jinja's comment opener."""
        offenders = [
            (i, line)
            for i, line in enumerate(path.read_text().splitlines(), start=1)
            if "${#" in line
        ]
        assert not offenders, (
            f"{path.name} contains bash length expansion(s), which Jinja2 reads "
            f"as a comment opener and will swallow: {offenders}"
        )

    @pytest.mark.parametrize("num_nodes", [1, 4])
    def test_topology_function_survives_rendering(self, num_nodes: int):
        rendered = _render(WRAPPER, num_nodes)
        assert "llmb_compute_topology() {" in rendered
        # The body, not just the signature -- a swallowed block leaves the
        # signature's preceding comment but drops the code.
        assert "LSB_MCPU_HOSTS" in rendered
        assert "export GB_NODE_RANK=" in rendered

    @pytest.mark.parametrize("num_nodes", [1, 4])
    def test_rendered_wrapper_is_valid_bash(self, num_nodes: int, tmp_path: Path):
        script = tmp_path / "wrapper.sh"
        script.write_text(_render(WRAPPER, num_nodes))
        proc = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, proc.stderr


class TestJobsubSpanFlag:
    def test_single_node_omits_span(self):
        """At -n 1 span[ptile=1] is a no-op; omitting it keeps the
        single-node scheduling path identical to the pre-multi-node behaviour."""
        assert "span[ptile=1]" not in _render(JOBSUB, 1)

    @pytest.mark.parametrize("num_nodes", [2, 4, 8])
    def test_multi_node_requests_one_slot_per_host(self, num_nodes: int):
        rendered = _render(JOBSUB, num_nodes)
        assert '-R "span[ptile=1]"' in rendered

    def test_span_flag_precedes_rusage_and_both_survive(self):
        """LSF accepts repeated -R; the rusage request must not be displaced."""
        rendered = _render(JOBSUB, 4)
        span_at = rendered.index('-R "span[ptile=1]"')
        rusage_at = rendered.index('-R "rusage[mem=')
        assert span_at < rusage_at
        assert '-n "${LLMB_LSF_NUM_NODES}"' in rendered
        assert "blaunch" in rendered

    def test_num_nodes_exported_to_job_env(self):
        assert "export LLMB_LSF_NUM_NODES='4'" in _render(JOBSUB, 4)


class TestWrapperRankContract:
    def test_exports_both_gb_and_llmb_prefixes(self):
        """GB_ is current; the LLMB_ aliases are the contract the BYOC step's
        README has advertised since v1 and are kept per the dual-accept policy."""
        rendered = _render(WRAPPER, 4)
        for var in ("NNODES", "NODE_RANK", "MASTER_ADDR", "MASTER_PORT"):
            assert f"export GB_{var}=" in rendered
            assert f"export LLMB_{var}=" in rendered

    def test_master_port_is_not_random(self):
        """Every node computes the port independently, so it must be derived
        from the job id -- $RANDOM would give each node a different port."""
        rendered = _render(WRAPPER, 4)
        assert "LSB_JOBID" in rendered
        port_line = next(
            line for line in rendered.splitlines() if "export GB_MASTER_PORT=" in line
        )
        assert "RANDOM" not in port_line

    def test_artifact_emission_is_gated_on_rank_zero(self):
        rendered = _render(WRAPPER, 4)
        assert 'if [[ "${GB_NODE_RANK:-0}" != "0" ]]; then' in rendered
        # exactly one artifact marker, inside the rank-0 branch
        assert rendered.count("GB_ARTIFACT_ID:") == 1

    def test_failure_reported_by_every_rank_success_only_by_rank_zero(self):
        rendered = _render(WRAPPER, 4)
        failed_at = rendered.index("GB_EVENT_WORKLOAD_STATUS:failed")
        success_at = rendered.index("GB_EVENT_WORKLOAD_STATUS:success")
        gate_at = rendered.index('if [[ "${GB_NODE_RANK:-0}" == "0" ]]; then')
        # the success marker sits inside the rank-0 gate; the failure path does not
        assert gate_at < success_at
        assert failed_at < gate_at

    def test_unknown_host_refuses_instead_of_defaulting_to_rank_zero(self):
        rendered = _render(WRAPPER, 4)
        assert "refusing to run rather than risk a duplicate rank 0" in rendered


class TestTopologyExecution:
    """Execute the rendered topology function against synthetic LSF host lists."""

    @staticmethod
    def _topology_fn(tmp_path: Path) -> Path:
        rendered = _render(WRAPPER, 4)
        match = re.search(
            r"^llmb_compute_topology\(\) \{.*?^\}$", rendered, re.S | re.M
        )
        assert match, "could not extract llmb_compute_topology from rendered wrapper"
        fn = tmp_path / "topo.sh"
        fn.write_text(match.group(0))
        return fn

    def _run(self, tmp_path: Path, hostname: str, env: dict) -> dict:
        fn = self._topology_fn(tmp_path)
        script = f"""
        set -uo pipefail
        source {fn}
        hostname() {{ echo "{hostname}"; }}
        export LLMB_LSF_JOB_NAME=testjob LLMB_LSF_LOG_FILE_COMBINED=/dev/null
        llmb_compute_topology || exit 42
        echo "nnodes=$GB_NNODES rank=$GB_NODE_RANK master=$GB_MASTER_ADDR port=$GB_MASTER_PORT"
        """
        proc = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", **env},
        )
        if proc.returncode == 42:
            return {"refused": True}
        assert proc.returncode == 0, proc.stderr
        return dict(
            kv.split("=", 1) for kv in proc.stdout.strip().split("\n")[-1].split(" ")
        )

    def test_rank_from_mcpu_hosts(self, tmp_path: Path):
        env = {"LSB_MCPU_HOSTS": "hostA 1 hostB 1 hostC 1", "LSB_JOBID": "4242"}
        assert self._run(tmp_path, "hostB", env) == {
            "nnodes": "3",
            "rank": "1",
            "master": "hostA",
            "port": str(29500 + 4242 % 1000),
        }

    @pytest.mark.parametrize(
        "host,rank", [("hostA", "0"), ("hostB", "1"), ("hostC", "2")]
    )
    def test_each_host_gets_a_distinct_rank(self, tmp_path: Path, host, rank):
        env = {"LSB_MCPU_HOSTS": "hostA 1 hostB 1 hostC 1", "LSB_JOBID": "7"}
        assert self._run(tmp_path, host, env)["rank"] == rank

    def test_all_nodes_agree_on_master_port(self, tmp_path: Path):
        env = {"LSB_MCPU_HOSTS": "hostA 1 hostB 1 hostC 1", "LSB_JOBID": "999"}
        ports = {
            self._run(tmp_path, h, env)["port"] for h in ("hostA", "hostB", "hostC")
        }
        assert len(ports) == 1

    def test_fqdn_host_list_matches_short_hostname(self, tmp_path: Path):
        env = {"LSB_MCPU_HOSTS": "hostA.dc.ibm.com 8 hostB.dc.ibm.com 8"}
        result = self._run(tmp_path, "hostB", env)
        assert (result["rank"], result["nnodes"], result["master"]) == (
            "1",
            "2",
            "hostA",
        )

    def test_lsb_hosts_fallback_dedupes_per_slot_repeats(self, tmp_path: Path):
        env = {"LSB_HOSTS": "hostA hostA hostB hostB"}
        result = self._run(tmp_path, "hostB", env)
        assert (result["nnodes"], result["rank"]) == ("2", "1")

    def test_no_lsf_allocation_is_single_node(self, tmp_path: Path):
        result = self._run(tmp_path, "mybox", {})
        assert (result["nnodes"], result["rank"], result["master"]) == (
            "1",
            "0",
            "mybox",
        )

    def test_host_outside_multi_node_allocation_is_refused(self, tmp_path: Path):
        """Defaulting to rank 0 here would give the job two rank 0s, silently
        corrupting the output artifact instead of failing."""
        env = {"LSB_MCPU_HOSTS": "hostA 1 hostB 1"}
        assert self._run(tmp_path, "hostZZZ", env) == {"refused": True}
