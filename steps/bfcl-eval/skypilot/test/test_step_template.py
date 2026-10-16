"""Contract tests for step-template.yaml.

Modelled on distill-corpus-prep's. Scope note in the same spirit: this step carries NO
source-delivery block (everything it runs is baked into its own image), so none of the
twelve shared-checkout properties that test_source_contract.py owns apply here. What is
asserted below is specific to this step, plus one thing no other step's suite can assert —
that the replacement for the ICR-hosted legacy asset has not quietly reacquired an
IBM-internal dependency.
"""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

_HERE = Path(__file__).resolve().parent.parent
_STEP = _HERE / "step-template.yaml"
_RUN_BFCL = _HERE / "src" / "run-bfcl.sh"
_MAKEFILE = _HERE / "Makefile"
_DOCKERFILE = _HERE / "Dockerfile"


@pytest.fixture(scope="module")
def step():
    return yaml.safe_load(_STEP.read_text())


@pytest.fixture(scope="module")
def skypilot(step):
    return step["environment_configs"]["Skypilot"]


@pytest.fixture(scope="module")
def launcher(skypilot):
    return skypilot["launchers"]["bfcl"]["config"]


@pytest.fixture(scope="module")
def run_script(launcher):
    return launcher["run"]


@pytest.fixture(scope="module")
def bfcl_src():
    return _RUN_BFCL.read_text()


def _as_shell(script):
    """Approximate what fill_objtemplate leaves behind, for a syntax check.

    Block tags become a SPACE rather than nothing, so an ``{% if %}--flag{% else %}
    --no-flag{% endif %}`` pair does not collapse into ``--flag--no-flag``.
    """
    script = re.sub(r"\{%.*?%\}", " ", script, flags=re.S)
    return re.sub(r"\{\{.*?\}\}", "X", script, flags=re.S)


def _uncommented(text):
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


class TestRunScriptIsValidShell:
    def test_bash_accepts_the_rendered_script(self, run_script):
        result = subprocess.run(
            ["bash", "-n"],
            input=_as_shell(run_script),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    def test_bash_accepts_the_script_with_the_optional_block_absent(self, run_script):
        """The sampling flags are the only conditional region, and the empty rendering is
        the one a recipe gets by default: the line collapses to whitespace plus a line
        continuation, which must still join to the ``| tee`` that follows."""
        stripped = re.sub(
            r"\{%\s*if.*?%\}.*?\{%\s*endif\s*%\}", "", run_script, flags=re.S
        )
        result = subprocess.run(
            ["bash", "-n"],
            input=re.sub(r"\{\{.*?\}\}", "X", stripped, flags=re.S),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    def test_no_login_shell_anywhere(self, run_script):
        assert "bash -lc" not in run_script

    def test_no_jinja_comment_sequence(self, run_script):
        """``${#VAR}`` opens a Jinja comment: the renderer leaves ``comment_start_string``
        at the default ``{#``. This is why the score candidates are counted with ``wc -l``
        over a file instead of a bash array."""
        assert "${#" not in run_script

    def test_pipelines_are_checked(self, run_script):
        """The harness runs behind ``| tee``, so without pipefail a failing run-bfcl.sh
        would be masked by tee's exit 0 and the step would fail later, in the artifact
        selection, with a misleading message."""
        assert "set -o pipefail" in run_script


class TestLauncher:
    def test_not_restricted_to_one_subtype(self, skypilot):
        """Unlike the distill-* steps, which read a checkout from /proj and so are
        LSF-only, everything this step needs is inside its own image."""
        assert "subtypes" not in skypilot

    def test_this_is_an_image_step(self, launcher):
        """The inverse of distill-corpus-prep's assertion: the reference is rendered from
        IMAGE_REF at `make space` / `make publish-step` time, not hardcoded."""
        assert launcher["image_id"] == "docker:${IMAGE_REF}"

    def test_a_dockerfile_exists_so_common_mk_treats_it_as_an_image_step(self):
        """STEP_USES_IMAGE is Dockerfile presence; it is what makes REGISTRY mandatory and
        makes publish-step refuse until the image is on the registry."""
        assert _DOCKERFILE.exists()

    def test_the_gpu_comes_from_the_build_not_the_step(self, launcher):
        """One step serves a sampled smoke run on one GPU and a full corpus on eight."""
        assert launcher["resources"] == {}

    def test_no_file_mounts_because_the_sources_are_baked_in(self, launcher):
        """run-bfcl.sh and its helpers are COPYed to /opt/bfcl-eval by the Dockerfile; a
        file_mounts of src would shadow them with an unsynced copy."""
        assert "file_mounts" not in launcher

    def test_the_script_the_template_invokes_is_the_one_in_the_image(self, run_script):
        assert "bash /opt/bfcl-eval/run-bfcl.sh" in run_script
        assert _RUN_BFCL.exists()

    def test_uses_the_shipped_monitor(self, skypilot):
        assert (
            skypilot["monitors"]["skypilot_monitor"]["ref"]
            == "space://monitors/skypilot"
        )

    def test_the_monitor_honours_the_keys_the_pipeline_recipe_passes(self, skypilot):
        """recipes/granite4-gold-distillation/lsf/distill-pipeline-smoke passes poll_interval_seconds
        and log_retrieval_interval_seconds to this step. A step that does not thread them
        into its monitor config does not error -- it ignores them, which is how a recipe
        ends up believing it set a poll interval it did not set."""
        cfg = skypilot["monitors"]["skypilot_monitor"]["config"]
        assert "config.poll_interval_seconds" in cfg["poll_interval_seconds"]
        assert (
            "config.log_retrieval_interval_seconds"
            in cfg["log_retrieval"]["interval_seconds"]
        )

    def test_progress_regex_matches_lines_the_script_actually_prints(
        self, skypilot, bfcl_src
    ):
        """The legacy asset's regex ("Starting BFCL|Running.*stage|Evaluation complete")
        matched nothing this script emits, so its periodic log retrieval produced no status
        events at all. Every alternative here is checked against the real source."""
        events = skypilot["monitors"]["skypilot_monitor"]["config"][
            "extra_event_configs"
        ]
        (regex,) = [e["line_regex"] for e in events]
        alternatives = re.search(r"run-bfcl: \((.*?)\)", regex).group(1).split("|")
        assert len(alternatives) >= 4
        for alt in alternatives:
            assert re.search(
                r"run-bfcl: " + re.escape(alt), bfcl_src
            ), f"the monitor watches for 'run-bfcl: {alt}' but run-bfcl.sh never prints it"


class TestArtifactContract:
    """The artifact is the score FILE, chosen explicitly, not the output tree."""

    def test_the_only_declared_output_is_the_score_file(self, step):
        assert list(step["outputs"]["required"]) == ["bfcl_results"]
        assert step["outputs"]["required"]["bfcl_results"]["type"] == "dataset"

    def test_the_output_is_required_not_optional(self, step):
        """The legacy asset declared it optional. This step exits 1 when the harness
        leaves no score file, so a successful run always has one -- and `optional` on an
        output that always exists only hides a real failure as a green target."""
        assert "optional" not in step["outputs"]

    def test_declared_and_printed_ids_match(self, step, run_script):
        declared = set(step["outputs"]["required"])
        printed = set(re.findall(r"GB_ARTIFACT_ID:(\w+)", run_script))
        assert declared == printed, f"declared={declared} printed={printed}"

    def test_the_marker_uses_the_house_prefix(self, run_script):
        """Ours emit LLMB_; the monitor accepts either, but GB_ is the form on this
        branch."""
        assert "GB_ARTIFACT_ID:bfcl_results" in run_script
        assert "LLMB_ARTIFACT_ID" not in run_script

    def test_the_artifact_is_a_file_the_step_selected(self, run_script):
        assert "GB_ARTIFACT_PATH:${RESULT_FILE}" in run_script
        assert 'RESULT_FILE=$(head -n 1 "$CANDIDATES_FILE")' in run_script

    def test_the_output_dir_is_absolutised_before_the_marker(self, run_script):
        """A relative env: URI is rejected at config load, and the monitor may hand this
        path to the store from another host."""
        absolutise = 'case "$OUTPUT_DIR" in /*) ;; *) OUTPUT_DIR="${GB_BUILD_WORKDIR:-$PWD}/$OUTPUT_DIR" ;; esac'
        assert absolutise in run_script
        assert run_script.index(absolutise) < run_script.index(
            "GB_ARTIFACT_ID:bfcl_results"
        )

    def test_a_missing_score_file_fails_the_step_loudly(self, run_script):
        """Rather than registering nothing and completing green."""
        assert "exit 1" in run_script
        assert run_script.index("exit 1") < run_script.index(
            "GB_ARTIFACT_ID:bfcl_results"
        )


class TestDistributedEnvIsCleared:
    def test_rank_vars_are_unset_before_the_harness_starts(self, run_script):
        """vLLM refuses to start inside an initialised torch.distributed environment, and
        a recipe reaches this target directly after training targets that export these.
        """
        for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            assert re.search(rf"^\s*unset .*\b{var}\b", run_script, re.M)
        assert run_script.index("unset RANK") < run_script.index("run-bfcl.sh")


class TestFlagSurface:
    """template -> run-bfcl.sh drift, asserted in both directions."""

    # Script flags a recipe must NOT reach. Sharding partitions one corpus across
    # concurrent jobs and forces skip_evaluate, so a target passing --num-shards would
    # score nothing; --exclude-categories is only meaningful alongside it;
    # --evaluate-only re-scores an existing tree, which is a human operation on a
    # finished run, not a pipeline target; and --vllm-port is left at its default because
    # two targets sharing a node would otherwise silently collide on it.
    SCRIPT_ONLY = {
        "--num-shards",
        "--shard-index",
        "--exclude-categories",
        "--skip-evaluate",
        "--evaluate-only",
        "--vllm-port",
    }
    # Config keys that are deliberately not flags.
    NOT_FLAGS = {
        "experiment": "a path component of output_dir",
        "eval_name": "a path component of output_dir",
        "hf_home": "exported as an environment variable, not passed to the script",
    }
    # Flags belonging to other programs invoked in the run block.
    FOREIGN = {"--query-gpu", "--format"}

    def test_every_bfcl_config_key_reaches_the_script(self, step, run_script):
        """Either as a flag or as a template reference -- a key that reaches neither is a
        knob a recipe can set with no effect whatsoever."""
        for key in step["config"]["bfcl_config"]:
            if key in self.NOT_FLAGS:
                assert (
                    f"config.bfcl_config.{key}" in run_script
                ), f"{key} is documented as {self.NOT_FLAGS[key]} but is never referenced"
                continue
            flag = "--" + key.replace("_", "-")
            assert flag in run_script, f"{key} never reaches run-bfcl.sh"

    def test_script_parses_every_flag_the_template_passes(self, run_script, bfcl_src):
        """A flag run-bfcl.sh does not define is a hard error: its argument loop exits 1
        on anything unrecognised."""
        defined = set(re.findall(r"^\s*(--[a-z][a-z0-9-]*)\)", bfcl_src, re.M))
        body = _uncommented(_as_shell(run_script))
        for flag in set(re.findall(r"(?<![-\w])(--[a-z][a-z0-9-]*)", body)):
            if flag in self.FOREIGN:
                continue
            assert flag in defined, f"run-bfcl.sh does not define {flag}"

    def test_the_script_only_flags_are_not_wired(self, run_script, bfcl_src):
        for flag in self.SCRIPT_ONLY:
            assert flag in bfcl_src, f"{flag} is no longer a script flag"
            assert flag not in run_script, f"{flag} must not be settable from a recipe"

    def test_the_sampling_flags_travel_together(self, run_script):
        """sample_seed alone determines WHICH ids, so passing a fraction without the seed
        would make two runs of the same fraction incomparable. One conditional, both
        flags."""
        block = re.search(
            r"\{%\s*if config\.bfcl_config\.sample_fraction\s*%\}(.*?)\{%",
            run_script,
            re.S,
        )
        assert block, "sample_fraction is not conditional"
        assert "--sample-fraction" in block.group(1)
        assert "--sample-seed" in block.group(1)


class TestConfigDefaults:
    def test_the_model_is_not_defaulted(self, step):
        """model_path and model_id both change WHAT is measured -- model_id selects the
        handler that parses tool calls out of raw output, so a wrong value scores the right
        model with the wrong parser and reports a plausible number. run-bfcl.sh refuses an
        empty value for either, so "" is a hard stop rather than a silent fallback."""
        cfg = step["config"]["bfcl_config"]
        assert cfg["model_path"] == ""
        assert cfg["model_id"] == ""

    def test_the_full_corpus_is_the_default(self, step):
        """Sampling is a cost knob, and a default that quietly scored a fraction of the
        corpus would report a number nobody asked for."""
        cfg = step["config"]["bfcl_config"]
        assert cfg["sample_fraction"] == ""
        assert cfg["test_categories"] == "all"

    def test_sampling_is_reproducible_from_the_seed_alone(self, step):
        assert step["config"]["bfcl_config"]["sample_seed"] == "42"

    def test_gpu_memory_utilization_is_a_string(self, step):
        """As the legacy asset had it; vLLM takes a float and the value round-trips
        through Jinja as text either way."""
        assert isinstance(step["config"]["bfcl_config"]["gpu_memory_utilization"], str)

    def test_hf_home_is_not_in_the_typed_workload_section(self, step):
        """config.workload is parsed as StepConfigWorkloadSection, whose fields are
        path/args/workspace_dir/output_dir/python_env. Pydantic IGNORES an unknown key
        there rather than rejecting it, so hf_home placed in that section would vanish
        without a word -- exactly what the legacy asset's `workload.cwd` does."""
        assert "workload" not in step["config"]
        assert "hf_home" in step["config"]["bfcl_config"]

    def test_step_type_is_a_real_enum_member(self, step):
        from gbcommon.types.stepconfig import StepType

        assert step["type"] in {m.value for m in StepType}


class TestNoInternalDependency:
    """The point of this step: the legacy asset it replaces runs
    docker:us.icr.io/cil15-shared-registry/bfcl-py311:0.02, which is not pullable outside
    IBM. Both files name that image in a COMMENT, to say what is being replaced; neither
    may name an internal host on a line that does anything.
    """

    INTERNAL = ("icr.io", "github.ibm.com", "artifactory")

    @pytest.mark.parametrize("path", [_STEP, _MAKEFILE, _DOCKERFILE])
    def test_no_internal_host_on_an_effective_line(self, path):
        for i, line in enumerate(_uncommented(path.read_text()).splitlines(), 1):
            for host in self.INTERNAL:
                assert (
                    host not in line
                ), f"{path.name}:{i} references {host}: {line.strip()}"

    def test_the_registry_is_a_placeholder_not_a_real_one(self):
        """common.mk makes REGISTRY mandatory for image steps with no default. The value
        here must be an obvious placeholder -- an internal registry left in a committed
        Makefile is how `make image` starts pushing somewhere nobody else can pull."""
        assert re.search(
            r"^REGISTRY \?= quay\.io/your-org$", _MAKEFILE.read_text(), re.M
        )
