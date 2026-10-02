"""Contract tests for step-template.yaml.

Scope note: the source-delivery half of this template is spliced VERBATIM from
distill-tokenizer-align, and
``steps/distill/distill-tokenizer-align/skypilot/test/test_source_contract.py`` asserts that
every ported step's copy is byte-identical to it. That step's own suite asserts the
region is *correct*. So this file does not re-assert those twelve properties — it would
duplicate a guarantee that already has one home, and two homes is how they drift. What
is asserted here is everything specific to THIS step.
"""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

_HERE = Path(__file__).resolve().parent.parent
_STEP = _HERE / "step-template.yaml"
_PREP = _HERE / "src" / "prep_corpus.py"


@pytest.fixture(scope="module")
def step():
    return yaml.safe_load(_STEP.read_text())


@pytest.fixture(scope="module")
def launcher(step):
    return step["environment_configs"]["Skypilot"]["launchers"]["prep"]["config"]


@pytest.fixture(scope="module")
def run_script(launcher):
    return launcher["run"]


@pytest.fixture(scope="module")
def prep_src():
    return _PREP.read_text()


def _as_shell(script):
    """Approximate what fill_objtemplate leaves behind, for a syntax check.

    Block tags become a SPACE rather than nothing, so an ``{% if %}--flag{% else %}
    --no-flag{% endif %}`` pair does not collapse into ``--flag--no-flag``.
    """
    script = re.sub(r"\{%.*?%\}", " ", script, flags=re.S)
    return re.sub(r"\{\{.*?\}\}", "X", script, flags=re.S)


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

    def test_no_login_shell_anywhere(self, run_script):
        assert "bash -lc" not in run_script

    def test_no_jinja_comment_sequence(self, run_script):
        """``${#VAR}`` opens a Jinja comment: the renderer leaves
        ``comment_start_string`` at the default ``{#``."""
        assert "${#" not in run_script


class TestLauncher:
    def test_lsf_only(self, step):
        assert step["environment_configs"]["Skypilot"]["subtypes"] == ["lsf"]

    def test_runs_in_a_prebuilt_registry_image(self, launcher):
        assert launcher["image_id"].startswith("docker:")
        assert "IMAGE_REF" not in launcher["image_id"]

    def test_no_dockerfile_so_common_mk_treats_it_as_non_image(self):
        assert not (_HERE / "Dockerfile").exists()

    def test_no_gpu_is_requested(self, launcher):
        """Rendering and counting tokens is CPU work."""
        assert launcher["resources"] == {}

    def test_the_entrypoints_are_shipped(self, launcher):
        assert launcher["file_mounts"] == {"src": "src"}

    def test_both_entrypoints_are_present(self):
        """merge_shards imports prep_corpus by flat name, so both must ship together."""
        assert (_HERE / "src" / "prep_corpus.py").exists()
        assert (_HERE / "src" / "merge_shards.py").exists()

    def test_uses_the_shipped_monitor(self, step):
        monitors = step["environment_configs"]["Skypilot"]["monitors"]
        assert monitors["skypilot_monitor"]["ref"] == "space://monitors/skypilot"


class TestArtifactContract:
    """The artifact is the train JSONL FILE, not out_dir.

    Two independent reasons, either of which alone settles it: the trainer dispatches on
    the ``.jsonl`` suffix and sends anything else to ``load_dataset()``, which fails on a
    directory; and it locates the manifest as ``corpus_path.parent /
    corpus_manifest.json``, which only resolves if corpus_path is the file.
    """

    def test_the_only_declared_output_is_the_corpus(self, step):
        assert list(step["outputs"]["required"]) == ["corpus"]
        assert step["outputs"]["required"]["corpus"]["type"] == "dataset"

    def test_declared_and_printed_ids_match(self, step, run_script):
        declared = set(step["outputs"]["required"])
        printed = set(re.findall(r"GB_ARTIFACT_ID:(\w+)", run_script))
        assert declared == printed, f"declared={declared} printed={printed}"

    def test_the_artifact_is_the_file_not_the_directory(self, run_script):
        assert "GB_ARTIFACT_PATH:${OUT_DIR}/train.jsonl" in run_script

    def test_the_artifact_path_is_absolutised(self, run_script):
        """A relative env: URI is rejected at config load, and the monitor may hand this
        path to the store from another host."""
        assert 'OUT_DIR="$WORK/$OUT_DIR"' in run_script
        # and the absolutisation must come BEFORE the marker that uses it
        assert run_script.index('OUT_DIR="$WORK/$OUT_DIR"') < run_script.index(
            "GB_ARTIFACT_ID:corpus"
        )

    def test_the_eval_split_is_not_declared(self, step):
        """It exists only when eval_fraction > 0, and a declared-but-absent output is a
        resolver failure. A recipe that needs it reads the manifest's splits.eval.path.
        """
        assert "eval" not in step["outputs"]["required"]
        assert "optional" not in step["outputs"]


class TestFlagSurface:
    """template -> prep_corpus.py drift, asserted in both directions."""

    # Script-only flags: the sharded path is driven by merge_shards, not by a recipe.
    SCRIPT_ONLY = {"--shard-count", "--shard-index"}

    def test_every_corpus_config_key_reaches_the_script(self, step, run_script):
        for key in step["config"]["corpus_config"]:
            if key == "emit_row_id":
                continue  # a --flag/--no-flag pair, asserted below
            flag = "--" + key.replace("_", "-")
            assert flag in run_script, f"{key} never reaches prep_corpus.py"

    def test_booleans_are_flag_pairs_not_values(self, step, run_script):
        """Jinja renders a YAML boolean with PYTHON casing, so ``--flag {{ x }}`` arrives
        as the literal string ``False``: nothing errors and the setting is silently
        inverted. argparse's BooleanOptionalAction supplies the ``--no-`` form."""
        assert "emit_row_id" in step["config"]["corpus_config"]
        assert "{% if config.corpus_config.emit_row_id %}--emit-row-id" in run_script
        assert "--no-emit-row-id" in run_script
        assert "--emit-row-id {{" not in run_script

    def test_script_parses_every_flag_the_template_passes(self, run_script, prep_src):
        """A flag argparse does not define is a hard parse error, not a warning."""
        defined = set(re.findall(r'"(--[a-z][a-z0-9-]*)"', prep_src))
        # BooleanOptionalAction defines the --no- form implicitly.
        defined |= {f"--no-{f[2:]}" for f in defined}
        body = "\n".join(
            line
            for line in _as_shell(run_script).splitlines()
            if not line.lstrip().startswith("#")
        )
        for flag in set(re.findall(r"(?<![-\w])(--[a-z][a-z0-9-]*)", body)):
            if flag in ("--quiet", "--porcelain", "--all"):  # git's own flags
                continue
            assert flag in defined, f"prep_corpus.py does not define {flag}"

    def test_the_script_only_flags_are_not_wired(self, run_script, prep_src):
        """They exist in the script; a recipe passing them would be sharding by accident."""
        for flag in self.SCRIPT_ONLY:
            assert flag in prep_src, f"{flag} is no longer a script flag"
            assert flag not in run_script


class TestConfigDefaults:
    def test_dataset_and_tokenizer_are_not_defaulted(self, step):
        """A wrong default here silently changes which examples survive."""
        corpus = step["config"]["corpus_config"]
        assert corpus["dataset"] == ""
        assert corpus["tokenizer"] == ""

    def test_the_filtering_policies_are_the_measured_defaults(self, step):
        """Each of these changes WHICH examples survive, so each is pinned rather than
        left to the script."""
        corpus = step["config"]["corpus_config"]
        assert corpus["length_policy"] == "drop"
        assert corpus["completion_boundary"] == "last_message"
        assert corpus["documents_policy"] == "drop"
        assert corpus["min_messages"] == 2

    def test_emit_row_id_mirrors_the_script_default(self, step, prep_src):
        """The step's default must not be a policy the script has not validated."""
        assert step["config"]["corpus_config"]["emit_row_id"] is False
        assert re.search(r'"--emit-row-id".*?default=False', prep_src, re.S)

    def test_the_split_is_deterministic_from_seed_alone(self, step):
        assert step["config"]["corpus_config"]["seed"] == 42

    def test_step_type_is_a_real_enum_member(self, step):
        from gbcommon.types.stepconfig import StepType

        assert step["type"] in {m.value for m in StepType}
