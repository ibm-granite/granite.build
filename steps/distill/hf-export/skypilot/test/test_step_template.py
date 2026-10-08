"""Contract tests for step-template.yaml.

Scope note: the source-delivery half of this template is spliced VERBATIM from
distill-tokenizer-align, and that step's test_source_contract.py asserts every ported
step's copy is byte-identical to it, while its own test_step_template.py asserts the
region is correct. So this file asserts only what is specific to THIS step.
"""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

_HERE = Path(__file__).resolve().parent.parent
_STEP = _HERE / "step-template.yaml"
_EXPORT = _HERE / "src" / "export_hf_model.py"


@pytest.fixture(scope="module")
def step():
    return yaml.safe_load(_STEP.read_text())


@pytest.fixture(scope="module")
def launcher(step):
    return step["environment_configs"]["Skypilot"]["launchers"]["export"]["config"]


@pytest.fixture(scope="module")
def run_script(launcher):
    return launcher["run"]


@pytest.fixture(scope="module")
def export_src():
    return _EXPORT.read_text()


def _as_shell(script):
    """Approximate what fill_objtemplate leaves behind, for a syntax check.

    Block tags become a SPACE, so an ``{% if %}--flag{% else %}--no-flag{% endif %}``
    pair does not collapse into ``--flag--no-flag``.
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
        """``${#VAR}`` opens a Jinja comment — comment_start_string stays the default."""
        assert "${#" not in run_script


class TestLauncher:
    def test_lsf_only(self, step):
        assert step["environment_configs"]["Skypilot"]["subtypes"] == ["lsf"]

    def test_uses_image_id_not_image(self, launcher):
        """``image:`` is not the SkyPilot launcher's key — upstream's template used it,
        which would leave the task with no image at all. ``image_id`` is the one the
        provisioner reads."""
        assert "image_id" in launcher
        assert "image" not in launcher

    def test_runs_in_a_prebuilt_registry_image(self, launcher):
        assert launcher["image_id"].startswith("docker:")
        assert "IMAGE_REF" not in launcher["image_id"]

    def test_no_dockerfile_so_common_mk_treats_it_as_non_image(self):
        assert not (_HERE / "Dockerfile").exists()

    def test_no_accelerator_is_pinned(self, launcher):
        """CPU-only by design: pruning and copying files needs no GPU. Resources come
        from the build.yaml, like every other ported step."""
        assert launcher["resources"] == {}

    def test_the_entrypoint_is_shipped(self, launcher):
        assert launcher["file_mounts"] == {"src": "src"}

    def test_uses_the_shipped_monitor(self, step):
        monitors = step["environment_configs"]["Skypilot"]["monitors"]
        assert monitors["skypilot_monitor"]["ref"] == "space://monitors/skypilot"


class TestArtifactContract:
    def test_the_only_declared_output_is_the_model(self, step):
        assert list(step["outputs"]["required"]) == ["hf_model"]
        assert step["outputs"]["required"]["hf_model"]["type"] == "model"

    def test_declared_and_printed_ids_match(self, step, run_script):
        declared = set(step["outputs"]["required"])
        printed = set(re.findall(r"GB_ARTIFACT_ID:(\w+)", run_script))
        assert declared == printed, f"declared={declared} printed={printed}"

    def test_the_artifact_path_is_absolutised_before_it_is_printed(self, run_script):
        assert 'DEST="$WORK/$DEST"' in run_script
        assert run_script.index('DEST="$WORK/$DEST"') < run_script.index(
            "GB_ARTIFACT_ID:hf_model"
        )

    def test_the_export_dest_is_what_is_published(self, run_script):
        """Not the trainer's output_dir: publishing that would ship the DeepSpeed resume
        state this step exists to prune."""
        assert "GB_ARTIFACT_PATH:${DEST}" in run_script
        assert "train_output_dir" not in run_script.split("GB_ARTIFACT_ID")[1]


class TestFlagSurface:
    """template -> export_hf_model.py drift, asserted in both directions."""

    BOOLEAN_KEYS = ("allow_unknown", "verify")

    def test_every_export_config_key_reaches_the_script(self, step, run_script):
        for key in step["config"]["export_config"]:
            if key in self.BOOLEAN_KEYS:
                continue
            flag = "--" + key.replace("_", "-")
            assert flag in run_script, f"{key} never reaches export_hf_model.py"

    def test_booleans_are_flag_pairs_not_values(self, step, run_script):
        """Jinja renders a YAML boolean with PYTHON casing, so ``--flag {{ x }}`` arrives
        as the literal string ``False``: nothing errors and the setting is silently
        inverted."""
        for key in self.BOOLEAN_KEYS:
            assert key in step["config"]["export_config"]
            flag = "--" + key.replace("_", "-")
            assert f"{{% if config.export_config.{key} %}}{flag}" in run_script
            assert f"--no-{key.replace('_', '-')}" in run_script
            assert f"{flag} {{{{" not in run_script

    def test_verify_is_never_an_omitted_flag(self, run_script):
        """Rendered as an explicit --verify/--no-verify pair so that turning it off reads
        as a decision in the command line a run actually executed."""
        assert "--no-verify" in run_script

    def test_script_defines_every_flag_the_template_passes(
        self, run_script, export_src
    ):
        defined = set(re.findall(r'"(--[a-z][a-z0-9-]*)"', export_src))
        defined |= {f"--no-{f[2:]}" for f in defined}  # BooleanOptionalAction
        body = "\n".join(
            line
            for line in _as_shell(run_script).splitlines()
            if not line.lstrip().startswith("#")
        )
        for flag in set(re.findall(r"(?<![-\w])(--[a-z][a-z0-9-]*)", body)):
            if flag in ("--quiet", "--porcelain", "--all"):  # git's own flags
                continue
            assert flag in defined, f"export_hf_model.py does not define {flag}"


class TestConfigDefaults:
    def test_the_input_is_not_defaulted(self, step):
        assert step["config"]["export_config"]["train_output_dir"] == ""

    def test_padding_side_defaults_to_publish_correct(self, step):
        """The trainer sets 'left' because it GENERATES during on-policy rollout. That is
        right for a trainer and wrong for a published model, where left padding silently
        corrupts batched non-generative use. Normalising it is why this step exists."""
        assert step["config"]["export_config"]["padding_side"] == "right"

    def test_unknown_files_are_refused_by_default(self, step):
        """The keep list is explicit, so a future transformers or TRL release that starts
        writing a new file would otherwise have it silently published in a directory
        users download."""
        assert step["config"]["export_config"]["allow_unknown"] is False

    def test_verify_defaults_on(self, step):
        assert step["config"]["export_config"]["verify"] is True

    def test_the_checkpoint_choice_defaults_to_highest_step(self, step):
        """Empty means highest STEP NUMBER, which is not newest mtime: a resumed run
        rewrites older checkpoints' mtimes, so mtime ordering can pick one that is
        behind."""
        assert step["config"]["export_config"]["checkpoint"] == ""

    def test_the_chat_template_is_copied_verbatim_by_default(self, step):
        """Which generation prompt a published model hands out is a decision about the
        artifact; this step's job is to make it explicit and recorded, not to make it.
        """
        assert step["config"]["export_config"]["chat_template_thinking"] == "keep"

    def test_the_tokenizer_cross_check_is_opt_in(self, step):
        """It needs a second directory to compare against, which only a recipe knows."""
        assert step["config"]["export_config"]["expect_tokenizer_from"] == ""

    def test_step_type_is_a_real_enum_member(self, step):
        from gbcommon.types.stepconfig import StepType

        assert step["type"] in {m.value for m in StepType}


class TestVendoredTemplate:
    def test_the_real_chat_template_is_vendored_for_the_tests(self):
        """The ported unit tests assert against the REAL template on purpose — a stub
        would assert nothing about the property the thinking-policy rewrite preserves.
        """
        tpl = _HERE / "test-data" / "chatml_granite_42_generation.jinja"
        assert tpl.exists()
        assert re.search(
            r"\{%-? *generation *-?%\}", tpl.read_text()
        ), "the vendored template must carry a real {% generation %} block tag"
