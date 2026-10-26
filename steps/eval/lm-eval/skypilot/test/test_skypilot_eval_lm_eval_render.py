# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render tests for the ``eval/lm-eval`` step's step-template.yaml.

Cluster-agnostic, so this sits at the root of the step's ``test/`` dir (Mode 1
only) and is not copied by ``make publish-step``. No network.

The rendered ``setup``/``run`` blocks are executed for real, in a scratch dir
holding the step's own ``src/`` scripts, with stubs for the expensive parts:

* ``lm_eval`` is a stub on ``PATH`` that records its argv (one word per line)
  and writes a ``results_*.json`` like the real CLI;
* ``pip``/``uv`` are stubs that record their argv, so the install phases can be
  asserted without installing anything.

Asking bash rather than regex-matching the rendered text means a quoting slip
shows up here exactly as it would on the node.
"""

import os
import pathlib
import shutil
import subprocess

import pytest
import yaml

jinja2 = pytest.importorskip("jinja2", reason="jinja2 renders the step template")

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_TEMPLATE = _STEP_DIR / "step-template.yaml"
_SRC = _STEP_DIR / "src"

_MODEL = "/staged/model"
_BINDINGS = {"model": {"binding": {"path": _MODEL}}}

# Writes a results file where the real lm_eval would, and records its argv.
_LM_EVAL_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$@" > "$STUB_LOG/lm_eval.argv"
out=""
while [ "$#" -gt 0 ]; do
  [ "$1" = "--output_path" ] && out="$2"
  shift
done
[ -n "${STUB_NO_RESULTS:-}" ] && exit 0
mkdir -p "$out/model"
echo '{"results": {}}' > "$out/model/results_2026-01-01T00-00-00.json"
exit "${STUB_RC:-0}"
"""

# pip and uv: record each call; `uv venv [options] <dir>` makes an activatable
# venv at its last argument.
_RECORDER_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$STUB_LOG/$(basename "$0").calls"
if [ "$(basename "$0")" = uv ] && [ "$1" = venv ]; then
  dir="${@: -1}"
  mkdir -p "$dir/bin" && : > "$dir/bin/activate"
fi
"""


@pytest.fixture(scope="module")
def template() -> dict:
    """The step template, parsed as YAML (its Jinja lives inside string scalars)."""
    return yaml.safe_load(_TEMPLATE.read_text())


@pytest.fixture(scope="module")
def defaults(template) -> dict:
    return dict(template["config"]["lm_eval_config"])


@pytest.fixture(scope="module")
def launcher(template) -> dict:
    return template["environment_configs"]["Skypilot"]["launchers"]["lm_eval"]["config"]


def _render(source: str, cfg: dict, bindings: dict | None = None) -> str:
    """Render one of the launcher's blocks the way gbserver would (strictly)."""
    return jinja2.Template(source, undefined=jinja2.StrictUndefined).render(
        config={"lm_eval_config": cfg},
        bindings=_BINDINGS if bindings is None else bindings,
    )


def _cfg(defaults: dict, **over) -> dict:
    cfg = dict(defaults)
    cfg.update(over)
    return cfg


@pytest.fixture
def node(tmp_path):
    """A scratch workdir with the step's src/ and stubbed lm_eval, pip and uv."""
    if shutil.which("bash") is None:  # pragma: no cover - bash is present in CI
        pytest.skip("bash not available")
    work = tmp_path / "work"
    shutil.copytree(_SRC, work / "src")
    # What setup leaves behind for run (the stub `uv venv` makes the same).
    (work / "venv" / "bin").mkdir(parents=True)
    (work / "venv" / "bin" / "activate").write_text("")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (
        ("lm_eval", _LM_EVAL_STUB),
        ("pip", _RECORDER_STUB),
        ("uv", _RECORDER_STUB),
        # The run script reads the installed version through python; a stub keeps
        # the metadata line deterministic.
        ("python", "#!/usr/bin/env bash\necho 9.9.9\n"),
    ):
        stub = bin_dir / name
        stub.write_text(body)
        stub.chmod(0o755)
    log = tmp_path / "log"
    log.mkdir()

    class Node:
        def __init__(self):
            self.work, self.log = work, log

        def run(self, script: str, **env) -> subprocess.CompletedProcess:
            full_env = dict(
                os.environ,
                PATH=f"{bin_dir}:{os.environ['PATH']}",
                STUB_LOG=str(log),
                **env,
            )
            return subprocess.run(
                ["bash", "-c", script],
                cwd=work,
                env=full_env,
                capture_output=True,
                text=True,
            )

        def argv(self) -> list[str]:
            return (log / "lm_eval.argv").read_text().splitlines()

        def calls(self, tool: str) -> list[str]:
            path = log / f"{tool}.calls"
            return path.read_text().splitlines() if path.exists() else []

    return Node()


def _opt(argv: list[str], flag: str) -> list[str]:
    """The values following ``flag`` up to the next ``--`` option."""
    i = argv.index(flag)
    vals = []
    for word in argv[i + 1 :]:
        if word.startswith("--"):
            break
        vals.append(word)
    return vals


class TestShape:
    def test_name_type_and_contract_key(self, template):
        assert template["name"] == "lm-eval"
        assert template["type"] == "custom"
        assert set(template["config"]) == {"lm_eval_config", "compute_config"}

    def test_no_image_ref_token(self):
        # A public-image step: nothing for `make space` to substitute.
        assert "${IMAGE_REF}" not in _TEMPLATE.read_text()

    def test_does_not_read_llmb_env(self):
        # Only the bash and LSF launchers export $LLMB_*_INPUT_<NAME>.
        assert "LLMB_" not in _TEMPLATE.read_text()

    def test_serves_every_skypilot_endpoint(self, template):
        sky = template["environment_configs"]["Skypilot"]
        assert "subtypes" not in sky
        assert sky["default_launcher"] == "lm_eval"
        assert sky["monitors"]["skypilot_monitor"] == {
            "ref": "space://monitors/skypilot"
        }

    def test_bundles_src(self, launcher):
        assert launcher["file_mounts"] == {"src": "src"}

    def test_declares_model_input_and_results_output(self, template):
        assert template["inputs"] == {
            "allow_unknown": True,
            "required": {"model": {"type": "model", "accept": ["uri", "binding"]}},
        }
        assert template["outputs"] == {"optional": {"results": {"type": "dataset"}}}

    def test_defaults(self, defaults):
        assert defaults["image"] == "python:3.12-slim"
        assert defaults["install"] is True
        assert defaults["lm_eval_version"] == "0.4.13"
        assert defaults["tasks"] == ["aime25"]
        assert defaults["backend"] == "hf"
        assert defaults["torch_index_url"] == "https://download.pytorch.org/whl/cpu"

    def test_guard_knows_every_default_key(self, defaults):
        # The guard refuses unknown keys, so its list must cover the contract.
        guard = (_SRC / "lm_eval_guard.sh").read_text()
        known = " ".join(
            line.split('"')[1]
            for line in guard.splitlines()
            if line.startswith(("KNOWN=", "KNOWN+="))
        ).split()
        assert sorted(known) == sorted(defaults)


class TestImage:
    def _image_id(self, launcher, defaults, **over) -> str:
        return _render(launcher["image_id"], _cfg(defaults, **over))

    def test_default_is_python_slim(self, launcher, defaults):
        assert self._image_id(launcher, defaults) == "docker:python:3.12-slim"

    def test_empty_means_bare_node(self, launcher, defaults):
        assert self._image_id(launcher, defaults, image="") == ""


class TestBashParses:
    @pytest.mark.parametrize("block", ["setup", "run"])
    @pytest.mark.parametrize(
        "over",
        [
            {},
            {"install": False, "image": "ghcr.io/x/lm-eval:1"},
            {"apply_chat_template": True, "log_samples": True, "args": {"x": True}},
            {"model_args": {"k'q": "v"}, "args": {"a b": "c"}, "tasks": ["it's"]},
        ],
    )
    def test_block_is_valid_bash(self, launcher, defaults, block, over):
        rendered = _render(launcher[block], _cfg(defaults, **over))
        assert subprocess.run(["bash", "-n"], input=rendered, text=True).returncode == 0


class TestRunArgv:
    def test_default_argv(self, node, launcher, defaults):
        proc = node.run(_render(launcher["run"], defaults))
        assert proc.returncode == 0, proc.stderr
        argv = node.argv()
        assert argv[0] == "run"  # the 0.4.13 subcommand form
        assert _opt(argv, "--model") == ["hf"]
        assert _opt(argv, "--model_args") == [f"pretrained={_MODEL}"]
        assert _opt(argv, "--tasks") == ["aime25"]
        assert _opt(argv, "--output_path") == [str(node.work / "results")]
        # Nothing lm_eval would default anyway.
        for flag in ("--limit", "--num_fewshot", "--batch_size", "--device", "--seed"):
            assert flag not in argv
        for flag in ("--gen_kwargs", "--apply_chat_template", "--log_samples"):
            assert flag not in argv

    def test_every_key_reaches_lm_eval(self, node, launcher, defaults):
        cfg = _cfg(
            defaults,
            backend="vllm",
            tasks=["aime25", "gsm8k"],
            model_args={"dtype": "bfloat16", "max_model_len": 4096},
            limit=1,
            num_fewshot=0,
            batch_size="auto",
            device="cuda",
            seed=42,
            gen_kwargs={"max_gen_toks": 16, "until": ["\n\n"]},
            apply_chat_template="chatml",
            log_samples=True,
            trust_remote_code=True,
            args={
                "system_instruction": "Be brief.",
                "write_out": True,
                "x": None,
                "y": False,
            },
        )
        proc = node.run(_render(launcher["run"], cfg))
        assert proc.returncode == 0, proc.stderr
        argv = node.argv()
        assert _opt(argv, "--model") == ["vllm"]
        assert _opt(argv, "--model_args") == [
            f"pretrained={_MODEL}",
            "dtype=bfloat16",
            "max_model_len=4096",
        ]
        assert _opt(argv, "--tasks") == ["aime25", "gsm8k"]
        assert _opt(argv, "--limit") == ["1"]
        # 0 is a real value, not "unset".
        assert _opt(argv, "--num_fewshot") == ["0"]
        assert _opt(argv, "--batch_size") == ["auto"]
        assert _opt(argv, "--device") == ["cuda"]
        assert _opt(argv, "--seed") == ["42"]
        # Each entry one word; Jinja renders the list the way ast.literal_eval reads it.
        assert _opt(argv, "--gen_kwargs") == ["max_gen_toks=16", "until=['\\n\\n']"]
        assert _opt(argv, "--apply_chat_template") == ["chatml"]
        assert "--log_samples" in argv and "--trust_remote_code" in argv
        assert _opt(argv, "--system_instruction") == ["Be brief."]
        assert "--write_out" in argv
        assert "--x" not in argv and "--y" not in argv

    def test_bare_apply_chat_template(self, node, launcher, defaults):
        proc = node.run(
            _render(launcher["run"], _cfg(defaults, apply_chat_template=True))
        )
        assert proc.returncode == 0, proc.stderr
        assert _opt(node.argv(), "--apply_chat_template") == []

    def test_single_task_string_is_one_task(self, node, launcher, defaults):
        proc = node.run(_render(launcher["run"], _cfg(defaults, tasks="aime25")))
        assert proc.returncode == 0, proc.stderr
        assert _opt(node.argv(), "--tasks") == ["aime25"]

    @pytest.mark.parametrize(
        "value", ["it's", "a b", "$(touch PWNED)", "`touch PWNED`", "x;touch PWNED"]
    )
    def test_values_are_quoted(self, node, launcher, defaults, value):
        cfg = _cfg(
            defaults, model_args={"revision": value}, args={"system_instruction": value}
        )
        bindings = {"model": {"binding": {"path": f"/m/{value}"}}}
        proc = node.run(_render(launcher["run"], cfg, bindings))
        assert proc.returncode == 0, proc.stderr
        argv = node.argv()
        assert _opt(argv, "--model_args") == [
            f"pretrained=/m/{value}",
            f"revision={value}",
        ]
        assert _opt(argv, "--system_instruction") == [value]
        assert not (node.work / "PWNED").exists()

    def test_bare_node_without_install_uses_path(self, node, launcher, defaults):
        rendered = _render(launcher["run"], _cfg(defaults, install=False))
        assert "./venv" not in rendered
        proc = node.run(rendered)
        assert proc.returncode == 0, proc.stderr


class TestMarker:
    def test_marker_at_line_start_with_absolute_path(self, node, launcher, defaults):
        proc = node.run(_render(launcher["run"], defaults))
        assert proc.returncode == 0, proc.stderr
        lines = proc.stdout.splitlines()
        assert (
            f"GB_ARTIFACT_ID:results GB_ARTIFACT_PATH:{node.work / 'results'}" in lines
        )
        assert (
            "GB_STEP_METADATA_KEY:lm_eval_version GB_STEP_METADATA_VALUE:9.9.9" in lines
        )

    def test_no_marker_when_lm_eval_fails(self, node, launcher, defaults):
        proc = node.run(_render(launcher["run"], defaults), STUB_RC="3")
        assert proc.returncode == 3
        assert "GB_ARTIFACT_ID" not in proc.stdout

    def test_no_marker_without_results_file(self, node, launcher, defaults):
        proc = node.run(_render(launcher["run"], defaults), STUB_NO_RESULTS="1")
        assert proc.returncode != 0
        assert "wrote no results_*.json" in proc.stderr
        assert "GB_ARTIFACT_ID" not in proc.stdout


class TestSetup:
    def test_two_phase_install(self, node, launcher, defaults):
        proc = node.run(_render(launcher["setup"], defaults))
        assert proc.returncode == 0, proc.stderr
        uv = node.calls("uv")
        # On uv's own Python, not the node's (see lm_eval_setup.sh).
        assert uv[0] == "venv --managed-python --python 3.12 ./venv"
        # torch first, alone, from the torch index; then lm-eval from PyPI, with no
        # --extra-index-url anywhere (uv would prefer it).
        assert (
            uv[1]
            == "pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch"
        )
        assert (
            uv[2]
            == "pip install --quiet --index-url https://pypi.org/simple lm-eval[hf]==0.4.13"
        )
        assert not any("extra-index-url" in c for c in uv)
        assert node.calls("pip")[0].endswith("uv")

    def test_caches_stay_in_the_workdir(self):
        setup = (_SRC / "lm_eval_setup.sh").read_text()
        run = (_SRC / "lm_eval_run.sh").read_text()
        assert 'export UV_CACHE_DIR="$PWD/.uv-cache"' in setup
        assert 'export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"' in setup
        assert 'export HF_HOME="$PWD/.hf-cache"' in run
        assert "${GB_SHARED_WORKDIR" not in setup + run

    def test_empty_torch_index_skips_phase_one(self, node, launcher, defaults):
        proc = node.run(_render(launcher["setup"], _cfg(defaults, torch_index_url="")))
        assert proc.returncode == 0, proc.stderr
        assert not any(c.endswith(" torch") for c in node.calls("uv"))

    def test_extras_and_extra_packages(self, node, launcher, defaults):
        cfg = _cfg(defaults, extras=["vllm", "math"], extra_packages=["sympy==1.13.3"])
        proc = node.run(_render(launcher["setup"], cfg))
        assert proc.returncode == 0, proc.stderr
        assert node.calls("uv")[-1].endswith("lm-eval[vllm,math]==0.4.13 sympy==1.13.3")

    def test_no_extras(self, node, launcher, defaults):
        proc = node.run(_render(launcher["setup"], _cfg(defaults, extras=[])))
        assert proc.returncode == 0, proc.stderr
        assert node.calls("uv")[-1].endswith(" lm-eval==0.4.13")

    def test_install_false_skips_install_even_with_image(
        self, node, launcher, defaults
    ):
        proc = node.run(
            _render(launcher["setup"], _cfg(defaults, install=False, image="x/y:1"))
        )
        assert proc.returncode == 0, proc.stderr
        assert node.calls("uv") == [] and node.calls("pip") == []

    def test_image_set_still_installs(self, node, launcher, defaults):
        proc = node.run(
            _render(launcher["setup"], _cfg(defaults, image="ubuntu:24.04"))
        )
        assert proc.returncode == 0, proc.stderr
        assert len(node.calls("uv")) == 3


# Refused in BOTH blocks, before the install (setup) and before lm_eval (run).
_REFUSALS = [
    ({"tasks": []}, "tasks is empty"),
    ({"tasks": [""]}, "empty task name"),
    ({"tasks": ["aime25,gsm8k"]}, "is not one task name"),
    ({"backend": "openai-completions"}, "backend must be 'hf' or 'vllm'"),
    ({"lm_eval_version": ""}, "pin a version"),
    (
        {"model_args": {"pretrained": "gpt2"}},
        "model_args.pretrained is set by the step",
    ),
    ({"model_args": {"bad key": 1}}, "is not a valid key"),
    ({"args": {"output_path": "/x"}}, "args.output_path is set by the step"),
    ({"args": {"tasks": "x"}}, "args.tasks is set by the step"),
    ({"args": {"limit": 1}}, "use lm_eval_config.limit instead"),
    ({"args": {"--limit": 1}}, "is not a flag name"),
    ({"args": {"x'y": 1}}, "is not a flag name"),
    ({"model_path": "/m"}, "the model is the step's required 'model' input"),
    ({"output_path": "/o"}, "results go to the step's 'results' output"),
    ({"taks": ["aime25"]}, "unknown key lm_eval_config.taks"),
]


class TestRefusals:
    @pytest.mark.parametrize("block", ["setup", "run"])
    @pytest.mark.parametrize("over, message", _REFUSALS)
    def test_refused_before_work(self, node, launcher, defaults, block, over, message):
        proc = node.run(_render(launcher[block], _cfg(defaults, **over)))
        assert proc.returncode == 1, proc.stdout
        assert "eval/lm-eval: ERROR" in proc.stderr and message in proc.stderr
        assert not (node.log / "lm_eval.argv").exists()
        assert node.calls("uv") == []
        assert "GB_ARTIFACT_ID" not in proc.stdout

    def test_empty_version_is_fine_without_install(self, node, launcher, defaults):
        cfg = _cfg(defaults, install=False, lm_eval_version="")
        proc = node.run(_render(launcher["run"], cfg))
        assert proc.returncode == 0, proc.stderr

    def test_unbound_model_fails_render(self, launcher, defaults):
        # Validation guarantees the required input is bound; the strict renderer
        # turns a missing one into an error rather than an empty path.
        with pytest.raises(jinja2.UndefinedError):
            _render(launcher["run"], defaults, bindings={})

    def test_extra_ordering_input_is_ignored(self, node, launcher, defaults):
        bindings = dict(_BINDINGS, gate={"binding": {"path": "/elsewhere"}})
        proc = node.run(_render(launcher["run"], defaults, bindings))
        assert proc.returncode == 0, proc.stderr
        assert "/elsewhere" not in " ".join(node.argv())
