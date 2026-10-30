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

"""Render tests for the `data-designer` step's step-template.yaml.

Cluster-agnostic, so this sits at the root of the step's ``test/`` dir (Mode 1 only)
and is not copied by ``make publish-step``.

The rendered blocks invoke the bundled scripts, so — as in the dpk step's render
tests — the meaningful assertion is "what argv does each script receive?". The
rendered block is run by bash with stub scripts that echo their argv, because bash
is what splits and unquotes these words on the node.
"""

import base64
import collections
import decimal
import enum
import json
import pathlib
import shutil
import subprocess

import dd_run
import pytest
import yaml

jinja2 = pytest.importorskip("jinja2", reason="jinja2 renders the step template")

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_TEMPLATE = _STEP_DIR / "step-template.yaml"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not available"
)


@pytest.fixture(scope="module")
def template() -> dict:
    """The step template, parsed as YAML (its Jinja lives inside string scalars)."""
    return yaml.safe_load(_TEMPLATE.read_text())


@pytest.fixture(scope="module")
def defaults(template) -> dict:
    return dict(template["config"]["data_designer_config"])


@pytest.fixture(scope="module")
def launcher(template) -> dict:
    return template["environment_configs"]["Skypilot"]["launchers"]["data_designer"][
        "config"
    ]


def _env() -> jinja2.Environment:
    """A Jinja env with the gbserver filters the template uses, as gbserver defines them
    (``src/gbserver/utils/template.py``: ``to_yaml`` is ``yaml.dump``)."""
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    env.filters["json_dumps"] = json.dumps
    env.filters["to_yaml"] = lambda v: yaml.dump(v, default_flow_style=False)

    def raise_error(msg):
        raise ValueError(msg)

    env.globals["raise_error"] = raise_error
    env.filters["b64encode"] = lambda s: base64.b64encode(
        s if isinstance(s, bytes) else s.encode()
    ).decode()
    return env


def _render(source: str, cfg: dict, bindings: dict | None = None) -> str:
    return (
        _env()
        .from_string(source)
        .render(config={"data_designer_config": cfg}, bindings=bindings or {})
    )


def _cfg(defaults: dict, **over) -> dict:
    base = dict(
        defaults,
        config={"columns": [{"name": "a", "column_type": "sampler"}]},
        model_providers=[{"name": "rits", "endpoint": "https://x/v1"}],
        output="questions",
    )
    base.update(over)
    return base


def _run_block(rendered: str, tmp_path: pathlib.Path) -> dict[str, list[list[str]]]:
    """Run a rendered block with every bundled script stubbed; return argv per script."""
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    (tmp_path / "venv" / "bin").mkdir(parents=True, exist_ok=True)
    (tmp_path / "venv" / "bin" / "activate").write_text("")
    for name in ("dd_guard.sh", "dd_setup.sh", "dd_ollama.sh"):
        (src / name).write_text(
            '#!/usr/bin/env bash\necho "CALL:%s"\nfor a in "$@"; do echo "ARG:$a"; done\n'
            % name
        )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    # `python ./src/dd_run.py ...` — stub python itself, recording the script name.
    (bin_dir / "python").write_text(
        '#!/usr/bin/env bash\necho "CALL:$(basename "$1")"; shift\n'
        'for a in "$@"; do echo "ARG:$a"; done\n'
    )
    (bin_dir / "python").chmod(0o755)
    proc = subprocess.run(
        ["bash", "-c", rendered],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    calls: dict[str, list[list[str]]] = {}
    current = None
    for line in proc.stdout.splitlines():
        if line.startswith("CALL:"):
            current = []
            calls.setdefault(line[5:], []).append(current)
        elif line.startswith("ARG:") and current is not None:
            current.append(line[4:])
    return calls


def _opts(argv: list[str]) -> dict[str, str]:
    """Pair `--name value` words (every option these scripts take has a value)."""
    out, it = {}, iter(argv)
    for word in it:
        if word == "--":
            out["--"] = " ".join(it)
            break
        out[word] = next(it)
    return out


class TestRenderedShell:
    @pytest.mark.parametrize("block", ["setup", "run"])
    def test_parses_under_bash_n(self, launcher, defaults, block):
        rendered = _render(launcher[block], _cfg(defaults))
        assert subprocess.run(["bash", "-n"], input=rendered, text=True).returncode == 0

    def test_guard_argv_identical_in_both_phases(self, launcher, defaults, tmp_path):
        cfg = _cfg(defaults, min_records=3, export_format="jsonl")
        setup = _run_block(_render(launcher["setup"], cfg), tmp_path)["dd_guard.sh"]
        run = _run_block(_render(launcher["run"], cfg), tmp_path)["dd_guard.sh"]
        assert setup == run

    def test_guard_sees_presence_flags(self, launcher, defaults, tmp_path):
        calls = _run_block(_render(launcher["run"], _cfg(defaults)), tmp_path)
        opts = _opts(calls["dd_guard.sh"][0])
        assert opts["--has-config"] == "x" and opts["--has-providers"] == "x"
        empty = _cfg(defaults, config={}, model_providers=[])
        opts = _opts(
            _run_block(_render(launcher["run"], empty), tmp_path)["dd_guard.sh"][0]
        )
        assert opts["--has-config"] == "" and opts["--has-providers"] == ""


class TestRunnerArgv:
    def test_defaults(self, launcher, defaults, tmp_path):
        opts = _opts(
            _run_block(_render(launcher["run"], _cfg(defaults)), tmp_path)["dd_run.py"][
                0
            ]
        )
        assert opts["--output-path"] == "./output"
        assert opts["--artifact-id"] == "questions"
        assert opts["--num-records"] == "10"
        assert opts["--resume"] == "if_possible"
        assert opts["--min-records"] == ""
        assert opts["--seed-path"] == ""

    def test_quotes_survive(self, launcher, defaults, tmp_path):
        """A quote in author config arrives as data, never as shell."""
        cfg = _cfg(
            defaults, output_path="/shared/it's here", seed_glob="*`touch pwned`*"
        )
        opts = _opts(
            _run_block(_render(launcher["run"], cfg), tmp_path)["dd_run.py"][0]
        )
        assert opts["--output-path"] == "/shared/it's here"
        assert opts["--seed-glob"] == "*`touch pwned`*"
        assert not (tmp_path / "pwned").exists()

    def test_seed_binding_becomes_seed_path(self, launcher, defaults, tmp_path):
        bindings = {"seed": {"binding": {"path": "/staged/seed"}}}
        calls = _run_block(_render(launcher["run"], _cfg(defaults), bindings), tmp_path)
        assert _opts(calls["dd_run.py"][0])["--seed-path"] == "/staged/seed"
        assert _opts(calls["dd_guard.sh"][0])["--seed-path"] == "/staged/seed"

    def test_ordering_only_binding_ignored(self, launcher, defaults, tmp_path):
        bindings = {"gate": {"binding": {"path": "/elsewhere"}}}
        calls = _run_block(_render(launcher["run"], _cfg(defaults), bindings), tmp_path)
        assert _opts(calls["dd_run.py"][0])["--seed-path"] == ""


class TestLocalModel:
    def _on(self, defaults, **over):
        lm = dict(
            defaults["local_model"], runtime="ollama", model="granite4:350m", **over
        )
        return _cfg(defaults, model_providers=[], local_model=lm)

    def test_off_by_default(self, launcher, defaults, tmp_path):
        assert defaults["local_model"]["runtime"] == ""
        setup = _render(launcher["setup"], _cfg(defaults))
        assert "dd_ollama" not in setup and "zstandard" not in setup
        assert "dd_ollama" not in _render(launcher["run"], _cfg(defaults))

    def test_setup_installs_server_and_model(self, launcher, defaults, tmp_path):
        calls = _run_block(_render(launcher["setup"], self._on(defaults)), tmp_path)
        assert (
            _opts(calls["dd_setup.sh"][0])["--"]
            == f"data-designer=={defaults['dd_version']} zstandard==0.25.0"
        )
        assert "dd_ollama.sh install" in _render(launcher["setup"], self._on(defaults))

    @pytest.mark.parametrize("block", ["setup", "run"])
    def test_parses_under_bash_n(self, launcher, defaults, block):
        rendered = _render(launcher[block], self._on(defaults))
        assert subprocess.run(["bash", "-n"], input=rendered, text=True).returncode == 0

    def test_run_wraps_the_runner(self, launcher, defaults):
        run = _render(launcher["run"], self._on(defaults, port=12000))
        wrapper = run.index("bash ./src/dd_ollama.sh run")
        assert wrapper < run.index("python ./src/dd_run.py")
        assert "--port '12000'" in run[wrapper:]

    def test_guard_sees_local_model_in_both_phases(self, launcher, defaults, tmp_path):
        cfg = self._on(defaults)
        setup = _run_block(_render(launcher["setup"], cfg), tmp_path)["dd_guard.sh"]
        run = _run_block(_render(launcher["run"], cfg), tmp_path)["dd_guard.sh"]
        assert setup == run
        opts = _opts(setup[0])
        assert (
            opts["--local-runtime"],
            opts["--local-model"],
            opts["--local-port"],
        ) == ("ollama", "granite4:350m", "11434")

    def test_spec_carries_local_model(self, launcher, defaults):
        spec = yaml.safe_load(
            base64.b64decode(
                _render(launcher["envs"]["GB_DD_SPEC_B64"], self._on(defaults))
            )
        )
        assert spec["local_model"]["model"] == "granite4:350m"


class TestImageMode:
    def test_bare_node_installs_pinned_release(self, launcher, defaults, tmp_path):
        calls = _run_block(_render(launcher["setup"], _cfg(defaults)), tmp_path)
        opts = _opts(calls["dd_setup.sh"][0])
        assert opts["--venv"] == "./venv"
        assert opts["--"] == f"data-designer=={defaults['dd_version']}"
        assert _render(launcher["image_id"], _cfg(defaults)) == ""

    def test_image_skips_install(self, launcher, defaults, tmp_path):
        cfg = _cfg(defaults, dd_image="quay.io/org/dd:1")
        calls = _run_block(_render(launcher["setup"], cfg), tmp_path)
        assert "dd_setup.sh" not in calls
        assert ". ./venv/bin/activate" not in _render(launcher["run"], cfg)
        assert _render(launcher["image_id"], cfg) == "docker:quay.io/org/dd:1"


class TestSpecEnv:
    def test_round_trips_config_and_providers(self, launcher, defaults):
        prompt = "{% raw %}never rendered{% endraw %} {{ topic | upper }}"
        cfg = _cfg(
            defaults,
            config={
                "columns": [{"name": "q", "column_type": "llm-text", "prompt": prompt}]
            },
            model_providers=[
                {
                    "name": "rits",
                    "endpoint": "https://e/v1",
                    "extra_headers_from_env": {"RITS_API_KEY": "RITS_API_KEY"},
                }
            ],
        )
        # The value as it already stands in the step config (the build rendered it
        # once, earlier); this render must carry it byte-for-byte.
        rendered = _render(launcher["envs"]["GB_DD_SPEC_B64"], cfg)
        spec = yaml.safe_load(base64.b64decode(rendered))
        assert spec["config"]["columns"][0]["prompt"] == prompt
        assert spec["model_providers"] == cfg["model_providers"]
        assert spec["run_config"] == {}

    def test_run_config_travels_in_spec(self, launcher, defaults):
        cfg = _cfg(defaults, run_config={"buffer_size": 2})
        spec = yaml.safe_load(
            base64.b64decode(_render(launcher["envs"]["GB_DD_SPEC_B64"], cfg))
        )
        assert spec["run_config"] == {"buffer_size": 2}

    def test_unquoted_dates_render(self, launcher, defaults):
        """``start: 2024-01-01`` unquoted in a build.yaml is a date object. json_dumps
        failed the whole target render on it (review of #543); the YAML spec carries it,
        and dd_run.py turns it back into an ISO string."""
        params = yaml.safe_load("{start: 2024-01-01, end: 2024-12-31 10:00:00}")
        cfg = _cfg(
            defaults,
            config={
                "columns": [
                    {
                        "name": "d",
                        "column_type": "sampler",
                        "sampler_type": "datetime",
                        "params": params,
                    }
                ]
            },
        )
        with pytest.raises(TypeError, match="not JSON serializable"):
            json.dumps(cfg["config"])  # what the old encoding hit
        rendered = _render(launcher["envs"]["GB_DD_SPEC_B64"], cfg)
        spec = dd_run.load_spec({dd_run.SPEC_ENV: rendered})
        assert spec["config"]["columns"][0]["params"] == {
            "start": "2024-01-01",
            "end": "2024-12-31T10:00:00",
        }

    @pytest.mark.parametrize(
        "bad",
        [decimal.Decimal("1.5"), (1, 2), b"ab", object()],
        ids=["decimal", "tuple", "bytes", "object"],
    )
    def test_a_value_yaml_cant_hold_plainly_fails_the_render(
        self, launcher, defaults, bad
    ):
        """Re-review of #543: to_yaml wrote these as !!python/ (or !!binary) tags, which
        failed on the node after cluster setup; now the render itself refuses."""
        cfg = _cfg(defaults, run_config={"x": bad})
        with pytest.raises(
            ValueError, match="run_config.x can't be passed to the node"
        ):
            _render(launcher["envs"]["GB_DD_SPEC_B64"], cfg)

    def test_the_render_error_names_the_path(self, launcher, defaults):
        """The message names where the bad value sits, by key names and list indexes,
        however large the config (an excerpt of the dump could miss it)."""
        columns = [
            {"name": f"c{i}", "column_type": "llm-text", "prompt": "x" * 40}
            for i in range(10)
        ] + [{"name": "d", "params": {"opts": [1, {"w": (1, 2)}]}}]
        cfg = _cfg(defaults, config={"columns": columns})
        with pytest.raises(ValueError) as e:
            _render(launcher["envs"]["GB_DD_SPEC_B64"], cfg)
        assert "data_designer_config.config.columns[10].params.opts[1].w " in str(
            e.value
        )

    def test_subclasses_of_plain_types_fail_the_render(self, launcher, defaults):
        """Second independent review: str/int/list/dict subclasses and Enums pass Jinja's
        ``is string``/``is mapping`` tests but yaml.dump tags them !!python/, which
        failed on the node. The walker checks each value's own dump."""

        class S(str):
            pass

        class E(str, enum.Enum):
            A = "a"

        for bad in (S("x"), E.A, collections.OrderedDict(a=1)):
            cfg = _cfg(defaults, run_config={"k": bad})
            with pytest.raises(ValueError, match="run_config.k can't be passed"):
                _render(launcher["envs"]["GB_DD_SPEC_B64"], cfg)

    def test_plain_scalars_render(self, launcher, defaults):
        rc = {"a": 1e20, "b": -0.0, "c": 10**30, "d": True, "e": False, "f": None}
        rc["g"] = [[1, [2]], []]
        spec = yaml.safe_load(
            base64.b64decode(
                _render(
                    launcher["envs"]["GB_DD_SPEC_B64"], _cfg(defaults, run_config=rc)
                )
            )
        )
        assert spec["run_config"] == rc

    def test_a_key_that_is_not_a_string_fails_the_render(self, launcher, defaults):
        cfg = _cfg(defaults, run_config={(1, 2): "v"})
        with pytest.raises(ValueError, match="a key that is not a string"):
            _render(launcher["envs"]["GB_DD_SPEC_B64"], cfg)

    def test_the_render_error_never_prints_credentials(self, launcher, defaults):
        """Third review of #543: the message printed ~400 characters of the dumped
        spec, and model_providers sorts right before run_config, so literal provider
        credentials landed in the build's error. Also a bad value INSIDE the provider
        that holds them."""
        provider = {
            "name": "gw",
            "endpoint": "https://e/v1",
            "api_key": "sk-LITERAL-SECRET-123",
            "extra_headers": {"Authorization": "Bearer HDR-SECRET-456"},
        }
        for over in (
            {"model_providers": [provider], "run_config": {"buffer_size": (1, 2)}},
            {"model_providers": [dict(provider, extra_body={"x": (1, 2)})]},
        ):
            with pytest.raises(ValueError) as e:
                _render(launcher["envs"]["GB_DD_SPEC_B64"], _cfg(defaults, **over))
            assert "LITERAL-SECRET" not in str(e.value)
            assert "HDR-SECRET" not in str(e.value)

    @pytest.mark.parametrize(
        "prompt",
        [
            "Explain what the !!python/tuple YAML tag means",
            "!!python/tuple",
            "line one\nSECRETKEY: !!python/tuple\nend",
            ("w " * 40) + "PASS: !!python/tuple more",
        ],
        ids=["mention", "whole-value", "multi-line", "wrapped"],
    )
    def test_text_that_looks_like_a_tag_renders(self, launcher, defaults, prompt):
        """Independent review of the #543 fixes: a substring check refused a prompt
        that only mentions a tag, and read "key: !!python/..." out of a wrapped or
        multi-line string. The guard walks the config, not the dumped text."""
        cfg = _cfg(
            defaults,
            config={
                "columns": [{"name": "q", "column_type": "llm-text", "prompt": prompt}]
            },
        )
        spec = yaml.safe_load(
            base64.b64decode(_render(launcher["envs"]["GB_DD_SPEC_B64"], cfg))
        )
        assert spec["config"]["columns"][0]["prompt"] == prompt

    def test_a_to_yaml_error_string_fails_the_render(self, launcher, defaults):
        """gbserver's to_yaml returns "Error: ..." instead of raising. Its text quotes
        the value it got, so the message leaves that text out."""
        env = _env()
        env.filters["to_yaml"] = lambda v: "Error: YAML conversion failed: sk-SECRET"
        with pytest.raises(ValueError) as e:
            env.from_string(launcher["envs"]["GB_DD_SPEC_B64"]).render(
                config={"data_designer_config": _cfg(defaults)}, bindings={}
            )
        assert "sk-SECRET" not in str(e.value)

    def test_gbserver_real_render(self, launcher, defaults):
        """Third review of #543: the tests above use a stand-in Jinja env. This renders
        through gbserver's own fill_template (SandboxedEnvironment, its to_yaml and
        raise_error), which re-raises the guard's error as the RuntimeError's cause."""
        template = pytest.importorskip(
            "gbserver.utils.template", reason="gbserver is not importable here"
        )
        src = launcher["envs"]["GB_DD_SPEC_B64"]
        cfg = _cfg(defaults, model_providers=[])
        rendered = template.fill_template(
            src, {"config": {"data_designer_config": cfg}}, strict=True
        )
        assert yaml.safe_load(base64.b64decode(rendered))["config"] == cfg["config"]
        bad = _cfg(
            defaults,
            model_providers=[{"name": "gw", "api_key": "sk-LITERAL-SECRET-123"}],
            run_config={"buffer_size": (1, 2)},
        )
        with pytest.raises(RuntimeError) as e:
            template.fill_template(
                src, {"config": {"data_designer_config": bad}}, strict=True
            )
        cause = str(e.value.__cause__)
        assert "data_designer_config.run_config.buffer_size can't be passed" in cause
        assert "LITERAL-SECRET" not in cause

    def test_telemetry_is_off(self, launcher):
        assert launcher["envs"]["NEMO_TELEMETRY_ENABLED"] == "false"


def test_step_is_cpu_only_by_default(template):
    assert template["config"]["compute_config"]["num_gpus_per_node"] == 0


def test_declares_no_secrets_by_default(template):
    """Least-privilege: a build declares the secrets its providers read."""
    sky = template["config"]["skypilot"]["secrets"]
    assert sky["secret_names_to_use_as_env_variable"] == []
