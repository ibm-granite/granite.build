# SPDX-License-Identifier: Apache-2.0

"""Experiment tracking for logit-distill and student-sft: one tool per run (#752).

tracking.py (steps/train/distill/src/gb_steps_post_training/distillation) decides whether
the chosen tool is on, which server it talks to, and whether credentials are present. Its
own self-test is the bulk of the coverage, and it runs here so a suite failure enforces it:
the tool choice, the default servers and user-set ones, netrc parity with wandb 0.31.2,
apply(), and the run-continuity pins.

The rest of this file pins the plumbing: the config keys exist on both trainers'
CustomArguments, this step's template passes them to the renderer only when one is set, and
the renderer emits only the ones that are set.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from gb_steps_post_training.distillation import tracking

_HERE = Path(__file__).resolve().parent
_STEP = _HERE.parent / "step-template.yaml"
_RENDER = _HERE.parent / "src" / "render_gold_config.py"
_PKG = Path(tracking.__file__).resolve().parent
_TRACKING_KEYS = (
    "tracking_tool",
    "clearml_project",
    "clearml_run_name",
    "wandb_entity",
    "wandb_project",
    "wandb_run_name",
)


def test_the_tracking_self_test_passes(capsys):
    assert tracking.self_test() == 0, capsys.readouterr().out


def test_the_key_list_is_trackings_own():
    assert tracking.ALL_CONFIG_FIELDS == _TRACKING_KEYS


def _custom_arguments_fields(path):
    tree = ast.parse(path.read_text())
    [cls] = [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "CustomArguments"
    ]
    return {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}


@pytest.mark.parametrize("trainer", ["gold.py", "sft.py"])
def test_every_tracking_key_is_a_trainer_field(trainer):
    """TrlParser rejects an unknown top-level key, so a key the renderer emits must be a field.
    Read from the source, since the trainers need torch and trl to import."""
    missing = set(tracking.ALL_CONFIG_FIELDS) - _custom_arguments_fields(_PKG / trainer)
    assert not missing, f"{trainer} CustomArguments lacks {sorted(missing)}"


def _render_run(tracking_block):
    from jinja2 import Template

    step = yaml.safe_load(_STEP.read_text())
    script = step["environment_configs"]["Skypilot"]["launchers"]["gold"]["config"][
        "run"
    ]
    config = dict(step["config"])
    config["tracking"] = dict(config["tracking"], **tracking_block)
    bindings = {
        "student": {"binding": {"path": "/models/student"}},
        "teacher": {"binding": {"path": "/models/teacher"}},
        "corpus": {"binding": {"path": "/data/corpus/train.jsonl"}},
    }
    return Template(script).render(config=config, bindings=bindings)


class TestTemplate:
    def test_tracking_defaults_to_off(self):
        step = yaml.safe_load(_STEP.read_text())
        assert step["config"]["tracking"] == {
            "tool": "",
            "clearml_project": "",
            "clearml_run_name": "",
            "wandb_entity": "",
            "wandb_project": "",
            "wandb_run_name": "",
        }

    def test_nothing_set_passes_no_tracking_flag(self):
        assert "--tracking-tool" not in _render_run({})
        assert "--wandb-project" not in _render_run({})

    def test_a_chosen_tool_passes_every_tracking_flag(self):
        script = _render_run(
            {"tool": "wandb", "wandb_entity": "ibm", "wandb_project": "kd"}
        )
        for flag, value in (
            ("--tracking-tool", "wandb"),
            ("--wandb-entity", "ibm"),
            ("--wandb-project", "kd"),
            ("--clearml-project", ""),
        ):
            assert f'{flag} "{value}"' in script

    @pytest.mark.parametrize("step", ["logit-distill", "student-sft"])
    @pytest.mark.parametrize(
        "block,needle",
        [
            ({"clearml_project": "d"}, "tracking.tool is not"),
            ({"tool": "none", "wandb_project": "p"}, "tracking.tool is not"),
            ({"tool": "mlflow"}, "must be wandb, clearml or none"),
            (
                {
                    "tool": "wandb",
                    "wandb_entity": "e",
                    "wandb_project": "p",
                    "clearml_project": "d",
                },
                "clearml keys",
            ),
            (
                {"tool": "clearml", "clearml_project": "d", "wandb_entity": "e"},
                "wandb keys",
            ),
            ({"tool": "wandb", "wandb_project": "p"}, "needs tracking.wandb_entity"),
            ({"tool": "clearml"}, "needs tracking.clearml_project"),
        ],
    )
    def test_a_bad_tracking_block_is_refused_when_gbserver_renders_it(
        self, step, block, needle
    ):
        """Breaking in #752: a block with keys and no tool, as every block before it was, fails
        when gbserver renders the run block (strict, with its raise_error), before any launch.
        """
        # gbserver is in the repo venv (the step suite), not in the image (test-in-image).
        fill_template = pytest.importorskip("gbserver.utils.template").fill_template

        template = _HERE.parent.parent.parent / step / "skypilot" / "step-template.yaml"
        doc = yaml.safe_load(template.read_text())
        launchers = doc["environment_configs"]["Skypilot"]["launchers"]
        (launcher,) = launchers.values()
        config = dict(doc["config"])
        config["tracking"] = dict(config["tracking"], **block)
        with pytest.raises(RuntimeError) as err:
            fill_template(
                launcher["config"]["run"],
                {"config": config, "bindings": {}},
                strict=True,
            )
        # fill_template wraps the template's own error; gbserver logs the cause as the reason.
        assert isinstance(err.value.__cause__, ValueError), err.value.__cause__
        assert needle in str(err.value.__cause__), err.value.__cause__

    @pytest.mark.parametrize("step", ["logit-distill", "student-sft"])
    @pytest.mark.parametrize(
        "block",
        [
            {},
            {"tool": "none"},
            {"tool": "wandb", "wandb_entity": "e", "wandb_project": "p"},
            {"tool": "ClearML", "clearml_project": "d"},
        ],
    )
    def test_a_good_tracking_block_passes_the_render_guard(self, step, block):
        from jinja2 import Environment

        template = _HERE.parent.parent.parent / step / "skypilot" / "step-template.yaml"
        doc = yaml.safe_load(template.read_text())
        (launcher,) = doc["environment_configs"]["Skypilot"]["launchers"].values()
        run = launcher["config"]["run"]
        guard = run[: run.index('{% if _tool == "clearml" and not')]
        guard += run[len(guard) :].split("\n", 1)[0]
        config = dict(doc["config"])
        config["tracking"] = dict(config["tracking"], **block)

        def refuse(msg):
            raise AssertionError(msg)

        env = Environment()
        env.globals["raise_error"] = refuse
        env.from_string(guard).render(config=config)

    def test_the_rendered_script_is_still_valid_bash(self):
        script = _render_run({"tool": "clearml", "clearml_project": "d"})
        result = subprocess.run(
            ["bash", "-n"], input=script, capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr


class TestRenderer:
    _REQUIRED = [
        "--model-name-or-path",
        "/m/student",
        "--teacher-model-name-or-path",
        "/m/teacher",
        "--dataset-name",
        "/d/train.jsonl",
    ]

    def _render(self, tmp_path, *extra):
        out = tmp_path / "gold.yaml"
        cmd = [sys.executable, str(_RENDER), "--output", str(out), "--total-nodes", "1"]
        result = subprocess.run(
            cmd + self._REQUIRED + list(extra),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return yaml.safe_load(out.read_text())

    def test_no_tracking_key_is_emitted_when_none_is_set(self, tmp_path):
        cfg = self._render(tmp_path)
        assert not set(_TRACKING_KEYS) & set(cfg)

    def test_set_keys_are_emitted_stripped_and_empty_ones_are_not(self, tmp_path):
        cfg = self._render(
            tmp_path,
            "--tracking-tool",
            " clearml ",
            "--clearml-project",
            "distill",
            "--wandb-entity",
            "  ",
        )
        assert cfg["tracking_tool"] == "clearml"
        assert cfg["clearml_project"] == "distill"
        assert "wandb_entity" not in cfg and "clearml_run_name" not in cfg


# ─── Against the real libraries (the GPU image; skipped where they are not installed) ─────────

_CORP = "https://corp:8008"
_KEYS = 'credentials { access_key: "AK", secret_key: "SK" }'

# (case, files under HOME (or under the config dir), environment, clearml's API server, credentials)
_CLEARML_CASES = [
    ("nothing", {}, {}, "https://api.clear.ml", False),
    (
        "nested api_server and keys",
        {"clearml.conf": f"api {{\n  api_server: {_CORP}\n  {_KEYS}\n}}\n"},
        {},
        _CORP,
        True,
    ),
    (
        "dotted keys",
        {
            "clearml.conf": f'api.api_server: "{_CORP}"\napi.credentials.access_key: "AK"\napi.credentials.secret_key: "SK"\n'
        },
        {},
        _CORP,
        True,
    ),
    (
        "one-line block",
        {"clearml.conf": f'api {{ api_server: "{_CORP}", {_KEYS} }}\n'},
        {},
        _CORP,
        True,
    ),
    (
        "UI layout, quoted names and =",
        {
            "clearml.conf": f'api {{\n  web_server: https://corp:8080\n  api_server: {_CORP}\n  credentials {{\n    "access_key" = "AK"\n    "secret_key" = "SK"\n  }}\n}}\n'
        },
        {},
        _CORP,
        True,
    ),
    (
        "legacy api.host",
        {"clearml.conf": f"api {{\n  host: {_CORP}\n  {_KEYS}\n}}\n"},
        {},
        _CORP,
        True,
    ),
    (
        "commented-out server and keys",
        {
            "clearml.conf": f'api {{\n  # api_server: {_CORP}\n  credentials {{\n    # access_key: "AK"\n    # secret_key: "SK"\n  }}\n}}\n'
        },
        {},
        "https://api.clear.ml",
        False,
    ),
    (
        "empty keys",
        {
            "clearml.conf": 'api {\n  credentials { access_key: "", secret_key: "" }\n}\n'
        },
        {},
        "https://api.clear.ml",
        False,
    ),
    (
        "one key only",
        {"clearml.conf": 'api {\n  credentials { access_key: "AK" }\n}\n'},
        {},
        "https://api.clear.ml",
        False,
    ),
    (
        "an S3 host is not the API server",
        {
            "clearml.conf": f'api {{ {_KEYS} }}\nsdk {{ aws {{ s3 {{ credentials: [ {{ host: "minio:9000", key: "k", secret: "s" }} ] }} }} }}\n'
        },
        {},
        "https://api.clear.ml",
        True,
    ),
    (
        "trains.conf",
        {"trains.conf": f"api {{\n  api_server: {_CORP}\n  {_KEYS}\n}}\n"},
        {},
        _CORP,
        True,
    ),
    (
        "~/.clearml.conf is not read",
        {".clearml.conf": f"api {{\n  api_server: {_CORP}\n  {_KEYS}\n}}\n"},
        {},
        "https://api.clear.ml",
        False,
    ),
    (
        "keys in the environment",
        {},
        {"CLEARML_API_ACCESS_KEY": "a", "CLEARML_API_SECRET_KEY": "b"},
        "https://api.clear.ml",
        True,
    ),
    (
        "TRAINS_ key aliases",
        {},
        {"TRAINS_API_ACCESS_KEY": "a", "TRAINS_API_SECRET_KEY": "b"},
        "https://api.clear.ml",
        True,
    ),
    (
        "CLEARML_AUTH_TOKEN",
        {},
        {"CLEARML_AUTH_TOKEN": "t"},
        "https://api.clear.ml",
        True,
    ),
    (
        "mixed: server from the file, keys from the environment",
        {"clearml.conf": f"api {{\n  api_server: {_CORP}\n}}\n"},
        {"CLEARML_API_ACCESS_KEY": "a", "CLEARML_API_SECRET_KEY": "b"},
        _CORP,
        True,
    ),
    (
        "mixed: CLEARML_API_HOST over the file's server",
        {"clearml.conf": f"api {{\n  api_server: {_CORP}\n  {_KEYS}\n}}\n"},
        {"CLEARML_API_HOST": "https://env-host:8008"},
        "https://env-host:8008",
        True,
    ),
    (
        "CLEARML_CONFIG_FILE replaces the home files",
        {
            "clearml.conf": f"api {{\n  api_server: {_CORP}\n  {_KEYS}\n}}\n",
            "other.conf": 'api { credentials { access_key: "AK" } }\n',
        },
        {"CLEARML_CONFIG_FILE": "{home}/other.conf"},
        "https://api.clear.ml",
        False,
    ),
    (
        "CLEARML_CONFIG_PATH",
        {
            "cfg/default/api.conf": f'api_server: {_CORP}\ncredentials {{ access_key: "AK", secret_key: "SK" }}\n'
        },
        {"CLEARML_CONFIG_PATH": "{home}/cfg"},
        _CORP,
        True,
    ),
]


_RESOLVE_HOST = (
    "from clearml.config import config_obj as c; "
    "from clearml.backend_api.session.session import Session; "
    "print(Session.get_api_server_host(c))"
)


def _clearml_api_host(environ, home):
    """The API server clearml itself resolves, read directly (the test may see URLs; the probe
    may not print them)."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-I", "-c", _RESOLVE_HOST],
        env={**environ, "HOME": str(home)},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().splitlines()[-1]


def _home_with(tmp_path, files):
    home = tmp_path / "home"
    home.mkdir()
    for rel, text in files.items():
        (home / rel).parent.mkdir(parents=True, exist_ok=True)
        (home / rel).write_text(text)
    return home


@pytest.mark.parametrize(
    "name,files,env,api,keyed", _CLEARML_CASES, ids=[c[0] for c in _CLEARML_CASES]
)
def test_clearml_server_and_credentials_follow_clearml(
    tmp_path, name, files, env, api, keyed
):
    """clearml_probe() is clearml's own loader; tracking must report what it resolves (public
    or self-hosted), set no ClearML server of its own (a set CLEARML_API_HOST would override a
    self-hosted one), and call it credentials exactly when clearml can log in."""
    pytest.importorskip("clearml")
    home = _home_with(tmp_path, files)
    environ = {k: v.format(home=home) for k, v in env.items()}
    environ["PATH"] = "/usr/bin:/bin"
    assert _clearml_api_host(environ, home) == api
    probe = tracking.clearml_probe(environ, home)
    assert probe and "error" not in probe, probe
    assert probe["public"] is (api == "https://api.clear.ml"), probe
    plan = tracking.resolve(
        {"tracking_tool": "clearml", "clearml_project": "p"}, environ=environ, home=home
    )
    assert not any(
        k in plan.env
        for k in ("CLEARML_API_HOST", "CLEARML_WEB_HOST", "CLEARML_FILES_HOST")
    )
    assert (plan.enabled == ("clearml",)) is keyed, plan.lines
    where = "public ClearML server" if api == "https://api.clear.ml" else "self-hosted"
    assert any(where in line for line in plan.lines), plan.lines


def test_the_probe_prints_booleans_only(tmp_path):
    """A server URL can carry userinfo, and an error message can quote the config: the probe's
    output is booleans, or an exception's type name, and nothing else."""
    pytest.importorskip("clearml")
    import subprocess

    home = _home_with(
        tmp_path,
        {
            "clearml.conf": 'api.api_server: "https://leakuser:leakpass@corp:8008"\n'
            'api.credentials.access_key: "AKSECRETVALUE"\napi.credentials.secret_key: "SKSECRETVALUE"\n'
        },
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", tracking._CLEARML_PROBE],
        env={"PATH": "/usr/bin:/bin", "HOME": str(home)},
        capture_output=True,
        text=True,
        check=False,
    )
    for secret in ("leakuser", "leakpass", "corp", "AKSECRETVALUE", "SKSECRETVALUE"):
        assert secret not in result.stdout, result.stdout
    probe = tracking.clearml_probe({"PATH": "/usr/bin:/bin"}, home)
    assert set(probe) == {"public", "token", "access", "secret"}, probe
    assert all(isinstance(v, bool) for v in probe.values()), probe

    (home / "clearml.conf").write_text('api { api_server: "https://leakuser@corp" \n')
    probe = tracking.clearml_probe({"PATH": "/usr/bin:/bin"}, home)
    assert set(probe) == {"error"} and "leak" not in probe["error"], probe
    assert probe["error"].isidentifier(), probe


def test_a_chosen_but_missing_clearml_is_fatal_on_the_real_path():
    """Where clearml is not installed there is no probe at all; resolve() must still refuse,
    not report 'off'."""
    import importlib.util

    if importlib.util.find_spec("clearml") is not None:
        pytest.skip("clearml is installed here")
    plan = tracking.resolve(
        {"tracking_tool": "clearml", "clearml_project": "p"},
        environ={"CLEARML_API_ACCESS_KEY": "a", "CLEARML_API_SECRET_KEY": "b"},
        home=Path("/nonexistent"),
    )
    assert not plan.enabled and any(
        "not installed" in f for f in plan.fatal
    ), plan.lines


def test_a_self_hosted_server_keeps_its_server_through_apply(tmp_path):
    """The review's reproduction: a dotted self-hosted config keeps its server after apply()."""
    pytest.importorskip("clearml")
    home = _home_with(
        tmp_path,
        {
            "clearml.conf": f'api.api_server: "{_CORP}"\napi.credentials.access_key: "AK"\napi.credentials.secret_key: "SK"\n'
        },
    )
    environ = {"PATH": "/usr/bin:/bin"}
    plan = tracking.resolve(
        {"tracking_tool": "clearml", "clearml_project": "p"}, environ=environ, home=home
    )

    class Args:
        report_to = []
        run_name = "x"

    env = dict(environ)
    plan.apply(Args(), env, task_init=lambda **kw: type("T", (), {"id": "t1"})())
    assert _clearml_api_host(env, home) == _CORP


_URLS = [
    "https://api.wandb.ai",
    "https://api.wandb.ai/",
    "https://api.forge.coreweave.com",
    "https://forge.coreweave.com/api/wandb",
    "https://forge.coreweave.com/api/wandb/",
    "https://api.forge.coreweave.com:443",
    "http://forge.coreweave.com/api/wandb",
    "https://forge.coreweave.com",
    "https://forge.coreweave.com:443/api/wandb",
    "https://wandb.corp.example:8080",
    "http://wandb.corp.example",
    "https://API.WANDB.AI",
    "https://user@wandb.corp.example",
    "https://wandb.corp.example:99999",
    "https://wandb corp.example",
    "https://api.wandb.ai:443",
    "ftp://wandb.corp.example",
    " https://wandb.corp.example ",
    "https://wandb.corp.example ",
    "https://wandb.corp.example/path",
    "https://api.wandb.ai.evil.com",
    "https://app.wandb.ai",
    "http://api.wandb.ai",
    "api.wandb.ai",
]


@pytest.mark.parametrize("url", _URLS)
def test_the_server_check_matches_wandbs_own(url):
    """Both of this module's validators -- wandb's Settings, used where wandb is installed, and
    the fallback written against 0.31.2 -- give wandb 0.31.2's verdict and normalised URL.
    """
    wandb = pytest.importorskip("wandb")
    if wandb.__version__ != "0.31.2":
        pytest.skip(f"written against wandb 0.31.2, not {wandb.__version__}")
    from wandb.sdk.wandb_settings import Settings

    try:
        want = Settings(base_url=url).base_url
    except Exception:  # noqa: BLE001
        want = None
    for use_wandb in (True, False):
        tracking.USE_WANDB_VALIDATOR = use_wandb
        try:
            got = tracking.wandb_normalize_base_url(url)
        except ValueError:
            got = None
        finally:
            tracking.USE_WANDB_VALIDATOR = True
        assert got == want, (use_wandb, url, got, want)


def test_the_netrc_lookup_matches_wandbs_own(tmp_path):
    """The self-test's parity table, against wandb 0.31.2's own wbnetrc rather than letters."""
    wandb = pytest.importorskip("wandb")
    if wandb.__version__ != "0.31.2":
        pytest.skip(f"written against wandb 0.31.2, not {wandb.__version__}")
    import os

    from wandb.sdk.lib.wbauth import wbnetrc
    from wandb.sdk.lib.wbauth.host_url import HostUrl

    pw = "0123456789abcdef0123456789abcdef01234567"
    netrcs = [
        f"machine api.wandb.ai\n  login user\n  password {pw}\n",
        f"machine api.forge.coreweave.com login u password {pw}\n",
        f"default login u password {pw}\n",
        f'machine "api.wandb.ai" login u password {pw}\n',
        "machine api.forge.coreweave.com login u\nmachine github.com login x password y\n",
        f"machine api.forge.coreweave.com login u\ndefault login u password {pw}\n",
        f"machine API.WANDB.AI login u password {pw}\n",
        f"# a comment\nmachine api.wandb.ai login u password {pw}\n",
        f"machine api.wandb.ai login u password {pw}\nmacdef init\ncd /\n\n",
    ]
    bases = [
        None,
        "https://api.forge.coreweave.com",
        "https://api.forge.coreweave.com:443",
        "https://forge.coreweave.com/api/wandb",
        "https://wandb.corp.example",
    ]
    saved = os.environ.get("NETRC")
    try:
        for text in netrcs:
            (tmp_path / "netrc").write_text(text)
            os.environ["NETRC"] = str(tmp_path / "netrc")
            for base in bases:
                env = {"NETRC": str(tmp_path / "netrc")}
                if base:
                    env["WANDB_BASE_URL"] = base
                theirs = (
                    wbnetrc.read_netrc_auth_with_source(
                        host=HostUrl(base or tracking.WANDB_DEFAULT_BASE_URL)
                    )
                    is not None
                )
                ours = tracking.wandb_credentials(env, tmp_path)[0]
                assert ours == theirs, (text, base, ours, theirs)
    finally:
        if saved is None:
            os.environ.pop("NETRC", None)
        else:
            os.environ["NETRC"] = saved
