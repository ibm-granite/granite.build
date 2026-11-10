"""scripts/setup-local-space.sh, run against a throwaway GB_HOME_DIR.

The script renders a $${KEY}-templated environment from a private site file,
writes it to a standalone space's overlay (or a private repo with --out), and
points the site file's ENVIRONMENT_URI at it.
"""

import os
import pathlib
import stat
import subprocess
import sys

import yaml

_REPO = pathlib.Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "setup-local-space.sh"
_RENDER = _REPO / "scripts" / "render_env_template.py"

_TEMPLATE = """\
name: skypilot-local-$${NAME_SUFFIX}
type: skypilot
config:
  cluster: $${CLUSTER}
  login_hosts:
<% for h in HOSTS %>
    - $${h}
<% endfor %>
  workdir: "{{ binding.path }}"
"""

_SITE = """\
# my site
CLUSTER: c1
NAME_SUFFIX: demo
HOSTS:
  - login1
  - login2
OLD_ENVIRONMENT_URI: keep-me   # contains the key as a substring
ENVIRONMENT_URI: space://environments/old
MODEL_URI: hf://x/y
"""


def _files(tmp_path, template=_TEMPLATE, site=_SITE):
    t = tmp_path / "environment-template.yaml"
    t.write_text(template)
    s = tmp_path / "site.yaml"
    s.write_text(site)
    return t, s


def _run(tmp_path, *args):
    home = tmp_path / "gbhome"
    return subprocess.run(
        ["bash", str(_SCRIPT), *map(str, args)],
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(tmp_path),
            "GB_HOME_DIR": str(home),
            "GB_PYTHON": sys.executable,
            "GB_ENVIRONMENT": "STANDALONE",
        },
        capture_output=True,
        text=True,
        check=False,
    )


def _space(tmp_path):
    return tmp_path / "gbhome" / "local-space"


def _env_file(tmp_path, name="demo"):
    return (
        _space(tmp_path)
        / "overlay/environments/skypilot-local"
        / name
        / "environment.yaml"
    )


def _setup(tmp_path, t, s, *extra, name="demo"):
    return _run(tmp_path, "--template", t, "--site", s, "--name", name, *extra)


def test_renders_values_lists_and_keeps_gbserver_templates(tmp_path):
    t, s = _files(tmp_path)
    proc = _setup(tmp_path, t, s)
    assert proc.returncode == 0, proc.stderr
    text = _env_file(tmp_path).read_text()
    data = yaml.safe_load(text)
    assert data["name"] == "skypilot-local-demo"
    assert data["config"]["cluster"] == "c1"
    assert data["config"]["login_hosts"] == ["login1", "login2"]
    assert "{{ binding.path }}" in text
    assert data["config"]["workdir"] == "{{ binding.path }}"


def test_header_and_modes(tmp_path):
    t, s = _files(tmp_path)
    os.chmod(s, 0o640)
    assert _setup(tmp_path, t, s).returncode == 0
    env_file = _env_file(tmp_path)
    head = env_file.read_text().splitlines()[:5]
    joined = "\n".join(head)
    assert head[0].startswith("#")
    assert str(t) in joined and str(s) in joined
    assert "written by scripts/setup-local-space.sh" in joined
    assert "keep it out of git" in joined
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert stat.S_IMODE((_space(tmp_path) / "space.yaml").stat().st_mode) == 0o600
    # The site file keeps its own mode.
    assert stat.S_IMODE(s.stat().st_mode) == 0o640


def test_missing_key_fails_and_writes_nothing(tmp_path):
    t, s = _files(tmp_path, template=_TEMPLATE + "  extra: $${NOT_IN_SITE}\n")
    proc = _setup(tmp_path, t, s)
    assert proc.returncode != 0
    assert "NOT_IN_SITE" in proc.stderr
    env_dir = _env_file(tmp_path).parent
    assert not env_dir.exists()
    assert not (tmp_path / "gbhome").exists()
    assert s.read_text() == _SITE


def test_invalid_yaml_after_render_fails_and_writes_nothing(tmp_path):
    t, s = _files(tmp_path, template="a: [$${CLUSTER}\nb: c: d\n")
    proc = _setup(tmp_path, t, s)
    assert proc.returncode != 0
    assert "yaml" in proc.stderr.lower()
    env_dir = _env_file(tmp_path).parent
    assert not env_dir.exists()
    assert not (tmp_path / "gbhome").exists()


def test_space_yaml_written_then_kept(tmp_path):
    t, s = _files(tmp_path)
    proc = _setup(tmp_path, t, s)
    assert proc.returncode == 0, proc.stderr
    assert "written" in proc.stdout
    space = yaml.safe_load((_space(tmp_path) / "space.yaml").read_text())
    assert space["name"] == "public"
    assert space["secret_manager"]["type"] == "local"
    assert space["base_uris"] == [
        f"file://{_space(tmp_path)}/overlay",
        f"file://{_REPO}/configurations/assets",
    ]
    before = (_space(tmp_path) / "space.yaml").read_text()
    proc = _setup(tmp_path, t, s)
    assert proc.returncode == 0, proc.stderr
    assert "kept" in proc.stdout
    assert (_space(tmp_path) / "space.yaml").read_text() == before
    assert f"gbserver standalone --space-dir {_space(tmp_path)}" in proc.stdout
    assert f"--parameters-path {s}" in proc.stdout


def test_space_yaml_without_overlay_is_not_changed(tmp_path):
    t, s = _files(tmp_path)
    _space(tmp_path).mkdir(parents=True)
    other = "name: public\nbase_uris:\n  - file:///elsewhere\n"
    (_space(tmp_path) / "space.yaml").write_text(other)
    proc = _setup(tmp_path, t, s)
    assert proc.returncode == 0, proc.stderr
    assert (_space(tmp_path) / "space.yaml").read_text() == other
    assert "NOT changed" in proc.stdout
    assert "warning" in proc.stderr and "overlay" in proc.stderr


def test_space_dir_flag(tmp_path):
    t, s = _files(tmp_path)
    sd = tmp_path / "myspace"
    proc = _setup(tmp_path, t, s, "--space-dir", sd)
    assert proc.returncode == 0, proc.stderr
    assert (sd / "overlay/environments/skypilot-local/demo/environment.yaml").is_file()
    assert (sd / "space.yaml").is_file()
    assert not (tmp_path / "gbhome").exists()


def test_site_env_uri_replaced_in_place(tmp_path):
    t, s = _files(tmp_path)
    assert _setup(tmp_path, t, s).returncode == 0
    got = s.read_text()
    old_lines = _SITE.splitlines(keepends=True)
    new_lines = got.splitlines(keepends=True)
    assert len(new_lines) == len(old_lines)
    for o, n in zip(old_lines, new_lines):
        if o.startswith("ENVIRONMENT_URI:"):
            assert yaml.safe_load(n) == {
                "ENVIRONMENT_URI": "space://environments/skypilot-local/demo"
            }
        else:
            assert o == n
    # Idempotent: a second run doesn't duplicate the line.
    assert _setup(tmp_path, t, s).returncode == 0
    assert s.read_text() == got


def test_site_env_uri_appended_when_absent(tmp_path):
    site = "CLUSTER: c1\nNAME_SUFFIX: demo\nHOSTS: [a]"  # no trailing newline
    t, s = _files(tmp_path, site=site)
    assert _setup(tmp_path, t, s).returncode == 0
    got = s.read_text()
    assert got.startswith(site + "\n")
    assert yaml.safe_load(got)["ENVIRONMENT_URI"] == (
        "space://environments/skypilot-local/demo"
    )
    assert _setup(tmp_path, t, s).returncode == 0
    assert s.read_text() == got


def test_env_key_flag(tmp_path):
    t, s = _files(tmp_path)
    assert _setup(tmp_path, t, s, "--env-key", "OTHER").returncode == 0
    data = yaml.safe_load(s.read_text())
    assert data["OTHER"] == "space://environments/skypilot-local/demo"
    assert data["ENVIRONMENT_URI"] == "space://environments/old"
    assert s.read_text().startswith(_SITE)


def test_out_writes_under_dir_only(tmp_path):
    t, s = _files(tmp_path)
    out = tmp_path / "private-repo"
    proc = _setup(tmp_path, t, s, "--out", out)
    assert proc.returncode == 0, proc.stderr
    env_file = out / "environments/skypilot-local/demo/environment.yaml"
    assert yaml.safe_load(env_file.read_text())["config"]["cluster"] == "c1"
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert not (tmp_path / "gbhome").exists()
    assert "gbserver standalone" not in proc.stdout
    assert f"--parameters-path {s}" in proc.stdout


def test_bad_name_rejected(tmp_path):
    t, s = _files(tmp_path)
    for bad in ["Demo", "-x", "a_b", "a/b", ""]:
        proc = _setup(tmp_path, t, s, name=bad)
        assert proc.returncode != 0, bad
        assert "name" in proc.stderr
    assert not (tmp_path / "gbhome").exists()


def test_missing_required_flags_give_usage(tmp_path):
    t, s = _files(tmp_path)
    for args in (
        ["--site", s, "--name", "demo"],
        ["--template", t, "--name", "demo"],
        ["--template", t, "--site", s],
    ):
        proc = _run(tmp_path, *args)
        assert proc.returncode != 0
        assert "usage" in proc.stderr.lower()


def test_unknown_flag_rejected(tmp_path):
    t, s = _files(tmp_path)
    proc = _setup(tmp_path, t, s, "--bogus")
    assert proc.returncode != 0
    assert "--bogus" in proc.stderr
    assert not (tmp_path / "gbhome").exists()


def _render(tmp_path, template, site=_SITE):
    t, s = _files(tmp_path, template=template, site=site)
    out = tmp_path / "out.yaml"
    proc = subprocess.run(
        [sys.executable, str(_RENDER), str(t), str(s), str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc, out


def test_render_exit_codes(tmp_path):
    proc, out = _render(tmp_path, "a: $${CLUSTER}\n")
    assert proc.returncode == 0, proc.stderr
    assert yaml.safe_load(out.read_text()) == {"a": "c1"}
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    out.unlink()

    proc, out = _render(tmp_path, "a: $${MISSING}\n")
    assert proc.returncode == 2
    assert "MISSING" in proc.stderr
    assert not out.exists()

    proc, out = _render(tmp_path, "a: [$${CLUSTER}\n")
    assert proc.returncode == 3
    assert not out.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "environment-template.yaml",
        "site.yaml",
    ]
