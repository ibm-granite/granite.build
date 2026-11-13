"""recipes/distill/: the SLURM environment template and the example site file.

The template is rendered the way scripts/setup-local-space.sh renders it, by
scripts/render_env_template.py run as a subprocess, so these tests exercise the
real gb CLI parameter syntax (dollar-dollar-brace values, <% %> blocks).

The leak guard walks every file under recipes/distill/: nothing site-specific
may be committed there, only in a private site file.
"""

import os
import pathlib
import re
import subprocess
import sys

import jinja2
import pytest
import yaml
from jinja2 import meta

_REPO = pathlib.Path(__file__).resolve().parents[3]
_DISTILL = _REPO / "recipes" / "distill"
_TEMPLATE = _DISTILL / "environments" / "skypilot-slurm.template.yaml"
_SITE = _DISTILL / "site.example.yaml"
_RENDER = _REPO / "scripts" / "render_env_template.py"
# The site-specific environment the template was derived from. It may leave the
# public repo later; the test comparing against it then skips.
_REAL_ENV = (
    _REPO
    / "configurations"
    / "assets"
    / "environments"
    / "skypilot"
    / "slurm"
    / "bluevela"
    / "environment.yaml"
)

# Tokens that identify one site's infrastructure. Case-insensitive.
_LEAK = re.compile(
    r"bluevela|rmf\.ibm|ibm\.com|/proj/|us\.icr\.io|cil15|grp_granite|granitebuild"
    r"|mlx5_|login[0-9]+\.bluevela"
    r"|data-eng|llmb-read-write|g4os|gpu-mid|bv_ssh|1password"
    r"|\bp[0-9]+-r[0-9]+-n[0-9]+\b",
    re.IGNORECASE,
)

_VALUE = re.compile(r"\$\$\{\s*([A-Za-z_][A-Za-z0-9_]*)")
_LOOP_VAR = re.compile(r"<%-?\s*for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\b")
_GBSERVER_TEMPLATE = re.compile(r"\{\{.*?\}\}|\{%.*?%\}")

# Recipe-level keys the distill recipes read from the same site file.
_RECIPE_KEYS = {
    "ENVIRONMENT_URI",
    "WORKDIR_ROOT",
    "TEACHER_MODEL_URI",
    "STUDENT_MODEL_URI",
    "DATASET_URI",
    "DATASET_SPLITS",
    "SMOKE_DATASET_URI",
    "CHAT_TEMPLATE_URI",
    "GPU_ACCELERATOR",
    "GPU_PARTITION",
    "CPU_PARTITION",
    "EVAL_PARTITION",
    "EVAL_ACCELERATOR",
    "EXPORT_PARTITION",
    "EVAL_IMAGE_REGISTRY",
    "NCCL_IB_HCA",
    "SAGE_OUTPUT_DIR",
    "SAGE_RESULTS_DIR",
    "BFCL_OUTPUT_DIR",
    "BFCL_RESULTS_DIR",
    "OE_EVAL_BCB_API_URL",
}


def _site():
    with open(_SITE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _render(site_path, tmp_path):
    out = tmp_path / "rendered.yaml"
    proc = subprocess.run(
        [sys.executable, str(_RENDER), str(_TEMPLATE), str(site_path), str(out)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "GB_ENVIRONMENT": "STANDALONE"},
    )
    assert proc.returncode == 0, proc.stderr
    return out.read_text(encoding="utf-8")


@pytest.fixture(name="rendered_example")
def _rendered_example(tmp_path):
    return _render(_SITE, tmp_path)


def test_example_site_is_a_flat_mapping():
    site = _site()
    assert isinstance(site, dict)
    for key in site:
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*", key), key


def test_template_renders_from_example(rendered_example):
    assert "$${" not in rendered_example
    assert "<%" not in rendered_example
    assert isinstance(yaml.safe_load(rendered_example), dict)


def test_rendered_env_shape(rendered_example):
    site = _site()
    env = yaml.safe_load(rendered_example)
    assert env["name"] == site["ENV_NAME"]
    assert env["type"] == "Skypilot"
    assert env["subtype"] == "slurm"
    cfg = env["config"]
    assert cfg["default_cloud"] == "slurm"
    assert list(cfg["cloud_config"]["slurm"]["cluster_configs"]) == [cfg["cluster"]]
    (block,) = cfg["cluster_ssh_configs"]["slurm"]
    assert block["Host"] == cfg["cluster"]
    assert isinstance(block["HostName"], list)
    assert block["HostName"] == site["LOGIN_HOSTS"]
    assert len(block["HostName"]) == len(site["LOGIN_HOSTS"])
    assert block["IdentityKey"] == site["SSH_KEY_SECRET"]
    mounts = cfg["cloud_config"]["slurm"]["cluster_configs"][cfg["cluster"]][
        "container_mounts"
    ]
    assert len(mounts) == len(site["CONTAINER_MOUNTS"])


def test_gbserver_templates_pass_through(rendered_example):
    template = _TEMPLATE.read_text(encoding="utf-8")
    found = _GBSERVER_TEMPLATE.findall(template)
    for t in found:
        assert t in rendered_example, t
    if _REAL_ENV.is_file():
        source = _REAL_ENV.read_text(encoding="utf-8")
        assert sorted(_GBSERVER_TEMPLATE.findall(source)) == sorted(found)


def test_every_template_key_is_in_example_site():
    template = _TEMPLATE.read_text(encoding="utf-8")
    site = _site()
    loop_vars = set(_LOOP_VAR.findall(template))
    values = set(_VALUE.findall(template)) - loop_vars
    assert values, "the template uses no site values"
    assert values <= set(site), sorted(values - set(site))

    jenv = jinja2.Environment(
        variable_start_string="$${",
        variable_end_string="}",
        block_start_string="<%",
        block_end_string="%>",
    )
    undeclared = meta.find_undeclared_variables(jenv.parse(template))
    assert undeclared <= set(site), sorted(undeclared - set(site))


def test_example_site_has_recipe_keys():
    missing = _RECIPE_KEYS - set(_site())
    assert not missing, sorted(missing)


def test_example_site_uris_are_full_uris():
    for key, value in _site().items():
        if key.endswith("_URI") and key != "OE_EVAL_BCB_API_URL":
            assert re.match(r"[a-z][a-z0-9+]*://", str(value)), (key, value)


def test_no_site_specific_tokens_under_recipes_distill():
    files = [p for p in _DISTILL.rglob("*") if p.is_file()]
    assert _TEMPLATE in files and _SITE in files
    hits = []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            m = _LEAK.search(line)
            if m:
                hits.append(f"{p.relative_to(_REPO)}:{n}: {m.group(0)}")
    assert not hits, "\n".join(hits)


def test_leak_guard_matches_the_source_env():
    """The guard would catch the values this template was derived from."""
    if not _REAL_ENV.is_file():
        pytest.skip("source environment not present")
    with open(_REAL_ENV, encoding="utf-8") as f:
        real = yaml.safe_load(f)
    site = _site_from_real_env(real)
    for key in ("CLUSTER", "SLURM_ACCOUNT", "SHARED_WORKDIR", "SSH_USER"):
        assert _LEAK.search(str(site[key])), key
    for host in site["LOGIN_HOSTS"]:
        assert _LEAK.search(host), host


def _site_from_real_env(env):
    """The site values that reproduce env, read from env itself."""
    cfg = env["config"]
    slurm = cfg["cloud_config"]["slurm"]
    (cluster,) = slurm["cluster_configs"]
    cc = slurm["cluster_configs"][cluster]
    (cpu_partition,) = cc["partition_configs"]
    (ssh,) = cfg["cluster_ssh_configs"]["slurm"]
    cache = cfg["slurm_image_cache"]
    hf_pull = next(
        a for a in env["assetstores"] if a["store_uri"] == "space://assetstores/hf"
    )
    site = {
        "ENV_NAME": "rendered-for-comparison",
        "CLUSTER": cfg["cluster"],
        "GPU_PARTITION": cfg["zone"],
        "CPU_PARTITION": cpu_partition,
        "GPU_CONSTRAINT": cfg["sbatch_options"]["constraint"],
        "SLURM_ACCOUNT": slurm["sbatch_options"]["account"],
        "SLURM_QOS": slurm["sbatch_options"]["qos"],
        "CPU_QOS": cc["partition_configs"][cpu_partition]["sbatch_options"]["qos"],
        # The source env has no eval-lane entry: its evals share the GPU partition.
        "EVAL_PARTITION": cfg["zone"],
        "EVAL_QOS": "",
        "SHARED_WORKDIR": cfg["shared_workdir"],
        "CLUSTER_WORKDIR": cc["workdir"],
        "CLUSTER_TMPDIR": cc["tmpdir"],
        "CONTAINER_MOUNTS": [
            {"container_path": k, "host_path": v["host_path"], "mode": v["mode"]}
            for k, v in cc["container_mounts"].items()
        ],
        "IMAGE_CACHE_SHARE_PATH": cache["share_path"],
        "LOGIN_HOSTS": ssh["HostName"],
        "SSH_USER": ssh["User"],
        "SSH_KEY_SECRET": ssh["IdentityKey"],
        "HF_CACHE_PATH": hf_pull["pull"][0]["config"]["cache_path"],
    }
    # Values the template derives from one key rather than repeating it.
    assert cluster == cfg["cluster"] == ssh["Host"]
    assert cache["partition"] == cpu_partition
    assert cfg["steps"]["hfpull"]["zone"] == cpu_partition
    assert cfg["steps"]["hfpush"]["zone"] == cpu_partition
    assert cache["sbatch_options"] == {
        "account": site["SLURM_ACCOUNT"],
        "qos": site["CPU_QOS"],
    }
    return site


@pytest.mark.skipif(not _REAL_ENV.is_file(), reason="source environment not present")
def test_real_values_reproduce_the_source_env(tmp_path):
    with open(_REAL_ENV, encoding="utf-8") as f:
        real = yaml.safe_load(f)
    site_path = tmp_path / "site.yaml"
    site_path.write_text(yaml.safe_dump(_site_from_real_env(real)), encoding="utf-8")
    rendered = yaml.safe_load(_render(site_path, tmp_path))
    real.pop("name")
    assert rendered.pop("name") == "rendered-for-comparison"
    assert rendered == real


def _partition_configs(site, tmp_path):
    site_path = tmp_path / "site.yaml"
    site_path.write_text(yaml.safe_dump(site), encoding="utf-8")
    cfg = yaml.safe_load(_render(site_path, tmp_path))["config"]
    (cc,) = cfg["cloud_config"]["slurm"]["cluster_configs"].values()
    return cfg, cc["partition_configs"]


def test_eval_qos_gets_its_own_partition_entry(tmp_path):
    site = _site()
    site["EVAL_PARTITION"] = "gpu-preemptible"
    site["EVAL_QOS"] = "burst"
    cfg, parts = _partition_configs(site, tmp_path)
    assert parts["gpu-preemptible"] == {"sbatch_options": {"qos": "burst"}}
    assert parts[site["CPU_PARTITION"]] == {"sbatch_options": {"qos": site["CPU_QOS"]}}
    # The env-wide QOS, which the training targets use, is unchanged.
    assert cfg["cloud_config"]["slurm"]["sbatch_options"]["qos"] == site["SLURM_QOS"]


@pytest.mark.parametrize(
    "partition,qos",
    [("gpu-preemptible", ""), ("GPU", "burst"), ("CPU", "burst")],
    ids=["empty-qos", "eval-on-gpu-partition", "eval-on-cpu-partition"],
)
def test_eval_qos_adds_nothing_when_not_needed(tmp_path, partition, qos):
    site = _site()
    site["EVAL_PARTITION"] = {
        "GPU": site["GPU_PARTITION"],
        "CPU": site["CPU_PARTITION"],
    }.get(partition, partition)
    site["EVAL_QOS"] = qos
    _, parts = _partition_configs(site, tmp_path)
    assert set(parts) == {site["CPU_PARTITION"]}


def test_empty_gpu_constraint_drops_the_directive(tmp_path):
    site = _site()
    site["GPU_CONSTRAINT"] = ""
    site_path = tmp_path / "site.yaml"
    site_path.write_text(yaml.safe_dump(site), encoding="utf-8")
    cfg = yaml.safe_load(_render(site_path, tmp_path))["config"]
    assert "sbatch_options" not in cfg
    assert "launcher_config" not in cfg["steps"]["hfpull"]
    assert cfg["steps"]["hfpull"]["zone"] == site["CPU_PARTITION"]
