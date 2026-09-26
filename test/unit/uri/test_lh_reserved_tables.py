"""Reserved LH table names and non-destructive LH pushes.

Every LH namespace holds model/fileset metadata in physical tables named
``model``, ``model_shared``, ``fileset`` and ``fileset_shared``. A table-type
push to one of them corrupts the whole table, so ``LhURI`` refuses them; this
file checks that each push path hits that guard, and that the lhpush steps
never delete/append/overwrite existing LH content.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import gbcli.services.service_artifact as service_artifact
from gbserver.environment.environment import Environment
from gbserver.types.buildconfig import (
    BuildConfig,
    BuildTargetConfig,
    BuildTargetOutputConfig,
)
from gbserver.utils.template import fill_template

pytestmark = pytest.mark.standalone

RESERVED_TABLE_URI = "lh://prod/mynamespace/tables/model_shared"
LSF_LHPUSH = Path(
    "src/gbserver/builtins/steps/lsf/lhpush/lsf_scripts/lhpush/command.sh"
)
K8S_LHPUSH = Path(
    "src/gbserver/builtins/steps/k8s/lhpush/helm-charts/lhpush/templates/_helpers.tpl"
)
DESTRUCTIVE_DMF = re.compile(r"dmf \w+ delete|dmf table append|--overwrite")


def _build_config_with_outputs(outputs) -> BuildConfig:
    return BuildConfig(
        matched_base_key="granite.build",
        targets={
            "t": BuildTargetConfig(
                environment_uri="space://environments/lsf",
                outputs=outputs,
                steps=[],
            ),
        },
    )


# --- submit-time validation ---------------------------------------------------


def test_literal_reserved_table_output_is_rejected_at_validation():
    cfg = _build_config_with_outputs(
        {"out": BuildTargetOutputConfig(uri=RESERVED_TABLE_URI)}
    )
    errors = [str(e) for e in cfg.my_validate()]
    assert any(
        "Output `out`" in e and "reserved Lakehouse table name" in e for e in errors
    ), errors


def test_templated_and_non_table_lh_outputs_pass_validation():
    cfg = _build_config_with_outputs(
        {
            "tmpl": BuildTargetOutputConfig(
                uri="lh://prod/mynamespace/tables/t_{{ binding.path | short_hash }}"
            ),
            "model": BuildTargetOutputConfig(
                uri="lh://prod/mynamespace/models/model_shared/mymodel/rev1"
            ),
        }
    )
    errors = [str(e) for e in cfg.my_validate()]
    assert not any("Lakehouse" in e for e in errors), errors


# --- push-time guard ------------------------------------------------------------


def test_pushasset_rejects_reserved_table_before_dispatch():
    # URI parsing precedes any store lookup, so a stub self is enough: reaching
    # _get_storeconfig would mean the push was about to be dispatched.
    def _unreachable(**kwargs):
        raise AssertionError("push dispatched for a reserved table")

    stub = SimpleNamespace(_get_storeconfig=_unreachable)
    with pytest.raises(ValueError, match="reserved Lakehouse table name"):
        Environment.pushasset(
            stub,  # type: ignore[arg-type]
            task_group=None,  # type: ignore[arg-type]
            binding={"path": "/data/out.jsonl"},
            uristr=RESERVED_TABLE_URI,
        )


# --- lhpush step templates --------------------------------------------------------


@pytest.mark.parametrize("path", [LSF_LHPUSH, K8S_LHPUSH])
def test_lhpush_steps_have_no_destructive_dmf_ops(path):
    commands = [
        line
        for line in path.read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    assert not [line for line in commands if DESTRUCTIVE_DMF.search(line)]


def _render_lsf_lhpush(lh: dict) -> str:
    return fill_template(
        LSF_LHPUSH.read_text(),
        {
            "config": {
                "lhpush_config": {
                    "path": "/data/out.jsonl",
                    "uri": "lh://prod/mynamespace/x",
                    "use_aspera": False,
                    "lh": {"env": "prod", "namespace": "mynamespace", **lh},
                }
            }
        },
        strict=True,
    )


def test_lsf_lhpush_refuses_reserved_table_from_raw_config():
    # A hand-written lhpush_config never goes through LhURI; the step guards itself.
    rendered = _render_lsf_lhpush({"type": "table", "table_name": "Model"})
    refuse = rendered.index("reserved Lakehouse table")
    assert "exit 1" in rendered[refuse : rendered.index("\n", refuse)]
    assert refuse < rendered.index("dmf table push")


@pytest.mark.parametrize(
    "lh",
    [
        {"type": "table", "table_name": "mytable"},
        {"type": "dataset", "table_name": "model", "dataset_name": "mydataset"},
    ],
)
def test_lsf_lhpush_renders_without_guard_for_allowed_targets(lh):
    rendered = _render_lsf_lhpush(lh)
    assert "reserved Lakehouse table" not in rendered
    assert f"dmf {lh['type']} push" in rendered


# --- CLI table push -------------------------------------------------------------


def test_cli_table_upload_rejects_reserved_table(monkeypatch):
    monkeypatch.setattr(service_artifact, "getLH", lambda token: object())
    monkeypatch.setattr(
        service_artifact,
        "resolve_space",
        lambda *a, **kw: {"lakehouse_namespace": "mynamespace", "name": "myspace"},
    )

    def _unreachable(**kwargs):
        raise AssertionError("uploaded to a reserved table")

    monkeypatch.setattr(service_artifact, "upload_file_lh", _unreachable)
    with pytest.raises(ValueError, match="reserved Lakehouse table name"):
        service_artifact.upload_to_lh(
            github_token="gh",
            lh_token="lh",
            path_name="/data/out.jsonl",
            artifact_name="out",
            type="table",
            label=None,
            size="",
            variant="",
            model_type="",
            version="",
            space="myspace",
            table_name="fileset_shared",
        )
