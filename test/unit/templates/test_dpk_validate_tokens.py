#!/usr/bin/env python3

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

"""Unit tests for the DPK_Tokenize_Skypilot template's output validator.

The validator (``configurations/assets/templates/DPK_Tokenize_Skypilot/scripts/
validate_tokens.py``) is shipped to a SkyPilot worker via ``file_mounts`` and is
not part of the gbserver package, so it is loaded here by path via importlib —
mirroring ``test/unit/recipes/test_rl_checkpoint_eval_generate.py``.

These tests build synthetic ``tokenization2arrow`` output trees, so they cover
the corruption cases that matter without needing DPK, a tokenizer download, or a
cluster: the point of the validator is that it *fails* when the Arrow token
stream and its ``meta/`` sidecars disagree, and a validator that only ever passes
is worthless. Each corruption below is a way a truncated, duplicated, or
mis-ordered write would show up in a real run.

The template's build.yaml is also parsed here to keep the YAML, its parameter
substitution, and the generated shell commands from silently breaking — the
folded-scalar (``>-``) style used for ``command`` is easy to get wrong.
"""

import importlib.util
import json
import pathlib

import pytest
import yaml

pa = pytest.importorskip(
    "pyarrow", reason="pyarrow is required to write Arrow fixtures"
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_TEMPLATE_DIR = (
    _REPO_ROOT / "configurations" / "assets" / "templates" / "DPK_Tokenize_Skypilot"
)
_SCRIPT_PATH = _TEMPLATE_DIR / "scripts" / "validate_tokens.py"


def _load_validator():
    """Import the standalone validator script by path."""
    spec = importlib.util.spec_from_file_location("validate_tokens", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


# --------------------------------------------------------------------------- #
# Fixture builders — synthesize a tokenization2arrow output tree
# --------------------------------------------------------------------------- #


def _write_arrow(path: pathlib.Path, token_count: int) -> None:
    """Write an .arrow file holding ``token_count`` uint32 tokens."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"tokens": pa.array(range(token_count), type=pa.uint32())})
    with pa.OSFile(str(path), "wb") as sink:
        writer = pa.ipc.new_file(sink, table.schema)
        writer.write_table(table)
        writer.close()


def _write_sidecars(
    root: pathlib.Path,
    rel_stem: str,
    docs: list[tuple[str, int]],
    *,
    summary_docs: int | None = None,
    summary_tokens: int | None = None,
) -> None:
    """Write the meta/<stem>.docs and meta/<stem>.docs.ids sidecars."""
    meta = root / "meta" / rel_stem
    meta.parent.mkdir(parents=True, exist_ok=True)
    total = sum(c for _, c in docs)
    n_docs = summary_docs if summary_docs is not None else len(docs)
    n_tokens = summary_tokens if summary_tokens is not None else total
    meta.with_name(meta.name + ".docs").write_text(
        f"{pathlib.Path(rel_stem).name}.parquet, documents: {n_docs}, tokens: {n_tokens}\n"
    )
    meta.with_name(meta.name + ".docs.ids").write_text(
        "".join(f"{doc_id}, {count}\n" for doc_id, count in docs)
    )


def _build_tree(
    root: pathlib.Path,
    files: dict[str, list[tuple[str, int]]],
    *,
    num_tokens: int | None = None,
) -> pathlib.Path:
    """Build a well-formed output tree.

    :param files: maps a relative stem (no ``.arrow``) to its document list of
        ``(document_id, token_count)`` pairs.
    """
    root.mkdir(parents=True, exist_ok=True)
    total = 0
    for stem, docs in files.items():
        count = sum(c for _, c in docs)
        _write_arrow(root / f"{stem}.arrow", count)
        _write_sidecars(root, stem, docs)
        total += count
    stats = {
        "result_files": len(files),
        "num_tokens": total if num_tokens is None else num_tokens,
    }
    (root / "metadata.json").write_text(json.dumps({"job_output_stats": stats}))
    return root


@pytest.fixture
def good_tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """A valid two-file output tree, one with a multi-suffix stem."""
    return _build_tree(
        tmp_path / "out",
        {
            "lang=en/pq01": [("d01", 12), ("d02", 16), ("d03", 17)],
            # multi-suffix stem: sidecars are pq03.snappy.docs, NOT pq03.docs
            "lang=en/dataset=cyber/pq03.snappy": [("d10", 12)],
        },
    )


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


class TestValidatorAcceptsGoodOutput:
    def test_valid_tree_has_no_errors(self, good_tree):
        summary, errors = validator.validate(good_tree)
        assert errors == []
        assert summary["arrow_files"] == 2
        assert summary["total_documents"] == 4
        assert summary["total_tokens"] == 57

    def test_multi_suffix_stem_resolves_its_sidecars(self, good_tree):
        """pq03.snappy.arrow pairs with pq03.snappy.docs, not pq03.docs.

        Regression guard: ``Path.with_suffix('')`` would strip ``.snappy`` too and
        report both sidecars missing.
        """
        _, errors = validator.validate(good_tree)
        assert not [e for e in errors if "missing sidecar" in e]

    def test_main_exits_zero_and_writes_report(self, good_tree, tmp_path, capsys):
        report_dir = tmp_path / "report"
        rc = validator.main(["validate_tokens.py", str(good_tree), str(report_dir)])
        assert rc == 0
        assert "VALIDATION PASSED" in capsys.readouterr().out

        written = json.loads((report_dir / "validation.json").read_text())
        assert written["errors"] == []
        assert written["total_tokens"] == 57


# --------------------------------------------------------------------------- #
# Corruption cases — each must be caught
# --------------------------------------------------------------------------- #


class TestValidatorRejectsCorruptOutput:
    def test_truncated_token_stream(self, good_tree):
        """Arrow holds fewer tokens than the sidecars claim."""
        _write_arrow(good_tree / "lang=en/pq01.arrow", 10)  # was 45
        _, errors = validator.validate(good_tree)
        assert any("token count mismatch" in e for e in errors)

    def test_missing_docs_ids_sidecar(self, good_tree):
        (good_tree / "meta/lang=en/pq01.docs.ids").unlink()
        _, errors = validator.validate(good_tree)
        assert any("missing sidecar" in e and "docs.ids" in e for e in errors)

    def test_missing_docs_summary_sidecar(self, good_tree):
        (good_tree / "meta/lang=en/pq01.docs").unlink()
        _, errors = validator.validate(good_tree)
        assert any(
            "missing sidecar" in e and e.rstrip().endswith(".docs") for e in errors
        )

    def test_duplicate_document_ids(self, good_tree):
        _write_sidecars(good_tree, "lang=en/pq01", [("d01", 20), ("d01", 25)])
        _, errors = validator.validate(good_tree)
        assert any("duplicate document ids" in e for e in errors)

    def test_docs_summary_token_disagreement(self, good_tree):
        """The .docs summary total contradicts the arrow row count."""
        _write_sidecars(
            good_tree,
            "lang=en/pq01",
            [("d01", 12), ("d02", 16), ("d03", 17)],
            summary_tokens=999,
        )
        _, errors = validator.validate(good_tree)
        assert any(".docs summary claims 999 tokens" in e for e in errors)

    def test_docs_summary_document_count_disagreement(self, good_tree):
        _write_sidecars(
            good_tree,
            "lang=en/pq01",
            [("d01", 12), ("d02", 16), ("d03", 17)],
            summary_docs=99,
        )
        _, errors = validator.validate(good_tree)
        assert any(".docs summary claims 99 documents" in e for e in errors)

    def test_corrupt_arrow_file(self, good_tree):
        (good_tree / "lang=en/pq01.arrow").write_text("this is not arrow data")
        _, errors = validator.validate(good_tree)
        assert any("unreadable arrow file" in e for e in errors)

    def test_empty_arrow_file(self, good_tree):
        _write_arrow(good_tree / "lang=en/pq01.arrow", 0)
        _write_sidecars(good_tree, "lang=en/pq01", [])
        _, errors = validator.validate(good_tree)
        assert any("zero tokens" in e for e in errors)

    def test_arrow_without_tokens_column(self, good_tree):
        path = good_tree / "lang=en/pq01.arrow"
        table = pa.table({"wrong_column": pa.array([1, 2, 3], type=pa.uint32())})
        with pa.OSFile(str(path), "wb") as sink:
            writer = pa.ipc.new_file(sink, table.schema)
            writer.write_table(table)
            writer.close()
        _, errors = validator.validate(good_tree)
        assert any("missing 'tokens' column" in e for e in errors)

    def test_missing_metadata_json(self, good_tree):
        (good_tree / "metadata.json").unlink()
        _, errors = validator.validate(good_tree)
        assert any("missing metadata.json" in e for e in errors)

    def test_malformed_metadata_json(self, good_tree):
        (good_tree / "metadata.json").write_text("{not json")
        _, errors = validator.validate(good_tree)
        assert any("not valid JSON" in e for e in errors)

    def test_metadata_token_total_disagreement(self, tmp_path):
        """metadata.json's num_tokens contradicts the actual arrow totals."""
        tree = _build_tree(
            tmp_path / "out",
            {"lang=en/pq01": [("d01", 10)]},
            num_tokens=9999,
        )
        _, errors = validator.validate(tree)
        assert any("reports num_tokens=9999" in e for e in errors)

    def test_no_arrow_files(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        (empty / "metadata.json").write_text(
            json.dumps({"job_output_stats": {"result_files": 0, "num_tokens": 0}})
        )
        _, errors = validator.validate(empty)
        assert any("no .arrow files found" in e for e in errors)
        assert any("no result files" in e for e in errors)

    def test_malformed_docs_ids_line(self, good_tree):
        (good_tree / "meta/lang=en/pq01.docs.ids").write_text("d01, notanumber\n")
        _, errors = validator.validate(good_tree)
        assert any("non-integer token count" in e for e in errors)

    def test_non_positive_token_count(self, good_tree):
        _write_sidecars(good_tree, "lang=en/pq01", [("d01", 0), ("d02", 45)])
        _, errors = validator.validate(good_tree)
        assert any("non-positive token count" in e for e in errors)

    @pytest.mark.parametrize("corrupt", ["truncate", "unlink_meta", "bad_arrow"])
    def test_main_exits_nonzero_on_any_corruption(self, good_tree, tmp_path, corrupt):
        if corrupt == "truncate":
            _write_arrow(good_tree / "lang=en/pq01.arrow", 3)
        elif corrupt == "unlink_meta":
            (good_tree / "meta/lang=en/pq01.docs.ids").unlink()
        else:
            (good_tree / "lang=en/pq01.arrow").write_text("garbage")

        rc = validator.main(
            ["validate_tokens.py", str(good_tree), str(tmp_path / "rpt")]
        )
        assert rc == 1


class TestValidatorCli:
    def test_missing_input_dir_exits_one(self, tmp_path):
        rc = validator.main(
            ["validate_tokens.py", str(tmp_path / "nope"), str(tmp_path / "rpt")]
        )
        assert rc == 1

    def test_wrong_arg_count_exits_two(self):
        assert validator.main(["validate_tokens.py"]) == 2


# --------------------------------------------------------------------------- #
# The template's build.yaml itself
# --------------------------------------------------------------------------- #


class TestTemplateBuildYaml:
    """Guard the template YAML, its parameters, and the rendered shell commands.

    ``command`` uses a folded scalar (``>-``), which collapses newlines — an easy
    thing to break. These checks parse the YAML the way gbcli does (Jinja with
    ``$${`` delimiters and ``StrictUndefined``) so an unresolved or renamed
    parameter fails here rather than at submit time.
    """

    @staticmethod
    def _render() -> dict:
        jinja2 = pytest.importorskip("jinja2")
        params = yaml.safe_load((_TEMPLATE_DIR / "parameters.yaml").read_text())
        template = jinja2.Template(
            (_TEMPLATE_DIR / "build.yaml").read_text(),
            undefined=jinja2.StrictUndefined,
            variable_start_string="$${",
            variable_end_string="}",
            block_start_string="<%",
            block_end_string="%>",
        )
        return yaml.safe_load(template.render(params))

    def test_every_parameter_resolves(self):
        """StrictUndefined raises if build.yaml references an absent parameter."""
        assert self._render()["granite.build"]["name"] == "dpk-tokenize-validate"

    def test_targets_and_binding(self):
        targets = self._render()["granite.build"]["targets"]
        assert set(targets) == {"tokenize", "validate"}
        assert targets["tokenize"]["outputs"]["tokens"]["type"] == "dataset"
        # validate consumes tokenize's output, which is what orders the two.
        assert targets["validate"]["inputs"]["tokens"]["binding"] == "tokenize.tokens"

    def test_both_targets_use_the_builtin_command_step(self):
        """The command step is what makes this portable across Skypilot endpoints."""
        for target in self._render()["granite.build"]["targets"].values():
            assert target["steps"][0]["step_uri"] == "space://steps/command"

    def test_commands_survive_folding_as_single_line_shell(self):
        """A folded scalar must not leave embedded newlines in the command."""
        for name, target in self._render()["granite.build"]["targets"].items():
            command = target["steps"][0]["config"]["command_config"]["command"]
            assert "\n" not in command, f"{name} command contains a raw newline"
            assert command.startswith("set -euo pipefail;"), name

    def test_each_target_emits_its_artifact_marker(self):
        """Outputs are registered from the LLMB_ARTIFACT marker the command echoes."""
        targets = self._render()["granite.build"]["targets"]
        for name, target in targets.items():
            command = target["steps"][0]["config"]["command_config"]["command"]
            for output_name in target["outputs"]:
                assert f"LLMB_ARTIFACT_ID:{output_name} " in command, (
                    name,
                    output_name,
                )

    def test_file_mount_sources_exist_in_repo(self):
        """The rsync'd input dir and validator script must be committed."""
        targets = self._render()["granite.build"]["targets"]
        mounts = {}
        for target in targets.values():
            mounts.update(target["steps"][0]["config"].get("file_mounts") or {})
        assert mounts, "expected at least one file_mount"
        for source in mounts.values():
            assert (
                _REPO_ROOT / source
            ).exists(), f"missing file_mount source: {source}"

    def test_default_environment_is_a_skypilot_endpoint(self):
        env = self._render()["granite.build"]["targets"]["tokenize"]["environment_uri"]
        assert env.startswith("space://environments/skypilot")

    @pytest.mark.parametrize(
        "endpoint",
        [
            "skypilot/slurm",
            "skypilot/kubernetes",
            "skypilot/aws",
            "skypilot/lsf/ibm-bluevela",
        ],
    )
    def test_environment_is_overridable_per_endpoint(self, endpoint):
        """One build.yaml, four endpoints — the whole point of the template."""
        jinja2 = pytest.importorskip("jinja2")
        params = yaml.safe_load((_TEMPLATE_DIR / "parameters.yaml").read_text())
        params["ENVIRONMENT"] = endpoint
        template = jinja2.Template(
            (_TEMPLATE_DIR / "build.yaml").read_text(),
            undefined=jinja2.StrictUndefined,
            variable_start_string="$${",
            variable_end_string="}",
            block_start_string="<%",
            block_end_string="%>",
        )
        rendered = yaml.safe_load(template.render(params))
        for target in rendered["granite.build"]["targets"].values():
            assert target["environment_uri"] == f"space://environments/{endpoint}"

    def test_referenced_skypilot_environments_exist(self):
        """Each documented endpoint resolves to a real environment.yaml."""
        base = _REPO_ROOT / "configurations" / "assets" / "environments"
        for endpoint in ("skypilot/slurm", "skypilot/kubernetes", "skypilot/aws"):
            assert (base / endpoint / "environment.yaml").is_file(), endpoint
