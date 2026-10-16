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

"""Unit tests for libgbtest.hfpush: placeholder rendering of hf:// push fixtures."""

import pytest
from libgbtest import hfpush

_BUILD = """\
granite.build:
  targets:
    t:
      inputs:
        i:
          uri: hf:///datasets/ibm-research/vira-intents-live
      outputs:
        a:
          uri: hf://huggingface.co/datasets/__GBTEST_HF_PUSH_ORG__/x-test1___GBTEST_HF_PUSH_RUN__
        b:
          uri: hf://huggingface.co/datasets/__GBTEST_HF_PUSH_ORG__/x-test2___GBTEST_HF_PUSH_RUN__
        e:
          uri: "env:///tmp/out.txt"
"""


def _render(tmp_path, text=_BUILD):
    src = tmp_path / "src" / "build.yaml"
    src.parent.mkdir(parents=True)
    src.write_text(text)
    out = tmp_path / "out"
    out.mkdir()
    return hfpush.render_push_build_yaml(src, out)


def test_default_org_and_run_suffix(tmp_path, monkeypatch):
    monkeypatch.delenv(hfpush.ENV_VAR_GBTEST_HF_PUSH_ORG, raising=False)
    dest, repos = _render(tmp_path)
    text = dest.read_text()
    assert hfpush.ORG_PLACEHOLDER not in text and hfpush.RUN_PLACEHOLDER not in text
    assert len(repos) == 2
    run = repos[0].rsplit("_", 1)[1]
    assert repos == [
        f"ibm-research/x-test1_{run}",
        f"ibm-research/x-test2_{run}",
    ]
    # The public input is not a pushed dataset, so it is never deleted.
    assert "ibm-research/vira-intents-live" not in repos
    assert dest.parent == tmp_path / "out"


def test_org_override_and_fresh_run_each_time(tmp_path, monkeypatch):
    monkeypatch.setenv(hfpush.ENV_VAR_GBTEST_HF_PUSH_ORG, "someone")
    _, first = _render(tmp_path / "1")
    _, second = _render(tmp_path / "2")
    assert all(r.startswith("someone/") for r in first + second)
    assert first != second


def test_fixture_without_placeholders_fails(tmp_path):
    with pytest.raises(AssertionError, match="no hf:// dataset output"):
        _render(tmp_path, _BUILD.replace("___GBTEST_HF_PUSH_RUN__", "_fixed"))
