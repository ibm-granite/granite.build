# Copyright Granite.Build Authors
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

"""scripts/check_license_headers.py: which files need a header, and where --fix puts it.

Each test writes files under a temporary repo-shaped directory, chdirs into it and
calls the script's main() with explicit paths, so no git checkout is needed.
"""

import importlib.util
import pathlib
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "check_license_headers.py"

_spec = importlib.util.spec_from_file_location("check_license_headers", SCRIPT)
assert _spec is not None and _spec.loader is not None
clh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clh)

HASH_HEADER = "\n".join(clh.render_header("hash"))
SLASH_HEADER = "\n".join(clh.render_header("slash"))


@pytest.fixture(name="repo")
def _repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def write(rel: str, content: str | bytes) -> pathlib.Path:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content, encoding="utf-8")
        return p

    return write


def fix(path: str) -> str:
    assert clh.main(["--fix", path]) == 1
    assert clh.main([path]) == 0, "a fixed file must pass the check"
    return pathlib.Path(path).read_text(encoding="utf-8")


def test_plain_python_gets_header_first(repo):
    repo("src/pkg/mod.py", '"""Doc."""\n\nimport os\n')
    assert clh.main(["src/pkg/mod.py"]) == 1
    assert fix("src/pkg/mod.py") == f'{HASH_HEADER}\n\n"""Doc."""\n\nimport os\n'


def test_shebang_stays_first(repo):
    repo("scripts/run.sh", "#!/usr/bin/env bash\nset -e\n")
    assert fix("scripts/run.sh") == f"#!/usr/bin/env bash\n\n{HASH_HEADER}\n\nset -e\n"


def test_shebang_blank_line_not_doubled(repo):
    repo("scripts/run.sh", "#!/bin/sh\n\n\necho hi\n")
    assert fix("scripts/run.sh") == f"#!/bin/sh\n\n{HASH_HEADER}\n\necho hi\n"


def test_python_coding_line_stays_in_first_two_lines(repo):
    repo("src/m.py", "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\nx = 1\n")
    out = fix("src/m.py").splitlines()
    assert out[:2] == ["#!/usr/bin/env python3", "# -*- coding: utf-8 -*-"]
    assert out[3] == "# Copyright Granite.Build Authors"


def test_dockerfile_parser_directive_stays_first(repo):
    repo("steps/x/Dockerfile", "# syntax=docker/dockerfile:1\nFROM scratch\n")
    out = fix("steps/x/Dockerfile")
    assert out.startswith(
        "# syntax=docker/dockerfile:1\n\n# Copyright Granite.Build Authors"
    )


def test_typescript_uses_slash_comments_before_use_client(repo):
    repo("frontend/app/page.tsx", "'use client'\n\nexport default 1\n")
    assert (
        fix("frontend/app/page.tsx")
        == f"{SLASH_HEADER}\n\n'use client'\n\nexport default 1\n"
    )


def test_svelte_uses_html_comment_block(repo):
    repo("autotunex/src/ux/A.svelte", "<script>\n</script>\n")
    out = fix("autotunex/src/ux/A.svelte")
    assert out.startswith("<!--\n  Copyright Granite.Build Authors\n")
    assert "\n-->\n\n<script>" in out


def test_crlf_and_bom_preserved(repo):
    repo("src/w.py", b"\xef\xbb\xbfimport os\r\n")
    clh.main(["--fix", "src/w.py"])
    raw = pathlib.Path("src/w.py").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf# Copyright Granite.Build Authors\r\n")
    assert raw.endswith(b"\r\n\r\nimport os\r\n")
    assert b"\n" not in raw.replace(b"\r\n", b"")


def test_fix_changes_nothing_but_the_inserted_lines(repo):
    # str.splitlines() would split on the form feed, U+2028 and the lone CR,
    # and would normalise the mixed CRLF/LF endings.
    body = "a = 1\r\n\x0c\nb = 'x\u2028y'\nc = '\r'\n"
    repo("src/f.py", body)
    clh.main(["--fix", "src/f.py"])
    out = pathlib.Path("src/f.py").read_bytes().decode("utf-8")
    assert out == "\r\n".join(clh.render_header("hash")) + "\r\n\r\n" + body


@pytest.mark.parametrize(
    "first", ["def f():", "class C:", "async def f():", "@decorator"]
)
def test_python_top_level_def_gets_two_blank_lines(repo, first):
    # black requires two blank lines between a comment block and a def/class.
    repo("src/d.py", f"{first}\n    pass\n")
    assert fix("src/d.py") == f"{HASH_HEADER}\n\n\n{first}\n    pass\n"


def test_shell_function_keeps_one_blank_line(repo):
    repo("scripts/f.sh", "def() { :; }\n")
    assert fix("scripts/f.sh") == f"{HASH_HEADER}\n\ndef() {{ :; }}\n"


def test_no_trailing_newline_kept(repo):
    repo("scripts/n.sh", "#!/bin/sh\necho hi")
    assert fix("scripts/n.sh") == f"#!/bin/sh\n\n{HASH_HEADER}\n\necho hi"


def test_shebang_only_file(repo):
    repo("scripts/s.sh", "#!/bin/sh")
    assert fix("scripts/s.sh") == f"#!/bin/sh\n\n{HASH_HEADER}\n"


@pytest.mark.parametrize(
    "header",
    [
        HASH_HEADER,
        "# Copyright Granite.Build Authors\n# SPDX-License-Identifier: Apache-2.0",
        "#!/bin/sh\n" + HASH_HEADER,
    ],
)
def test_existing_headers_accepted(repo, header):
    repo("src/ok.sh", f"{header}\n\necho ok\n")
    assert clh.main(["src/ok.sh"]) == 0


@pytest.mark.parametrize("content", ["", "\n\n  \n"])
def test_empty_files_need_no_header(repo, content):
    repo("test/pkg/__init__.py", content)
    assert clh.main(["test/pkg/__init__.py"]) == 0


def test_ported_file_skipped(repo):
    repo(
        "steps/d/src/p.py",
        "#\n# PORTED, not authored here. Upstream source of truth:\nx = 1\n",
    )
    assert clh.main(["--fix", "steps/d/src/p.py"]) == 0


def test_other_holder_reported_not_fixed(repo, capsys):
    original = "# Copyright Example Corp. 2024\nx = 1\n"
    repo("src/o.py", original)
    assert clh.main(["--fix", "src/o.py"]) == 1
    captured = capsys.readouterr()
    assert "different copyright holder" in captured.out
    assert "PORTED, not authored here" in captured.err  # how to resolve it
    assert pathlib.Path("src/o.py").read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    "line",
    [
        "# Copyright 2020 Example Inc. (exported)",  # "ported" inside a word
        "# Copyright 2020 Example Inc. -- supported build",
        "# (C) Copyright Example Corp. 2024.",
    ],
)
def test_other_holder_not_mistaken_for_attribution(repo, line):
    repo("src/o.py", f"{line}\nx = 1\n")
    assert clh.main(["--fix", "src/o.py"]) == 1
    assert pathlib.Path("src/o.py").read_text(encoding="utf-8") == f"{line}\nx = 1\n"


@pytest.mark.parametrize(
    "line",
    [
        "# Copyright The HuggingFace Team, adapted from peft",
        "# Copyright the SkyPilot authors, from https://github.com/skypilot-org",
    ],
)
def test_third_party_attribution_not_reported(repo, line):
    repo("src/a.py", f"{line}\nx = 1\n")
    assert fix("src/a.py").endswith(f"\n\n{line}\nx = 1\n")


@pytest.mark.parametrize(
    "code", ["copyright = 'see NOTICE'", "COPYRIGHT: str = ''", "copyright(year)"]
)
def test_code_mentioning_copyright_is_not_a_holder(repo, code):
    repo("src/c.py", f"{code}\n")
    assert fix("src/c.py").endswith(f"\n\n{code}\n")


def test_holder_without_license_notice_reported(repo, capsys):
    repo("src/h.py", "# Copyright Granite.Build Authors\nx = 1\n")
    assert clh.main(["src/h.py"]) == 1
    assert "no Apache-2.0 license notice" in capsys.readouterr().out


@pytest.mark.parametrize(
    "path",
    [
        "docs/x.py",  # outside the code directories
        "src/config.yaml",  # not a source type
        "src/README.md",
        "steps/train/foo/skypilot/test-data/fixture.py",
        "frontend/apps/standalone/next-env.d.ts",
        "steps/eval/src/core/cli.py",  # vendored granite-evals
        "steps/eval/tau/src/tau.py",
        "steps/eval/tau/tests/test_tau.py",
    ],
)
def test_out_of_scope_paths_ignored(repo, path):
    repo(path, "x = 1\n")
    assert clh.main(["--fix", path]) == 0
    assert pathlib.Path(path).read_text(encoding="utf-8") == "x = 1\n"


@pytest.mark.parametrize(
    "path",
    [
        "steps/eval/eval-example/skypilot/src/eval.sh",  # ours, not vendored
        "steps/eval/tau/skypilot/test/test_step_template.py",
        "steps/train/foo/skypilot/Makefile",
        "steps/common.mk",
        "autotunex/Dockerfile.aio",
        "Makefile",  # repo root
        "Dockerfile.monitoring",
        "frontend/apps/standalone/next.config.ts",
    ],
)
def test_in_scope_paths_checked(repo, path):
    repo(path, "x = 1\n")
    assert clh.main([path]) == 1


@pytest.mark.parametrize("form", ["absolute", "dot-slash"])
def test_paths_normalised_to_repo_relative(repo, tmp_path, form):
    repo("src/x.py", "x = 1\n")
    arg = str(tmp_path / "src/x.py") if form == "absolute" else "./src/x.py"
    assert clh.main([arg]) == 1


def test_run_from_subdirectory_checks_whole_repo(repo, tmp_path, monkeypatch):
    repo("src/x.py", "x = 1\n")
    repo("src/pkg/y.py", HASH_HEADER + "\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    monkeypatch.chdir(tmp_path / "src/pkg")
    assert clh.main([]) == 1  # src/x.py is still found from src/pkg/


@pytest.mark.parametrize(
    "body",
    [
        "def f():\n    pass\n",
        "class C:\n    pass\n",
        "@dec\ndef f():\n    pass\n",
        "import os\n\n\ndef f():\n    pass\n",
        '"""Doc."""\n\nimport os\n',
        "# a comment\ndef f():\n    pass\n",
        "#!/usr/bin/env python3\ndef f():\n    pass\n",
        "#!/usr/bin/env python3\n\n\nimport os\n",
        "x = 1\n",
    ],
)
def test_fixed_python_is_black_clean(repo, body):
    black = pytest.importorskip("black")
    repo("src/b.py", body)
    out = fix("src/b.py")
    assert black.format_str(out, mode=black.Mode()) == out
