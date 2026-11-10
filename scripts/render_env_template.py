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

"""Render a $${KEY}-templated file from a site file.

    render_env_template.py TEMPLATE SITE OUT

The template uses the gb CLI's build.yaml parameter syntax: $${KEY} for values
and <% %> for blocks, rendered by gbcli's apply_parameters under StrictUndefined.
gbserver's own {{ ... }} and {% %} templates pass through untouched. SITE is a
flat YAML mapping, loaded as `gb build start --parameters-path` loads it.

OUT is written only when the result renders and parses as YAML: atomically, from
a temporary file in OUT's directory, with mode 600.

Exit codes: 0 written; 1 bad arguments or I/O error; 2 a key the template uses
is not in SITE; 3 the rendered text is not valid YAML.
"""

import os
import sys
import tempfile

import yaml
from jinja2.exceptions import TemplateSyntaxError, UndefinedError

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters


def _err(msg: str) -> None:
    print(f"render_env_template: {msg}", file=sys.stderr)


def render(template_path: str, site_path: str) -> str:
    """The template rendered with the site file's values."""
    with open(template_path, "r", encoding="utf-8") as f:
        text = f.read()
    site = get_params_from_file(site_path)
    # apply_parameters writes the applied parameters next to the build; keep
    # that copy of the site values in a scratch dir that is removed at once.
    with tempfile.TemporaryDirectory() as scratch:
        rendered = apply_parameters(text, [], site, scratch)
    # Jinja drops the final newline.
    if text.endswith("\n") and not rendered.endswith("\n"):
        rendered += "\n"
    return rendered


def write_atomic(path: str, text: str) -> None:
    """Write text to path via a temporary file in the same dir, mode 600."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main(argv: list) -> int:
    if len(argv) != 3:
        _err("usage: render_env_template.py TEMPLATE SITE OUT")
        return 1
    template_path, site_path, out_path = argv
    for p in (template_path, site_path):
        if not os.path.isfile(p):
            _err(f"{p}: no such file")
            return 1
    try:
        rendered = render(template_path, site_path)
    except UndefinedError as e:
        _err(f"{template_path}: {e} in {site_path}")
        return 2
    except TemplateSyntaxError as e:
        _err(f"{template_path}:{e.lineno}: template syntax error: {e.message}")
        return 1
    except (OSError, yaml.YAMLError) as e:
        _err(f"{e}")
        return 1
    except Exception as e:  # get_params_from_file raises a bare Exception
        _err(f"{site_path}: {e}")
        return 1
    try:
        yaml.safe_load(rendered)
    except yaml.YAMLError as e:
        _err(f"{template_path} does not render to valid YAML: {e}")
        return 3
    try:
        write_atomic(out_path, rendered)
    except OSError as e:
        _err(f"{out_path}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
