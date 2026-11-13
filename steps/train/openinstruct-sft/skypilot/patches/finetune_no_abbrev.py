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

"""Build-time patch for upstream open-instruct finetune.py (pinned commit).

finetune.py parses its flags with HfArgumentParser, which leaves argparse's
``allow_abbrev`` on: a unique prefix such as ``--learning_rat`` silently sets
``learning_rate``, and a prefix can change meaning when a later pin adds a flag.
Construct the parser with ``allow_abbrev=False`` (HfArgumentParser forwards it to
argparse), so only full flag names are accepted. Exits non-zero if the target
line is not found exactly once, so an upstream change fails the image build
instead of silently shipping an unpatched parser.
"""

import pathlib
import sys

OLD = "    parser = ArgumentParserPlus((FlatArguments, TokenizerConfig))\n"
NEW = "    parser = ArgumentParserPlus((FlatArguments, TokenizerConfig), allow_abbrev=False)\n"


def patch(path: pathlib.Path) -> None:
    """Apply the no-abbreviation patch to ``path`` in place.

    :param path: open_instruct/finetune.py.
    :raises SystemExit: If the target is not found exactly once.
    """
    text = path.read_text()
    if text.count(OLD) != 1:
        sys.exit(
            f"openinstruct-sft: allow_abbrev patch target not found exactly once in {path}"
        )
    path.write_text(text.replace(OLD, NEW))


if __name__ == "__main__":
    patch(pathlib.Path(sys.argv[1]))
