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

"""Utility for loading JSON fixture files from test-data/lib/fixtures/.

Fixtures are canned API responses used by mock-mode test fixtures, stored
in the existing test-data/ directory alongside other test configuration files.
"""

import json
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent.parent.parent / "test-data" / "lib" / "fixtures"


def load_fixture(service: str, name: str) -> dict:
    """Load a JSON fixture file.

    Args:
        service: Subdirectory name (e.g. "github", "kubernetes").
        name: Filename with extension (e.g. "user.json").

    Returns:
        Parsed JSON as a dict.
    """
    path = FIXTURE_DIR / service / name
    return json.loads(path.read_text())
