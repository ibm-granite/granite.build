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

"""Put this step's ``src/`` on ``sys.path`` so the tests import it by flat name."""

import sys
from pathlib import Path

_OWN_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_OWN_SRC) not in sys.path:
    sys.path.insert(0, str(_OWN_SRC))
