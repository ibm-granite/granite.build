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

import json

import pytest
from merge_shards import MANIFEST_NAME, merge
from prep_corpus import PrepError


def _empty_shard(root, index, count):
    d = root / f"shard-{index}"
    d.mkdir()
    manifest = {
        "shard": {"index": index, "count": count},
        "counts": {"input": 3, "kept": 0, "drop_reasons": {"over_length": 3}},
    }
    (d / MANIFEST_NAME).write_text(json.dumps(manifest))
    return d


def test_a_shard_set_that_kept_nothing_is_refused_before_merging(tmp_path):
    # The token means are divided by the kept count, which used to raise ZeroDivisionError
    # only after the whole lockstep pass over the sidecars.
    shards = [_empty_shard(tmp_path, i, 2) for i in range(2)]
    out = tmp_path / "merged"

    with pytest.raises(PrepError, match="no rows kept across 2 shards") as exc:
        merge(shards, out)

    assert '"over_length": 6' in str(exc.value)
    assert not out.exists()
