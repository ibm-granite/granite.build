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

# Image setup for the `judged` extra (profbench, gdpval).
# ProfBench's scripts are not a package: bake the pinned commit's files into
# /opt/profbench/<commit>, verified by sha256 (same pins as
# granite_evals/benchmarks/judge_general.py PROFBENCH_FILES).
COMMIT=b06a29cda4d1433e9a9aad8171e4086299083e94
DEST=/opt/profbench/$COMMIT
mkdir -p "$DEST"
while read -r sha name; do
    curl -fsSL --retry 5 -o "$DEST/$name" "https://raw.githubusercontent.com/NVlabs/ProfBench/$COMMIT/$name"
    echo "$sha  $DEST/$name" | sha256sum -c -
done <<'PINS'
6aa796c531e1601369cd7249e1f33b0a941e993ef90a7e9004170ee01665d3fe utils.py
f8cf57e2b02ba4e667281e27c942bd5d5b9e8ea3550d51ff565a71bc8b4ac179 score_report_generation.py
e24beb96496250a4f372d5626bd94681aabca870538e9708cf8086727ca9d55f run_report_generation.py
d9c925aa95402162cba1dcca200822ec43b0ddc23c8e25a2e29f87649ccfe059 score_llm_judge.py
5dc788c299ee9525feb4053830c1358c529c67a906e461311e2200a89f0d5ffc LICENSE
PINS
chmod -R a+rX /opt/profbench
