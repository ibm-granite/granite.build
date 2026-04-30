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

"""
Emulate the Helm merge done by a helm install/template
Example:

helm \
    install \
    gbmzovj1az \
    -f \
    /app/gbserverworkspace/tmp6f0l_pva/builds/4fb4411a-dc4d-4aa2-a8bd-972945b8603e/targetruns/579d9e02-a991-4db9-85e5-fec8582bfa7f/helm-charts/lhpull/values-default.yaml \
    -f \
    /app/gbserverworkspace/tmp6f0l_pva/builds/4fb4411a-dc4d-4aa2-a8bd-972945b8603e/targetruns/579d9e02-a991-4db9-85e5-fec8582bfa7f/helm-charts/lhpull/values.yaml \
    -f \
    /app/gbserverworkspace/tmp6f0l_pva/builds/4fb4411a-dc4d-4aa2-a8bd-972945b8603e/targetruns/579d9e02-a991-4db9-85e5-fec8582bfa7f/helm-charts/lhpull/values-config.yaml \
    --set \
    run_metadata.launch_id=4afa5690-6a31-46e8-a628-f5bb0e5fbe73 \
    /app/gbserverworkspace/tmp6f0l_pva/builds/4fb4411a-dc4d-4aa2-a8bd-972945b8603e/targetruns/579d9e02-a991-4db9-85e5-fec8582bfa7f/helm-charts/lhpull \
    --kubeconfig \
    /tmp/tmpxuwbw1l_
"""

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError as e:
    print("[ERROR] helm merge values script failed with import error:", e)
    sys.exit(1)


def merge_dicts(x: Dict, y: Dict) -> Dict:
    assert isinstance(x, dict), f"invalid x: {x}"
    assert isinstance(y, dict), f"invalid y: {y}"
    merged = {}
    merged = {**x}
    for k, v in y.items():
        if k not in merged:
            merged[k] = v
            continue
        base = merged[k]
        overlay = v
        if type(base) != type(overlay):
            merged[k] = v
            continue
        if not isinstance(base, dict):
            merged[k] = v
            continue
        merged[k] = merge_dicts(base, overlay)
    return merged


def remove_nulls(x: Any) -> Any:
    if isinstance(x, dict):
        result = {}
        for k, v in x.items():
            assert k is not None, f"invalid k: {k}"
            vv = remove_nulls(v)
            if vv is not None:
                result[k] = vv
        return result
    if isinstance(x, list):
        result = []
        for v in x:
            vv = remove_nulls(v)
            if vv is not None:
                result.append(vv)
        return result
    return x


def merge_values(args: Namespace) -> None:
    output_merged_path: Path = args.output_merged.resolve()
    input_values_default_path: Path = args.input_values_default.resolve()
    input_values_path: Path = args.input_values.resolve()
    input_values_config_path: Path = args.input_values_config.resolve()
    assert (
        input_values_default_path.is_file()
    ), f"invalid input_values_default_path: {input_values_default_path}"
    assert input_values_path.is_file(), f"invalid input_values_path: {input_values_path}"
    assert (
        input_values_config_path.is_file()
    ), f"invalid input_values_config_path: {input_values_config_path}"
    with open(input_values_default_path, "r", encoding="utf-8") as f:
        input_values_default = yaml.safe_load(f)
    with open(input_values_path, "r", encoding="utf-8") as f:
        input_values = yaml.safe_load(f)
    with open(input_values_config_path, "r", encoding="utf-8") as f:
        input_values_config = yaml.safe_load(f)
    t1_merged_values = merge_dicts(input_values_default, input_values)
    t2_merged_values = merge_dicts(t1_merged_values, input_values_config)
    merged_values = remove_nulls(t2_merged_values)
    assert isinstance(merged_values, dict), f"invalid merged_values: {merged_values}"
    with open(output_merged_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged_values, f)
    print("[INFO] output merged values written to", output_merged_path)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--input-values",
        dest="input_values",
        type=Path,
        default="values.yaml",
        required=False,
        help="input path for values.yaml",
    )
    parser.add_argument(
        "--input-values-config",
        dest="input_values_config",
        type=Path,
        default="values-config.yaml",
        required=False,
        help="input path for values-config.yaml",
    )
    parser.add_argument(
        "--input-values-default",
        dest="input_values_default",
        type=Path,
        default="values-default.yaml",
        required=False,
        help="input path for values-default.yaml",
    )
    parser.add_argument(
        "--output-merged",
        dest="output_merged",
        type=Path,
        default="output-values.yaml",
        required=False,
        help="output path",
    )
    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    print("[INFO] helm merge script running with args:", args)
    merge_values(args=args)


if __name__ == "__main__":
    main()
