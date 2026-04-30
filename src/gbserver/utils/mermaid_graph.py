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

"""Mermaid graph module."""

from typing import List, Set, Tuple

from gbserver.types.buildconfig import BuildConfig


def get_mermaid_graph_str(build_config: BuildConfig) -> str:
    """Get a Mermaid graph representation of a build.yaml"""
    edges: List[Tuple[str, str, str]] = []  # src, dest, edge label
    target_names: List[str] = []
    asset_store_inputs = 'id0[("Storage")]'
    asset_store_outputs = 'id1[("Storage")]'
    all_outputs: Set[Tuple[str, str]] = set()
    for t_name, target in build_config.targets.items():
        target_names.append(t_name)
        if target.inputs is not None:
            for t_input_name, t_input in target.inputs.items():
                if t_input.uri:
                    # edge = (asset_store_inputs, t_name, t_input.uri)
                    edge = (asset_store_inputs, t_name, t_input_name)
                    edges.append(edge)
                    continue
                if t_input.binding:
                    src_t_name, src_t_output_name = t_input.binding.split(".")
                    edge = (src_t_name, t_name, src_t_output_name)
                    edges.append(edge)
                    all_outputs.discard((src_t_name, src_t_output_name))
                    continue
        if target.outputs is not None:
            for t_output_name, t_output in target.outputs.items():
                all_outputs.add((t_name, t_output_name))
                # if t_output.uri:
                #     edge = (t_name, asset_store_outputs, t_output_name)
                #     edges.append(edge)
                #     continue
    graph_str = "flowchart TD\n"
    indent_str = "    "
    for t_name, t_output_name in all_outputs:
        edges.append((t_name, asset_store_outputs, t_output_name))
    for t_name in target_names:
        graph_str += f"{indent_str}{t_name}\n"
    for edge in edges:
        src, dest, edge_label = edge
        graph_str += f"{indent_str}{src} -->|{edge_label}| {dest}\n"

    return graph_str
