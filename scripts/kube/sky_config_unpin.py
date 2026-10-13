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

"""Undo what skypilot/kubernetes/local leaves in the global SkyPilot config.

gbserver deep-merges an environment's ``cloud_config`` into ``~/.sky/config.yaml``
and never removes keys, so after a build on skypilot/kubernetes/local the file
keeps ``kubernetes.allowed_contexts: [<context>]`` — which limits ALL later
SkyPilot Kubernetes use on the machine to the kind cluster — plus that context's
``context_configs`` entry. ``make kube-teardown`` runs this to remove them.

Removes only what the env adds:
  * ``kubernetes.allowed_contexts``, only when it is exactly ``[<context>]``
    (a list the user edited is left alone, with a note);
  * ``kubernetes.context_configs.<context>``;
  * then any of those sections left empty.
Everything else is preserved, and the file is replaced atomically.

Usage: sky_config_unpin.py <context> [<config-path>]   (default ~/.sky/config.yaml)
"""

import os
import sys
import tempfile

import yaml


def unpin(config: dict, context: str) -> list[str]:
    """Remove the env's keys from ``config`` in place; return what was done."""
    done: list[str] = []
    kube = config.get("kubernetes")
    if not isinstance(kube, dict):
        return done
    allowed = kube.get("allowed_contexts")
    if allowed == [context]:
        del kube["allowed_contexts"]
        done.append(f"removed kubernetes.allowed_contexts [{context}]")
    elif allowed is not None:
        done.append(
            f"left kubernetes.allowed_contexts {allowed!r} (not exactly [{context}])"
        )
    ctx_cfgs = kube.get("context_configs")
    if isinstance(ctx_cfgs, dict) and context in ctx_cfgs:
        del ctx_cfgs[context]
        done.append(f"removed kubernetes.context_configs.{context}")
        if not ctx_cfgs:
            del kube["context_configs"]
    if not kube:
        del config["kubernetes"]
    return done


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    context = argv[1]
    path = argv[2] if len(argv) == 3 else os.path.expanduser("~/.sky/config.yaml")
    if not os.path.exists(path):
        print(f"sky-config: {path} does not exist; nothing to undo")
        return 0
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    done = unpin(config, context)
    if not any(d.startswith("removed") for d in done):
        for d in done:
            print(f"sky-config: {d}")
        print(f"sky-config: no {context} pin in {path}; left unchanged")
        return 0
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".config.yaml.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        os.chmod(tmp, os.stat(path).st_mode & 0o777)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise
    for d in done:
        print(f"sky-config: {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
