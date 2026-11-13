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

"""Probe for test_skypilot_kube.TestServerClientCaches, run in a fresh process.

``patched`` / ``unpatched``: recreate the API server's main-process path
(``/validate``, ``/logs``): per-request env on shared threads. Prints the
cluster each request's ``core_api("X")`` client points at.

``guard``: list SkyPilot's request-scoped Kubernetes caches, apply the patches,
and print any binding that still reaches an unkeyed cache (checked on the
binding itself: the module attribute, the classmethod's function, or the
closure cells of an outer decorator).

``paths`` (patched): per-request results for a zero-argument cache (the current
context), a cache taking a context (its namespace), and that SkyPilot's
per-request clearing empties the keyed caches.
"""

import asyncio
import functools
import importlib
import inspect
import json
import os
import sys
import tempfile

import yaml

mode = sys.argv[1]


def _reaches(obj, pred):
    """Whether ``obj`` or a closure cell of it satisfies ``pred``."""
    if pred(obj):
        return True
    return any(pred(c.cell_contents) for c in (getattr(obj, "__closure__", None) or ()))


if mode == "guard":
    from gbserver.utils import sky_patches

    sky_patches.import_kube_modules()
    originals = sky_patches.kube_request_caches()
    names = [(c.__module__, c.__qualname__) for c in originals]
    sky_patches.apply()
    unkeyed = []
    for cached, (module, qualname) in zip(originals, names):
        owner = importlib.import_module(module)
        *path, name = qualname.split(".")
        for part in path:
            owner = getattr(owner, part)
        bound = inspect.getattr_static(owner, name)
        fn = bound.__func__ if isinstance(bound, (classmethod, staticmethod)) else bound
        still_original = _reaches(fn, lambda o, c=cached: o is c)
        keyed = _reaches(fn, lambda o: getattr(o, "gbserver_kubeconfig_keyed", False))
        if still_original or not keyed:
            unkeyed.append(f"{module}.{qualname}")
    print(json.dumps({"count": len(names), "unkeyed": unkeyed}))
    sys.exit(0)

patched = mode in ("patched", "paths")
from sky.utils import annotations, context, context_utils

context_utils.hijack_sys_attrs()  # as the server's uvicorn process does
if patched:
    from gbserver.utils import sky_patches

    sky_patches.apply()
from sky.adaptors import kubernetes as k8s
from sky.provision.kubernetes import utils as kube_utils

d = tempfile.mkdtemp()
files = {}
for name, host, token in (
    ("a", "https://10.0.0.1:6443", "tok-a"),
    ("b", "https://10.0.0.2:6443", "tok-b"),
):
    path = os.path.join(d, f"{name}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(
            {
                "apiVersion": "v1",
                "kind": "Config",
                "clusters": [
                    {
                        "name": "c",
                        "cluster": {"server": host, "insecure-skip-tls-verify": True},
                    }
                ],
                "users": [{"name": "u", "user": {"token": token}}],
                "contexts": [
                    {
                        "name": "X",
                        "context": {
                            "cluster": "c",
                            "user": "u",
                            "namespace": f"X-{name}",
                        },
                    },
                    {"name": f"ctx-{name}", "context": {"cluster": "c", "user": "u"}},
                ],
                "current-context": f"ctx-{name}",
            },
            f,
        )
    files[name] = path


async def in_request(path, fn, *args):
    context.initialize()
    context.get().override_envs({"KUBECONFIG": path})
    return await asyncio.to_thread(fn, *args)


async def main():
    host = lambda: k8s.core_api("X").api_client.configuration.host
    if mode == "paths":
        out = {
            "current": [
                await in_request(
                    files[n], kube_utils.get_current_kube_config_context_name
                )
                for n in "ab"
            ],
            "namespace": [
                await in_request(
                    files[n], kube_utils.get_kube_config_context_namespace, "X"
                )
                for n in "ab"
            ],
        }
        annotations.clear_request_level_cache()
        out["after_clear"] = [
            kube_utils.get_current_kube_config_context_name.cache_info().currsize,
            kube_utils.get_kube_config_context_namespace.cache_info().currsize,
        ]
        return out
    # Cold first: both requests race to build the client for context X.
    concurrent = await asyncio.gather(
        in_request(files["a"], host), in_request(files["b"], host)
    )
    sequential = [
        await in_request(files["a"], host),
        await in_request(files["b"], host),
    ]
    return {"concurrent": list(concurrent), "sequential": sequential}


print(json.dumps(asyncio.run(main())))
