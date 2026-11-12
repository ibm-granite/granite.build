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
and print any binding that still reaches an unkeyed cache.
"""

import asyncio
import json
import os
import sys
import tempfile

import yaml

mode = sys.argv[1]
if mode == "guard":
    import functools
    import importlib
    import inspect

    from gbserver.utils import sky_patches

    importlib.import_module("sky.adaptors.kubernetes")
    importlib.import_module("sky.provision.kubernetes.utils")
    names = [(c.__module__, c.__qualname__) for c in sky_patches.kube_request_caches()]
    sky_patches.apply()
    unkeyed = []
    for module, qualname in names:
        owner = importlib.import_module(module)
        *path, name = qualname.split(".")
        for part in path:
            owner = getattr(owner, part)
        bound = inspect.getattr_static(owner, name)
        fn = bound.__func__ if isinstance(bound, classmethod) else bound
        reached = [fn, getattr(fn, "__wrapped__", None)]
        if any(isinstance(f, functools._lru_cache_wrapper) for f in reached):
            unkeyed.append(f"{module}.{qualname}")
    print(json.dumps({"count": len(names), "unkeyed": unkeyed}))
    sys.exit(0)
patched = mode == "patched"
from sky.utils import context, context_utils

context_utils.hijack_sys_attrs()  # as the server's uvicorn process does
if patched:
    from gbserver.utils import sky_patches

    sky_patches.apply()
from sky.adaptors import kubernetes as k8s

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
                "contexts": [{"name": "X", "context": {"cluster": "c", "user": "u"}}],
                "current-context": "X",
            },
            f,
        )
    files[name] = path


async def request(path):
    context.initialize()
    context.get().override_envs({"KUBECONFIG": path})
    api = await asyncio.to_thread(k8s.core_api, "X")
    return api.api_client.configuration.host


async def main():
    a = await request(files["a"])
    b = await request(files["b"])
    both = await asyncio.gather(request(files["a"]), request(files["b"]))
    print(json.dumps({"sequential": [a, b], "concurrent": list(both)}))


asyncio.run(main())
