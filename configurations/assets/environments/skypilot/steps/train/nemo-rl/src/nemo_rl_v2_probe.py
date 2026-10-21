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

"""Report whether a NeMo-RL config trains with the V2 DTensor worker.

nemo_rl_run.sh forces checkpointing.model_save_format=torch_save only for the V2
worker (V1 rejects a non-null model_save_format). The answer must come from the
config the DRIVER sees: `defaults:` inheritance (recipes inherit _v2 from a
parent) plus the Hydra overrides (a user may set policy.dtensor_cfg._v2). So it
is resolved with NeMo-RL's own loader, exactly as examples/run_<algo>.py does,
inside the image's environment.

CONTRACT
  nemo_rl_v2_probe.py <config.yaml> [hydra overrides...]
  prints one final line: NEMO_RL_V2=true | NEMO_RL_V2=false
"""

import sys
from collections.abc import Mapping


def uses_v2(cfg: Mapping) -> bool:
    """policy.dtensor_cfg._v2, the same test lm_policy uses to pick the worker."""
    policy = cfg.get("policy") or {}
    dtensor = policy.get("dtensor_cfg") or {}
    return bool(dtensor.get("_v2", False))


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        sys.exit("usage: nemo_rl_v2_probe.py <config.yaml> [hydra overrides...]")
    config_path, overrides = argv[0], argv[1:]

    from nemo_rl.utils import config as nrl_config

    nrl_config.register_omegaconf_resolvers()
    cfg = nrl_config.load_config(config_path)
    if overrides:
        cfg = nrl_config.parse_hydra_overrides(cfg, overrides)
    print(f"NEMO_RL_V2={'true' if uses_v2(cfg) else 'false'}")


if __name__ == "__main__":
    main()
