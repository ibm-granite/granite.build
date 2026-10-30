"""ns's two-turn CritPt generation (``nemo_skills.inference.eval.critpt``)
with ``inference.temperature`` allowed to be null, as in ns's own
``generate.InferenceConfig``: its CritPt config types it as a plain float
with a greedy 0.0 default, so ``++inference.temperature=null`` (send no
sampling parameter; the checkpoint's generation_config applies) is refused.

Run as ``python -m granite_evals.ns_critpt`` (needs nemo_skills installed)."""

import sys
from dataclasses import field

import hydra
from nemo_skills.inference.eval import critpt
from nemo_skills.utils import nested_dataclass, setup_logging


@nested_dataclass(kw_only=True)
class InferenceConfig(critpt.CritPtInferenceConfig):
    temperature: float | None = None


@nested_dataclass(kw_only=True)
class GenerationConfig(critpt.CritPtGenerationConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)


hydra.core.config_store.ConfigStore.instance().store(name="granite_critpt_generation_config", node=GenerationConfig)


@hydra.main(version_base=None, config_name="granite_critpt_generation_config")
def generate(cfg) -> None:
    cfg = GenerationConfig(_init_nested=True, **cfg)
    critpt.LOG.info("Config used: %s", cfg)
    critpt.CritPtGenerationTask(cfg).generate()


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(critpt.HELP_MESSAGE)
    else:
        setup_logging()
        generate()
