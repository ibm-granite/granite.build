"""Every DPO loss_type a shipped config can sample must exist in the installed TRL.

TRL rejects an unknown loss type only when it first computes the loss, i.e. inside
training, so a stale value in a search space fails just the HPO trials that
happen to sample it. ``kto_pair`` sat in every shipped DPO search space for
several TRL releases after TRL dropped it (before 0.29).
"""

import inspect
from pathlib import Path

import pytest
import yaml

trl_dpo = pytest.importorskip("trl.trainer.dpo_trainer")

CONFIG_DIR = Path(__file__).resolve().parents[1] / "autotune" / "configs"


def _dpo_loss_types():
    out = []
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        hp = (((cfg or {}).get("tuners_rl_config") or {}).get("dpo") or {}).get("hyperparams") or {}
        spec = hp.get("loss_type")
        if spec:
            for value in set(spec.get("values") or []) | {spec.get("default")}:
                out.append((path.name, value))
    return out


LOSS_TYPES = _dpo_loss_types()


def test_configs_have_dpo_loss_types():
    assert len(LOSS_TYPES) >= 10, LOSS_TYPES  # guards the guard against a glob that matches nothing


@pytest.mark.parametrize("config, loss_type", LOSS_TYPES)
def test_loss_type_is_implemented_by_installed_trl(config, loss_type):
    source = inspect.getsource(trl_dpo.DPOTrainer)
    assert f'loss_type == "{loss_type}"' in source, (
        f'{config}: DPO loss_type "{loss_type}" has no branch in the installed TRL DPOTrainer '
        "(TRL would raise 'Unknown loss type' mid-trial)"
    )
