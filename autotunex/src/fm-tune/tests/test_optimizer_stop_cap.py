"""Tests for autotune.optimizer._stop_dict_* and _auto_derive_asha_max_t.

The stop dicts no longer cap training_iteration — the HF Trainer bounds
training duration via num_train_epochs, so adding a redundant controller-
level cap raced with the driver's terminal tune.report() and silently
broke BLDS arm updates and driver cleanup. STOP_LOSS remains as the
converged-trial safety net. See plan squiggly-bouncing-noether.md.

The asha_max_t auto-derivation keeps ASHA's bracket math aligned with the
trial's actual epoch budget by pulling max_t from hpo_num_epochs.
"""

from autotune.optimizer import (
    STOP_LOSS,
    _auto_derive_asha_max_t,
    _stop_dict_for_final,
    _stop_dict_for_hpo,
)


class TestHpoStopDict:
    def test_only_contains_loss_key(self):
        d = _stop_dict_for_hpo({"hpo_num_epochs": 5})
        assert set(d.keys()) == {"loss"}

    def test_uses_stop_loss(self):
        d = _stop_dict_for_hpo({"hpo_num_epochs": 5})
        assert d["loss"] == STOP_LOSS

    def test_independent_of_hpo_num_epochs(self):
        # The cap is no longer a function of epochs — the trainer bounds
        # training, not RunConfig.stop. See plan squiggly-bouncing-noether.md.
        d1 = _stop_dict_for_hpo({"hpo_num_epochs": 1})
        d2 = _stop_dict_for_hpo({"hpo_num_epochs": 100})
        assert d1 == d2

    def test_no_training_iteration_key(self):
        # Pin the bug-fix invariant: training_iteration must NOT cap the
        # trial. Otherwise the per-epoch report at epoch N races with the
        # driver's terminal tune.report(result) and kills the trial before
        # done=True can be set.
        d = _stop_dict_for_hpo({"hpo_num_epochs": 5})
        assert "training_iteration" not in d


class TestFinalStopDict:
    def test_only_contains_loss_key(self):
        d = _stop_dict_for_final({"num_train_epochs": 10})
        assert set(d.keys()) == {"loss"}

    def test_uses_stop_loss(self):
        d = _stop_dict_for_final({"num_train_epochs": 10})
        assert d["loss"] == STOP_LOSS

    def test_independent_of_num_train_epochs(self):
        d1 = _stop_dict_for_final({"num_train_epochs": 1})
        d2 = _stop_dict_for_final({"num_train_epochs": 100})
        assert d1 == d2

    def test_no_training_iteration_key(self):
        d = _stop_dict_for_final({"num_train_epochs": 10})
        assert "training_iteration" not in d


class TestHistoricalRegression:
    """Pin the bug-fix invariants from the timeline:

    1. training_iteration must not be in the stop dict (Option B). Pre-fix
       this caused trial death before the driver's `done=True` terminal
       report could fire.
    2. STOP_LOSS must be preserved — it's the converged-trial safety net.
    """

    def test_hpo_no_training_iteration(self):
        assert "training_iteration" not in _stop_dict_for_hpo({"hpo_num_epochs": 5})

    def test_final_no_training_iteration(self):
        assert "training_iteration" not in _stop_dict_for_final({"num_train_epochs": 10})

    def test_hpo_keeps_stop_loss(self):
        assert _stop_dict_for_hpo({})["loss"] == STOP_LOSS

    def test_final_keeps_stop_loss(self):
        assert _stop_dict_for_final({})["loss"] == STOP_LOSS


class TestAutoDeriveAshaMaxT:
    """Auto-derivation of asha_max_t from training_config['hpo_num_epochs']."""

    def test_sets_max_t_when_scheduler_is_asha(self):
        tune_config = {"scheduler": "asha"}
        training_config = {"hpo_num_epochs": 5}
        _auto_derive_asha_max_t(tune_config, training_config)
        assert tune_config["asha_max_t"] == 5

    def test_no_op_when_scheduler_is_fifo(self):
        tune_config = {"scheduler": "fifo"}
        training_config = {"hpo_num_epochs": 5}
        _auto_derive_asha_max_t(tune_config, training_config)
        assert "asha_max_t" not in tune_config

    def test_no_op_when_scheduler_is_hyperbandforbohb(self):
        tune_config = {"scheduler": "hyperbandforbohb"}
        training_config = {"hpo_num_epochs": 5}
        _auto_derive_asha_max_t(tune_config, training_config)
        assert "asha_max_t" not in tune_config

    def test_no_op_when_scheduler_missing(self):
        tune_config = {}
        training_config = {"hpo_num_epochs": 5}
        _auto_derive_asha_max_t(tune_config, training_config)
        assert "asha_max_t" not in tune_config

    def test_does_not_overwrite_explicit_value(self):
        # Defensive: if asha_max_t is already set (e.g., a future YAML
        # override path is reintroduced), respect the explicit value.
        tune_config = {"scheduler": "asha", "asha_max_t": 99}
        training_config = {"hpo_num_epochs": 5}
        _auto_derive_asha_max_t(tune_config, training_config)
        assert tune_config["asha_max_t"] == 99

    def test_overrides_when_existing_value_is_none(self):
        # null in YAML loads as None; the helper should treat that as "not set".
        tune_config = {"scheduler": "asha", "asha_max_t": None}
        training_config = {"hpo_num_epochs": 5}
        _auto_derive_asha_max_t(tune_config, training_config)
        assert tune_config["asha_max_t"] == 5

    def test_floors_at_one_when_hpo_num_epochs_missing(self):
        tune_config = {"scheduler": "asha"}
        training_config = {}
        _auto_derive_asha_max_t(tune_config, training_config)
        assert tune_config["asha_max_t"] == 1

    def test_floors_at_one_when_hpo_num_epochs_zero(self):
        tune_config = {"scheduler": "asha"}
        training_config = {"hpo_num_epochs": 0}
        _auto_derive_asha_max_t(tune_config, training_config)
        assert tune_config["asha_max_t"] == 1

    def test_coerces_floats(self):
        tune_config = {"scheduler": "asha"}
        training_config = {"hpo_num_epochs": 3.0}
        _auto_derive_asha_max_t(tune_config, training_config)
        assert tune_config["asha_max_t"] == 3
        assert isinstance(tune_config["asha_max_t"], int)

    def test_alignment_with_trainer_epoch_budget(self):
        # The whole point: asha_max_t must equal the trial's actual epoch
        # budget (HF Trainer's num_train_epochs, set to hpo_num_epochs
        # during HPO) so ASHA's bracket math knows the maximum number of
        # rungs available. Post-Option-B the controller no longer carries
        # a training_iteration cap; the alignment is solely with the
        # trainer's epoch budget.
        tune_config = {"scheduler": "asha"}
        training_config = {"hpo_num_epochs": 7}
        _auto_derive_asha_max_t(tune_config, training_config)
        assert tune_config["asha_max_t"] == training_config["hpo_num_epochs"]
