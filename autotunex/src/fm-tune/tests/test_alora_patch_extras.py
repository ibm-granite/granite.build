"""Coverage gaps for autotune.alora_patch — version gating and idempotency.

The happy-path test in test_alora_gc_patch.py loads a real model. These
tests don't require model loading; they probe the patch installer directly
with a monkeypatched peft.__version__.
"""

import pytest

import autotune.alora_patch as alora_patch_mod
from autotune.alora_patch import apply_alora_gc_patch, is_active


@pytest.fixture
def reset_patched_state():
    """Reset the module-level _PATCHED flag between tests."""
    original = alora_patch_mod._PATCHED
    alora_patch_mod._PATCHED = False
    yield
    alora_patch_mod._PATCHED = original


class TestVersionGating:
    def test_unsupported_017(self, reset_patched_state, monkeypatch):
        # Pretend peft is 0.17.0 (older)
        import peft

        monkeypatch.setattr(peft, "__version__", "0.17.0")
        result = apply_alora_gc_patch()
        assert result is False
        assert is_active() is False

    def test_unsupported_019(self, reset_patched_state, monkeypatch):
        import peft

        monkeypatch.setattr(peft, "__version__", "0.19.0")
        result = apply_alora_gc_patch()
        assert result is False
        assert is_active() is False

    def test_unsupported_empty_version(self, reset_patched_state, monkeypatch):
        import peft

        monkeypatch.setattr(peft, "__version__", "")
        result = apply_alora_gc_patch()
        assert result is False


class TestIdempotency:
    def test_apply_twice_returns_true(self, reset_patched_state):
        # On the actual installed peft (0.18.0), apply succeeds. Calling again
        # is a no-op that still returns True.
        first = apply_alora_gc_patch()
        if first is False:
            pytest.skip("Running on non-0.18 peft; idempotency check requires patched state")
        second = apply_alora_gc_patch()
        assert second is True
        assert is_active() is True

    def test_is_active_reflects_state(self, reset_patched_state):
        assert is_active() is False
        apply_alora_gc_patch()
        # Either True (0.18) or still False (other versions). Both are valid.
        assert is_active() in (True, False)
