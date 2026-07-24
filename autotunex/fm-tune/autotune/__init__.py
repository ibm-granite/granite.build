try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
except ImportError:
    raise ImportError(
        "autotune requires an install profile. Install with one of:\n"
        '  uv pip install -e ".[sft]"   # SFT + offline RL (DPO/KTO)\n'
        '  uv pip install -e ".[rl]"    # Online RL (PPO/GRPO/DAPO)\n'
    ) from None
