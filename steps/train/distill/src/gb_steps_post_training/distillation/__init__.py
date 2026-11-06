"""Distillation prep steps: tokenizer retag, corpus prep, launchers."""

# GOLD distillation source, ported out of an earlier exploratory checkout.
#
# gold.py and run_vllm_serve.py are ENTRYPOINTS run as scripts, not imported as
# submodules of this package: `python gold.py` puts this directory on sys.path[0], which
# is what makes their flat sibling imports (custom_gold_trainer, custom_gold_config,
# utils, liger_losses, _swa_arm) resolve. Do not "fix" those into relative imports
# without also changing the reference launchers, which cd into this directory and invoke
# the files by name.
#
# retag_student.py is the tokenizer-alignment step and IS importable as a module.
