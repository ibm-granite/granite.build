"""Render a recipe target's step config through the PUBLISHED step template.

A recipe that calls a step owns its wiring: which values reach which config keys. A
string check on the recipe cannot see a value that lands in the wrong key, or a list
the template does not iterate. Rendering the target's config through the same template
the server will use, and running the result, can.

Two template passes, as on the server: the step config is filled first, and the
launcher's ``run`` is then filled from the merged config and the same ``bindings``
(a step reads its declared inputs there, as ``{{ bindings.<name>.binding.path }}``).
"""

import importlib.util
import pathlib

import yaml

from gbserver.utils.template import fill_objtemplate, fill_template

_ASSETS = (
    pathlib.Path(__file__).resolve().parents[3]
    / "configurations"
    / "assets"
    / "environments"
    / "skypilot"
    / "steps"
)


def _merge(base, over):
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def render_run(step_path, step_config, *, bindings=None, **config_overrides):
    """Return ``(run_script, step_dir)`` for ``space://steps/<step_path>``.

    :param step_path: the step's path under ``steps/``, e.g. ``distill/gen-smoke``.
    :param step_config: the target's ``steps[0].config`` from the rendered recipe.
    :param bindings: ``{name: path}`` for any ``{{ bindings.<name>.binding.path }}``.
        A value that is a dict is used as the binding itself, so a ``mem://`` input is
        given as ``{name: {"state": value}}``.
    :param config_overrides: per-section overrides applied last (e.g. a stand-in
        ``python``), as ``{section: {key: value}}``.
    :returns: the filled run script, and the published step dir to run it from (its
        ``./src`` is what the step's file_mount ships).
    """
    step_dir = _ASSETS / step_path
    step = yaml.safe_load((step_dir / "step.yaml").read_text(encoding="utf-8"))
    data = {
        "bindings": {
            k: {"binding": v if isinstance(v, dict) else {"path": v}}
            for k, v in (bindings or {}).items()
        }
    }
    config = fill_objtemplate(step_config, data)
    config = _merge(_merge(step["config"], config), config_overrides)
    launchers = step["environment_configs"]["Skypilot"]["launchers"]
    (launcher,) = launchers.values()
    return (
        fill_template(templ=launcher["config"]["run"], data={"config": config, **data}),
        step_dir,
    )


def gen_smoke_module():
    """The PUBLISHED gen-smoke detector, imported from the asset the server ships.

    Its torch/transformers imports are inside the generation function, so importing it
    needs neither.
    """
    spec = importlib.util.spec_from_file_location(
        "published_gen_smoke",
        _ASSETS / "distill" / "gen-smoke" / "src" / "gen_smoke.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
