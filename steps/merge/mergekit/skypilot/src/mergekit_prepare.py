# Copyright Granite.Build Authors
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

"""Turn the step's config into a MergeKit recipe file and mergekit-yaml flags.

Called by run.sh, before the merge. Two jobs:

* Input names: every model reference in ``merge_config`` whose ``<model>`` or
  ``<lora>`` part (MergeKit syntax ``<model>[@<revision>][+<lora>]``) is the name of a
  BOUND input is replaced with that input's staged path. Anything else, a Hugging Face
  repo id, a node path, or the name of an unbound input, is left exactly as MergeKit
  reads it. The fields walked are every field MergeKit 0.1.5 parses as a model
  reference (``mergekit/config.py``, ``mergekit/tokenizer/config.py``), in both the
  string form and MergeKit's mapping form (``{model: <path or {path, revision}>,
  lora: ..., override_architecture: ...}``, ``mergekit/common.py`` ``ModelReference``).
* Flags: ``args`` is rendered dpk-style but for MergeKit's click options: ``true`` ->
  ``--k``, ``false`` -> ``--no-k``, ``null`` -> omitted, any other value -> ``--k v``.

It needs only the standard library and PyYAML (a mergekit dependency), so it runs in
the step's venv or in a ``mk_image``. It writes the recipe with ``yaml.safe_dump``,
not JSON: PyYAML reads YAML 1.1, where JSON's ``5e-06`` is a string, not a float.

Output on stdout: one mergekit-yaml flag per line, which run.sh reads into an array.
Values are refused when empty or when they hold a newline, so that every line is one
non-empty argv item and no flag can shift onto the next one's value.
"""

import argparse
import copy
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

INPUT_NAMES = ("model_1", "model_2", "model_3", "model_4", "base_model")

# Flags the step sets itself, or exposes as named keys; refused in args.
OWNED_FLAGS = {
    "transformers-cache": "the step sets it (a cache on the environment's shared root)",
    "lora-merge-cache": "the step sets it under the run's workdir",
    "cuda": "the step sets the device from the GPUs allocated",
    "device": "the step sets the device from the GPUs allocated",
    "gpu-rich": "the step sets the device from the GPUs allocated",
    "allow-crimes": "use mergekit_config.allow_crimes",
    "trust-remote-code": "use mergekit_config.trust_remote_code",
    "copy-tokenizer": "use mergekit_config.copy_tokenizer",
    "lazy-unpickle": "use mergekit_config.lazy_unpickle",
    "safe-serialization": (
        "the step checks the merged model's .safetensors files, so the output must be"
        " safetensors (MergeKit's default)"
    ),
}

# Values MergeKit accepts in tokenizer_source / tokenizer.source that are not models.
_TOKENIZER_KEYWORDS = ("union", "base")


class PrepareError(Exception):
    """A config problem, reported as one ``mergekit: ERROR`` line."""


def unbound_names(config: Dict[str, Any], bound: Dict[str, str]) -> List[str]:
    """Input names a recipe references that are not bound, in reference order.

    A bare ``model_1``..``model_4`` / ``base_model`` that isn't bound would otherwise
    reach MergeKit unchanged and be fetched as a Hugging Face repo of that name, failing
    only after provisioning with an error that never mentions the binding. None of the
    five is a realistic repo id. Walks the same fields, and splits the same way
    (``@``, and ``+`` in the string form), as the substitution.

    Args:
        config: the parsed ``merge_config`` mapping (not modified).
        bound: input name -> staged path, for bound inputs only.

    Returns:
        The unbound input names, each once.
    """
    return [n for n in referenced_names(config) if n not in bound]


def referenced_names(
    config: Dict[str, Any], bound: Optional[Dict[str, str]] = None
) -> List[str]:
    """Input names a recipe references, bound or not, in reference order.

    Args:
        config: the parsed ``merge_config`` mapping (not modified).
        bound: input name -> staged path. A reference that is a bound input's staged
            path (what ``{{ bindings.<name>.binding.path }}`` renders to) counts as
            naming that input.

    Returns:
        Each of ``INPUT_NAMES`` the recipe names in a model reference, once.
    """
    found: List[str] = []
    by_path = {_real(p): n for n, p in (bound or {}).items()}

    def seen(name: str) -> None:
        name = by_path.get(_real(name), name) if name.startswith("/") else name
        if name in INPUT_NAMES and name not in found:
            found.append(name)

    def as_ref(ref: str) -> str:
        model, _, lora = ref.partition("+")
        for part in (model, lora):
            seen(part.partition("@")[0])
        return ref

    def as_path(p: str) -> str:
        seen(p)
        return p

    walk_refs(copy.deepcopy(config), as_ref, as_path)
    return found


def _real(path: str) -> str:
    """A path in a comparable form (symlinks and ``..`` resolved, no trailing ``/``)."""
    return os.path.realpath(path)


def unfilled_refs(config: Dict[str, Any]) -> List[str]:
    """Model references still holding a ``{{ ... }}`` expression.

    gbserver's non-strict config fill writes an expression it can't resolve back as text,
    e.g. ``{{ bindings.model_2.binding.path }}`` when model_2 isn't bound arrives as
    ``{{ model_2.binding.path }}``, which MergeKit would try to fetch from the Hub.

    Args:
        config: the parsed ``merge_config`` mapping (not modified).

    Returns:
        Each such reference, once, in reference order.
    """
    found: List[str] = []

    def seen(ref: str) -> str:
        if "{{" in ref and ref not in found:
            found.append(ref)
        return ref

    walk_refs(copy.deepcopy(config), seen, seen)
    return found


def substitute_ref(ref: str, bound: Dict[str, str]) -> str:
    """Replace the input-name parts of one MergeKit model reference.

    Args:
        ref: a reference in MergeKit syntax, ``<model>[@<revision>][+<lora>]``.
        bound: input name -> staged path, for bound inputs only.

    Returns:
        The reference with each ``<model>`` / ``<lora>`` part that names a bound input
        replaced by its path; its ``@<revision>`` is kept.
    """

    def part(text: str) -> str:
        path, at, revision = text.partition("@")
        return bound.get(path, path) + at + revision

    model, plus, lora = ref.partition("+")
    return part(model) + (plus + part(lora) if plus else "")


def walk_refs(
    config: Dict[str, Any],
    fix: Callable[[str], str],
    fix_path: Callable[[str], str],
) -> Dict[str, Any]:
    """Rewrite every model reference in a MergeKit config, in place.

    Args:
        config: the parsed ``merge_config`` mapping.
        fix: maps one reference STRING (``<model>[@<rev>][+<lora>]``, which MergeKit
            parses) to its replacement.
        fix_path: maps the ``path`` of a mapping-form ModelPath (``{path, revision}``),
            which MergeKit takes literally: no ``@`` or ``+`` parsing, so the whole
            value is matched and ``revision`` is never touched.

    Returns:
        The same mapping, for convenience.
    """

    def path(v: Any) -> Any:
        """A ModelPath inside a mapping-form reference.

        MergeKit's ``ModelPath`` splits a string on ``@`` only (``+`` is parsed by the
        string form of ``ModelReference``, not here), and takes ``{path, revision}``
        literally.
        """
        if isinstance(v, dict) and isinstance(v.get("path"), str):
            v["path"] = fix_path(v["path"])
        elif isinstance(v, str):
            p, at, rev = v.partition("@")
            return fix_path(p) + at + rev
        return v

    def key(d: Any, k: str) -> None:
        """Fix the ModelReference at d[k]: a string, or the mapping form."""
        if not isinstance(d, dict):
            return
        ref = d.get(k)
        if isinstance(ref, str):
            d[k] = fix(ref)
        elif isinstance(ref, dict) and "model" in ref:
            ref["model"] = path(ref["model"])
            if ref.get("lora") is not None:
                ref["lora"] = path(ref["lora"])

    def slices(items: Any) -> None:
        for s in items if isinstance(items, list) else []:
            key(s, "base_model")
            for src in s.get("sources") or [] if isinstance(s, dict) else []:
                key(src, "model")

    def models(items: Any) -> None:
        for m in items if isinstance(items, list) else []:
            key(m, "model")

    key(config, "base_model")
    if config.get("tokenizer_source") not in _TOKENIZER_KEYWORDS:
        key(config, "tokenizer_source")
    models(config.get("models"))
    slices(config.get("slices"))
    modules = config.get("modules")
    for module in modules.values() if isinstance(modules, dict) else []:
        if isinstance(module, dict):
            models(module.get("models"))
            slices(module.get("slices"))
    tok = config.get("tokenizer")
    if isinstance(tok, dict):
        if tok.get("source") not in _TOKENIZER_KEYWORDS:
            key(tok, "source")
        for emb in (tok.get("tokens") or {}).values():
            if isinstance(emb, dict):
                key(emb, "source")
                key(emb.get("source"), "model")
    return config


def render_args(args: Dict[str, Any]) -> List[str]:
    """Render ``args`` as mergekit-yaml flags.

    Args:
        args: flag name (no dashes) -> value.

    Returns:
        The flags, one argv item each.

    Raises:
        PrepareError: a key is owned by the step, malformed, or a value is a list or
            mapping.
    """
    out: List[str] = []
    for k, v in args.items():
        k = str(k)
        if k in OWNED_FLAGS:
            raise PrepareError(f"args.{k} is not allowed: {OWNED_FLAGS[k]}.")
        if (
            not k
            or k.startswith("-")
            or k.startswith("no-")
            or not all(c.isascii() and (c.isalnum() or c == "-") for c in k)
        ):
            raise PrepareError(
                f"args key '{k}' must be a mergekit-yaml flag name without dashes "
                "(e.g. out-shard-size); write false to turn a flag off, not no-<flag>."
            )
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            raise PrepareError(
                f"args.{k} is empty; give it a value, or null to leave the flag out."
            )
        if v is True:
            out.append(f"--{k}")
        elif v is False:
            out.append(f"--no-{k}")
        elif isinstance(v, (list, dict)):
            raise PrepareError(f"args.{k} must be a scalar, got a list or mapping.")
        else:
            out += [f"--{k}", str(v)]
    return out


# chat_template names MergeKit 0.1.5 ships (mergekit/_data/chat_templates/*.jinja),
# for when mergekit can't be imported (a direct run of this script).
# test_chat_template_names_match_mergekit keeps it in sync.
FALLBACK_TEMPLATE_NAMES = ("alpaca", "chatml", "exaone", "llama3", "mistral")


def chat_template_names() -> List[str]:
    """The chat_template names MergeKit accepts: ``auto`` plus its shipped templates.

    Read from the installed mergekit (the step venv or an ``mk_image``) when it can be
    imported, as MergeKit's own lookup does; else the pinned list above.

    Returns:
        ``auto`` followed by the template names, sorted.
    """
    try:
        # pylint: disable-next=import-outside-toplevel
        import importlib.resources

        # pylint: disable-next=import-outside-toplevel,import-error
        from mergekit._data import chat_templates  # type: ignore[import-not-found]

        names = sorted(
            f.name[: -len(".jinja")]
            for f in importlib.resources.files(chat_templates).iterdir()
            if f.name.endswith(".jinja")
        )
    # Any failure (not installed for this interpreter, a broken torch in an mk_image, a
    # layout this step doesn't know) falls back rather than stopping with a traceback.
    # The fallback stays strict on purpose: a template name it doesn't know may be one the
    # config fill mangled, which MergeKit might accept and ship.
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(
            "mergekit: mergekit's chat templates can't be read here"
            f" ({type(exc).__name__}: {exc}); using the chat_template names of mergekit"
            f" 0.1.5 ({', '.join(FALLBACK_TEMPLATE_NAMES)}).",
            file=sys.stderr,
        )
        names = list(FALLBACK_TEMPLATE_NAMES)
    return ["auto", *names]


def merge_method_names() -> List[str]:
    """The merge methods the installed mergekit registers (empty if not importable).

    The node's own mergekit is the authority here, so a method a newer mergekit adds is
    accepted; render time checks the pinned 0.1.5 list only when it installs 0.1.5.
    Empty when there is nothing usable to check against.
    """
    try:
        # pylint: disable-next=import-outside-toplevel,import-error
        from mergekit.merge_methods import registry  # type: ignore[import-not-found]

        if hasattr(registry, "registered_methods"):  # 0.1.5 and later
            return sorted(m.spec.name for m in registry.registered_methods())
        # 0.1.4 and earlier: a dict keyed by method name (mergekit_version and mk_image
        # allow those).
        return sorted(registry.REGISTERED_MERGE_METHODS)
    # A broken or unfamiliar mergekit (e.g. an mk_image whose torch import fails, or a
    # registry layout this step doesn't know) must not stop the merge here: MergeKit
    # validates merge_method itself.
    except Exception:  # pylint: disable=broad-exception-caught
        return []


def check_recipe_shape(merge_config: Dict[str, Any]) -> None:
    """MergeKit's own top-level rules, so a bad recipe fails before the merge starts.

    Raises:
        PrepareError: no merge_method, an unregistered one (when mergekit is importable),
            a models / slices / modules of the wrong type, or not exactly one of them.
    """
    method = merge_config.get("merge_method")
    if not isinstance(method, str) or not method:
        raise PrepareError(
            "merge_config.merge_method is required, as a non-empty string."
        )
    known = merge_method_names()
    if known and method not in known:
        raise PrepareError(
            f"merge_config.merge_method '{method}' is not one of the installed"
            f" mergekit's methods: {', '.join(known)}."
        )
    for k, kind in (("models", list), ("slices", list), ("modules", dict)):
        v = merge_config.get(k)
        if v and not isinstance(v, kind):
            raise PrepareError(
                f"merge_config.{k} must be a"
                f" {'list' if kind is list else 'mapping of module name to module'},"
                f" got {type(v).__name__}."
            )
    present = [k for k in ("models", "slices", "modules") if merge_config.get(k)]
    if len(present) != 1:
        raise PrepareError(
            "merge_config must have exactly one of models, slices or modules (MergeKit's"
            f" own rule); it has {', '.join(present) if present else 'none'}."
        )


def check_chat_template(value: Any) -> None:
    """Refuse a chat_template that MergeKit would reject or that gbserver mangled.

    gbserver renders every config string as Jinja before the step sees it, and its
    non-strict fill drops ``{% ... %}`` blocks while writing an unknown ``{{ var }}``
    back out as text. An inline template that wasn't wrapped in
    ``{% raw %}...{% endraw %}`` therefore arrives empty, as fragments such as
    ``'{{ bos_token }}{{ eos_token }}'``, or as plain text such as ``'ASSISTANT:'``.
    MergeKit would ship the fragments, and with ``copy_tokenizer`` it rejects plain text
    inside a ``try`` that logs "The merge was still successful", so the model would be
    registered with no tokenizer. This is an early, specific error for the common
    mistakes; run.sh's output check (tokenizer and template present) is what guarantees
    a model is never registered without them. The step accepts only:

    * null: no template (MergeKit's default);
    * a name in :func:`chat_template_names`: ``auto`` or one of MergeKit's shipped
      templates. This is MergeKit's own name test (``merge.py`` ``_set_chat_template``:
      a ``<name>.jinja`` it ships), so a made-up or mangled name can't pass;
    * a Jinja template that still has its ``{%`` blocks, which survive the fill only
      inside ``{% raw %}``. Every template MergeKit ships has them. A template made only
      of ``{{ }}`` expressions is refused, although MergeKit would take it, because
      after the fill it can't be told apart from a mangled one.

    Raises:
        PrepareError: anything else.
    """
    if value is None:
        return
    names = chat_template_names()
    hint = (
        f" Use a template name ({', '.join(names)}) or an inline Jinja template with"
        " {% %} blocks, wrapped in {% raw %}...{% endraw %} in the build.yaml (gbserver"
        " renders config strings as Jinja otherwise)."
    )
    if not isinstance(value, str) or not value.strip():
        raise PrepareError(
            f"merge_config.chat_template is empty or not a string.{hint}"
        )
    if value in names or "{%" in value:
        return
    shown = value if len(value) <= 60 else value[:57] + "..."
    raise PrepareError(
        f"merge_config.chat_template '{shown}' is not a template name MergeKit ships,"
        f" and not a Jinja template with {{% %}} blocks.{hint}"
    )


def recipe_builds_tokenizer(merge_config: Dict[str, Any]) -> bool:
    """Whether MergeKit BUILDS a tokenizer for this recipe (rather than copying one).

    MergeKit's own test (``plan.py``): ``tokenizer`` is not None, or ``tokenizer_source``
    is not None. An empty ``tokenizer: {}`` builds one (its ``source`` defaults to
    ``union``). A built tokenizer is saved, chat template included, outside the
    ``try`` that swallows copy errors.
    """
    return (
        merge_config.get("tokenizer") is not None
        or merge_config.get("tokenizer_source") is not None
    )


def bool_flag(name: str, value: str) -> str:
    """Render one of the step's named boolean keys as an on/off flag.

    Args:
        name: the flag, e.g. ``copy-tokenizer``.
        value: ``true`` or ``false``, as the step-template renders it.

    Returns:
        ``--<name>`` or ``--no-<name>``.

    Raises:
        PrepareError: any other value.
    """
    if value not in ("true", "false"):
        raise PrepareError(
            f"{name.replace('-', '_')} must be true or false, got '{value}'."
        )
    return f"--{name}" if value == "true" else f"--no-{name}"


def check_bindings(merge_config: Dict[str, Any], bound: Dict[str, str]) -> None:
    """Every input the recipe names is bound, and every bound input is named.

    An unbound name, or an unfilled ``{{ bindings... }}`` reference, would reach MergeKit
    as a Hub repo id. A bound input the recipe never names (by its input name, or by its
    staged path, as the bindings form renders) is most often a typo (``model: model1`` for ``model_1``), which would
    otherwise also go to the Hub as a repo id, or be merged without that input while the
    build's lineage still lists it.

    Args:
        merge_config: the parsed ``merge_config`` mapping.
        bound: input name -> staged path, for bound inputs only.

    Raises:
        PrepareError: either mismatch.
    """
    unfilled = unfilled_refs(merge_config)
    if unfilled:
        raise PrepareError(
            f"merge_config has a model reference gbserver could not fill:"
            f" '{unfilled[0]}'. A {{{{ bindings.<name>.binding.path }}}} reference"
            " arrives like this when that input is not bound. Bind the input, or name it"
            " by its input name (e.g. `model: model_1`), which the step checks."
        )
    named = referenced_names(merge_config, bound)
    missing = [n for n in named if n not in bound]
    if missing:
        raise PrepareError(
            f"merge_config names input(s) {', '.join(missing)}, but"
            f" {'it is' if len(missing) == 1 else 'they are'} not bound. Bind"
            f" {'it' if len(missing) == 1 else 'them'} under the target's inputs, or"
            " name the model by its Hugging Face id or path instead."
        )
    unused = [n for n in INPUT_NAMES if n in bound and n not in named]
    if unused:
        raise PrepareError(
            f"input(s) {', '.join(unused)} {'is' if len(unused) == 1 else 'are'} bound,"
            " but merge_config never names"
            f" {'it' if len(unused) == 1 else 'them'}. Name each bound input in the"
            " recipe (e.g. `model: model_1`), or remove it from the target's inputs."
        )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--merge-config-json", required=True)
    p.add_argument("--args-json", required=True)
    p.add_argument("--recipe-out", required=True)
    p.add_argument(
        "--expect-out",
        required=True,
        help="file to write what the output must contain: 'tokenizer' and/or"
        " 'chat_template', one per line",
    )
    for name in INPUT_NAMES:
        p.add_argument(f"--input-{name}", default="")
    for name in BOOL_FLAGS:
        p.add_argument(f"--{name}", required=True)
    return p


# The step's named boolean keys, as mergekit-yaml on/off flags.
BOOL_FLAGS = ("copy-tokenizer", "lazy-unpickle", "allow-crimes", "trust-remote-code")


def check_tokenizer_for_template(
    merge_config: Dict[str, Any], copy_tokenizer: str
) -> None:
    """A chat_template needs a tokenizer to go in.

    With no tokenizer_source / tokenizer block MergeKit builds no tokenizer, and without
    --copy-tokenizer it only warns "Chat template will not be saved". With either key it
    builds one and saves it with the template (run.sh still checks the template landed:
    auto can set none).

    Raises:
        PrepareError: a chat_template with no tokenizer to put it in.
    """
    if (
        merge_config.get("chat_template") is not None
        and copy_tokenizer == "false"
        and not recipe_builds_tokenizer(merge_config)
    ):
        raise PrepareError(
            "merge_config.chat_template needs copy_tokenizer: true (or a"
            " tokenizer_source / tokenizer block in merge_config); otherwise the"
            " merged model has no tokenizer, so MergeKit drops the template."
        )


def expected_contents(merge_config: Dict[str, Any], copy_tokenizer: str) -> List[str]:
    """What the merged model must contain, so run.sh checks the result, not MergeKit.

    Returns:
        ``tokenizer`` whenever MergeKit builds or copies one, and ``chat_template``
        whenever one was asked for (auto can silently set none).
    """
    expect = []
    if recipe_builds_tokenizer(merge_config) or copy_tokenizer == "true":
        expect.append("tokenizer")
    if merge_config.get("chat_template") is not None:
        expect.append("chat_template")
    return expect


def prepare(a: argparse.Namespace) -> Tuple[Dict[str, Any], List[str]]:
    """Check the config and build the recipe and the flags.

    Args:
        a: the parsed command line.

    Returns:
        The recipe, with input names replaced by staged paths, and the mergekit-yaml
        flags.

    Raises:
        PrepareError, json.JSONDecodeError: a config problem.
    """
    merge_config = json.loads(a.merge_config_json)
    args = json.loads(a.args_json)
    if not isinstance(merge_config, dict) or not merge_config:
        raise PrepareError("mergekit_config.merge_config is required (the recipe).")
    check_recipe_shape(merge_config)
    if not isinstance(args, dict):
        raise PrepareError("mergekit_config.args must be a mapping.")
    bound = {
        n: getattr(a, f"input_{n}") for n in INPUT_NAMES if getattr(a, f"input_{n}")
    }
    check_bindings(merge_config, bound)
    walk_refs(
        merge_config,
        lambda ref: substitute_ref(ref, bound),
        lambda p: bound.get(p, p),
    )
    check_chat_template(merge_config.get("chat_template"))
    check_tokenizer_for_template(merge_config, a.copy_tokenizer)
    flags = [
        bool_flag(name, getattr(a, name.replace("-", "_"))) for name in BOOL_FLAGS
    ] + render_args(args)
    if any("\n" in f or "\0" in f for f in flags):
        raise PrepareError("args values must not contain a newline.")
    return merge_config, flags


def main(argv: List[str]) -> int:
    """Write the recipe and the expect file, and print the flags.

    Args:
        argv: the command line, without the program name.

    Returns:
        The process exit code: 0, or 1 after one ``mergekit: ERROR`` line on stderr.
    """
    a = _parser().parse_args(argv)
    try:
        merge_config, flags = prepare(a)
    except (PrepareError, json.JSONDecodeError) as exc:
        print(f"mergekit: ERROR {exc}", file=sys.stderr)
        return 1
    with open(a.recipe_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(merge_config, f, sort_keys=False)
    with open(a.expect_out, "w", encoding="utf-8") as f:
        f.write(
            "".join(f"{e}\n" for e in expected_contents(merge_config, a.copy_tokenizer))
        )
    print("\n".join(flags))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
