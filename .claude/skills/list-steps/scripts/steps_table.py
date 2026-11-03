"""Generate a Markdown inventory table of the step implementations under steps/.

Run from the repo root:  python3 .claude/skills/list-steps/scripts/steps_table.py

One row per step path (steps/<group>/<name>/), combining all of its environment
implementations (steps/<group>/<name>/<env>/). Only git-tracked steps are listed.
Notes needing human judgment are written to stderr.
"""

import glob
import os
import re
import subprocess
import sys

import yaml

# Top-level steps/ group -> table category. Unlisted groups fall to "Other".
GROUP_CATEGORY = {"data": "Data", "eval": "Eval", "misc": "Misc", "train": "Train", "tune": "Train"}

# Per-step overrides for groups (like distill/) whose steps span categories.
STEP_CATEGORY = {
    "distill/corpus-pin-check": "Data",
    "distill/corpus-prep": "Data",
    "distill/corpus-sources": "Data",
    "distill/logit-precompute": "Data",
    "distill/eval": "Eval",
    "distill/gen-smoke": "Eval",
    "distill/probe": "Eval",
    "distill/hf-export": "Misc",
    "distill/vllm-server": "Other",
    "distill/gold": "Train",
    "distill/sft": "Train",
    "distill/tokenizer-align": "Train",
}

ENV_URI_PREFIX = "space://environments/"
BARE = "none (bare node)"


def warn(msg):
    """Write a review note to stderr.

    Args:
        msg: Note text.
    """
    print(f"NOTE: {msg}", file=sys.stderr)


def tracked_templates():
    """Return git-tracked step-template.yaml paths under steps/<group>/<name>/<env>/.

    Returns:
        Sorted list of repo-relative paths.

    Raises:
        subprocess.CalledProcessError: If git fails (not run inside the repo).
    """
    out = subprocess.run(
        ["git", "ls-files", "steps/*/*/*/step-template.yaml"],
        check=True, capture_output=True, text=True,
    ).stdout
    return sorted(p for p in out.splitlines() if p.count("/") == 4)


def category_for(step_path):
    """Map a step path (e.g. 'distill/gold') to its table category.

    Args:
        step_path: '<group>/<name>' relative to steps/.

    Returns:
        One of Data, Eval, Misc, Train, Other.
    """
    if step_path in STEP_CATEGORY:
        return STEP_CATEGORY[step_path]
    group = step_path.split("/")[0]
    if group not in GROUP_CATEGORY:
        warn(f"{step_path}: no category mapping; using Other — add it to STEP_CATEGORY")
    return GROUP_CATEGORY.get(group, "Other")


def declared_environments(template):
    """List environments a template declares: its subtypes, else the env type lowercased.

    Args:
        template: Parsed step-template.yaml.

    Returns:
        List of environment names (e.g. ['lsf'] or ['skypilot']).
    """
    envs = []
    for env_type, cfg in (template.get("environment_configs") or {}).items():
        envs.extend((cfg or {}).get("subtypes") or [env_type.lower()])
    return envs


def shorten_digest(image):
    """Abbreviate an '@sha256:<digest>' suffix to 8 hex chars for readability.

    Args:
        image: Image reference.

    Returns:
        The reference with a shortened digest, or unchanged.
    """
    return re.sub(r"@sha256:([0-9a-f]{8})[0-9a-f]+", r"@sha256:\1…", image)


def resolve_image_id(raw, template):
    """Resolve a launcher image_id, following a single '{{ ... config.X.Y ... }}' indirection.

    Args:
        raw: The launcher's image_id value (may be None, literal, or templated).
        template: Parsed step-template.yaml (for config lookups).

    Returns:
        Image reference without a 'docker:' prefix, '' for a bare node, or None if unresolvable.
    """
    raw = (raw or "").strip()
    if "{{" in raw:
        m = re.search(r"config\.(\w+)\.(\w+)", raw)
        if not m:
            return None
        raw = str(((template.get("config") or {}).get(m.group(1)) or {}).get(m.group(2)) or "")
    if "${" in raw:
        return None
    return raw.removeprefix("docker:")


def image_for(impl_dir, template):
    """Return the Image cell for one implementation: public image, bare node, or '' if custom.

    Args:
        impl_dir: The implementation directory (steps/<group>/<name>/<env>).
        template: Parsed step-template.yaml.

    Returns:
        Cell text; '' when the step builds its own image from a Dockerfile.
    """
    if os.path.exists(os.path.join(impl_dir, "Dockerfile")):
        return ""
    images = set()
    for env_cfg in (template.get("environment_configs") or {}).values():
        for launcher in ((env_cfg or {}).get("launchers") or {}).values():
            image = resolve_image_id((launcher.get("config") or {}).get("image_id"), template)
            if image is None:
                warn(f"{impl_dir}: could not resolve image_id; fill the Image cell by hand")
                image = "?"
            images.add(f"`{shorten_digest(image)}`" if image and image != "?" else image or BARE)
    return ", ".join(sorted(images)) or BARE


def build_env_label(build_yaml):
    """Return the environment path a test build targets, e.g. 'skypilot/slurm/local'.

    Args:
        build_yaml: Path to a test-data/<case>/build.yaml.

    Returns:
        Environment path with the space://environments/ prefix removed, or None.
    """
    with open(build_yaml) as f:
        text = f.read()
    for m in re.finditer(r"^\s*environment_uri:\s*(\S+)", text, re.M):
        if m.group(1).startswith(ENV_URI_PREFIX):
            return m.group(1)[len(ENV_URI_PREFIX):]
    return None


def tested_cases(impl_dir):
    """Map each per-cluster test case to its environment path.

    A case is test/<case>/ holding a test_*.py, paired with test-data/<case>/build.yaml.

    Args:
        impl_dir: The implementation directory.

    Returns:
        Dict of case name -> (environment path, build.yaml path).
    """
    cases = {}
    for test_py in sorted(glob.glob(os.path.join(impl_dir, "test", "*", "test_*.py"))):
        case = os.path.basename(os.path.dirname(test_py))
        build = os.path.join(impl_dir, "test-data", case, "build.yaml")
        env = build_env_label(build) if os.path.exists(build) else None
        if env is None:
            warn(f"{impl_dir}: test/{case} has no test-data/{case}/build.yaml environment_uri")
            continue
        cases[case] = (env, build)
    return cases


def env_label(env_path, multi_cluster):
    """Turn an environment path into a table label ('slurm (local)', 'aws', 'bash').

    Args:
        env_path: e.g. 'skypilot/slurm/local'.
        multi_cluster: Subtypes seen with more than one cluster; only these get a '(cluster)'.

    Returns:
        Short label.
    """
    parts = env_path.split("/")
    if parts[0] != "skypilot" or len(parts) < 2:
        return parts[0]
    sub = parts[1]
    return f"{sub} ({parts[2]})" if len(parts) > 2 and sub in multi_cluster else sub


def sample_rank(env_path, case):
    """Sort key preferring local SLURM, then AWS, then the rest; shorter case names first.

    Args:
        env_path: Environment path of the case.
        case: Case directory name.

    Returns:
        A sortable tuple.
    """
    order = ["skypilot/slurm/local", "skypilot/aws"]
    return (order.index(env_path) if env_path in order else len(order), len(case), case)


def readme_link(step_path, env):
    """Link the published README in configurations/, falling back to the source README.

    Args:
        step_path: '<group>/<name>'.
        env: Implementation env dir name (skypilot, bash).

    Returns:
        (path, published) — published is False when only the steps/ README exists.
    """
    published = f"configurations/assets/environments/{env}/steps/{step_path}/README.md"
    if os.path.exists(published):
        return published, True
    return f"steps/{step_path}/{env}/README.md", False


def collect_impl(template_path):
    """Gather the per-implementation facts the table needs.

    Args:
        template_path: steps/<group>/<name>/<env>/step-template.yaml.

    Returns:
        Dict with step_path, env, envs, image, cases, readme, published.

    Raises:
        yaml.YAMLError: If the template does not parse.
    """
    impl_dir = os.path.dirname(template_path)
    _, group, name, env = impl_dir.split("/")
    with open(template_path) as f:
        template = yaml.safe_load(f)
    step_path = f"{group}/{name}"
    readme, published = readme_link(step_path, env)
    return {
        "step_path": step_path, "env": env, "envs": declared_environments(template),
        "image": image_for(impl_dir, template), "cases": tested_cases(impl_dir),
        "readme": readme, "published": published,
    }


def multi_cluster_subtypes(impls):
    """Find skypilot subtypes tested on more than one cluster across all steps.

    Args:
        impls: Output of collect_impl for every implementation.

    Returns:
        Set of subtype names (e.g. {'slurm'}).
    """
    clusters = {}
    for impl in impls:
        for env_path, _ in impl["cases"].values():
            parts = env_path.split("/")
            if parts[0] == "skypilot" and len(parts) > 2:
                clusters.setdefault(parts[1], set()).add(parts[2])
    return {sub for sub, names in clusters.items() if len(names) > 1}


def links(items):
    """Render (label, path) pairs as comma-joined Markdown links.

    Args:
        items: List of (label, path).

    Returns:
        Comma-joined links, or '—' when empty.
    """
    return ", ".join(f"[{label}]({path})" for label, path in items) or "—"


def render_row(step_path, impls, multi_cluster):
    """Render one table row for a step from all of its implementations.

    Args:
        step_path: '<group>/<name>'.
        impls: collect_impl dicts for this step.
        multi_cluster: See multi_cluster_subtypes.

    Returns:
        A Markdown table row string.
    """
    multi = len(impls) > 1
    envs = list(dict.fromkeys(e for i in impls for e in i["envs"]))
    tested = list(dict.fromkeys(
        env_label(env, multi_cluster) for i in impls for env, _ in i["cases"].values()))
    readmes = [((i["env"] if multi else "README.md"), i["readme"]) for i in impls]
    readme_cell = links(readmes) + ("" if all(i["published"] for i in impls) else " ‡")
    images = list(dict.fromkeys(i["image"] for i in impls))
    samples = []
    for i in impls:
        if i["cases"]:
            case = min(i["cases"], key=lambda c: sample_rank(i["cases"][c][0], c))
            samples.append((case, i["cases"][case][1]))
    cells = [category_for(step_path), step_path, ", ".join(envs), ", ".join(tested) or "—",
             readme_cell, ", ".join(images), links(samples)]
    return "| " + " | ".join(cells) + " |"


def main():
    """Print the steps inventory table to stdout."""
    impls = [collect_impl(p) for p in tracked_templates()]
    by_step = {}
    for impl in impls:
        by_step.setdefault(impl["step_path"], []).append(impl)
    multi_cluster = multi_cluster_subtypes(impls)
    print("| Category | Step | Environments | Tested Environments | README | Image | Sample build |")
    print("|---|---|---|---|---|---|---|")
    for step_path in sorted(by_step, key=lambda s: (category_for(s), s)):
        print(render_row(step_path, by_step[step_path], multi_cluster))
    print()
    print("‡ Not published to `configurations/`; links the README under `steps/`.")
    print("Image: blank = custom image built from the step's Dockerfile.")


if __name__ == "__main__":
    main()
