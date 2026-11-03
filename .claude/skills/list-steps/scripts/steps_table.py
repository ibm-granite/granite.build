"""Generate inventory tables (one per category) of the step implementations under steps/.

Run from the repo root:  python3 .claude/skills/list-steps/scripts/steps_table.py --help

One row per step path (steps/<group>/<name>/), combining all of its environment
implementations (steps/<group>/<name>/<env>/). Only git-tracked steps are listed.
Output formats: Markdown (default), Excel (.xlsx) and PowerPoint (.pptx).
Notes needing human judgment are written to stderr.
"""

import argparse
import glob
import os
import re
import subprocess
import sys

import yaml

from cells import Code, Link, part_text

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

CONFIG_ENV_ROOT = "configurations/assets/environments/"
CONFIG_STEPS_ROOT = "configurations/assets/steps/"
ENV_URI_PREFIX = "space://environments/"
BARE = "none (bare node)"
HELM = "Helm chart"
NONE = "—"
NON_FRAMEWORK_TITLE = "Non-Framework Steps"
FORMATS = {"markdown": "md", "xlsx": "xlsx", "pptx": "pptx"}


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


def tracked_config_steps():
    """Return git-tracked step.yaml paths anywhere under configurations/.

    Returns:
        Sorted list of repo-relative paths.

    Raises:
        subprocess.CalledProcessError: If git fails (not run inside the repo).
    """
    out = subprocess.run(
        ["git", "ls-files", "configurations/**/step.yaml"],
        check=True, capture_output=True, text=True,
    ).stdout
    return sorted(out.splitlines())


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
    """Label each environment class a template declares, with its subtypes when restricted.

    Args:
        template: Parsed step-template.yaml.

    Returns:
        List of labels, one per environment_configs entry: the class lowercased, plus
        ' - <subtype>,<subtype>' when it has a `subtypes:` list (e.g. ['skypilot - lsf'],
        ['skypilot'], ['bash']).
    """
    envs = []
    for env_type, cfg in (template.get("environment_configs") or {}).items():
        subtypes = (cfg or {}).get("subtypes") or []
        envs.append(env_type.lower() + (f" - {','.join(subtypes)}" if subtypes else ""))
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
    """Return the Image cell parts for one implementation: public image, bare node, or [] if custom.

    Args:
        impl_dir: The implementation directory (steps/<group>/<name>/<env>).
        template: Parsed step-template.yaml.

    Returns:
        List of parts, sorted: Code(image) per resolved image (launcher image_id, or image
        for docker/runpod), HELM for a Helm-chart launcher, BARE when a launcher has none,
        '?' when unresolvable; [] when the step builds its own image from a Dockerfile.
    """
    if os.path.exists(os.path.join(impl_dir, "Dockerfile")):
        return []
    images = set()
    for env_cfg in (template.get("environment_configs") or {}).values():
        for launcher in ((env_cfg or {}).get("launchers") or {}).values():
            config = launcher.get("config") or {}
            if "chart" in config:
                images.add(HELM)
                continue
            image = resolve_image_id(config.get("image_id") or config.get("image"), template)
            if image is None:
                warn(f"{impl_dir}: could not resolve image_id; fill the Image cell by hand")
                image = "?"
            images.add(Code(shorten_digest(image)) if image and image != "?" else image or BARE)
    return sorted(images, key=part_text) or [BARE]


def build_env_label(build_yaml):
    """Return the environment path a test build targets, e.g. 'skypilot/slurm'.

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
    """Turn an environment path into a table label ('slurm (local)', 'slurm (bluevela)', 'aws', 'bash').

    When a subtype is tested on several clusters, each label names its cluster; the base
    subtype environment (e.g. 'skypilot/slurm') is the local cluster, labelled '(local)'.

    Args:
        env_path: e.g. 'skypilot/slurm' or 'skypilot/slurm/bluevela'.
        multi_cluster: Subtypes seen with more than one cluster; only these get a '(cluster)'.

    Returns:
        Short label.
    """
    parts = env_path.split("/")
    if parts[0] != "skypilot" or len(parts) < 2:
        return parts[0]
    sub = parts[1]
    if sub not in multi_cluster:
        return sub
    return f"{sub} ({'/'.join(parts[2:]) or 'local'})"


def sample_rank(env_path, case):
    """Sort key preferring local SLURM (base 'skypilot/slurm'), then AWS, then the rest.

    Ties go to the shorter case name, then alphabetical.

    Args:
        env_path: Environment path of the case.
        case: Case directory name.

    Returns:
        A sortable tuple.
    """
    order = ["skypilot/slurm", "skypilot/aws"]
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


def config_step_location(step_dir):
    """Split a configurations/ step directory into its step path and environment label.

    Steps under configurations/assets/environments/<class>/[<sub>/[<cluster>/]]steps/<path>
    are visible only in that environment, so the directory implies the subtype. Steps under
    configurations/assets/steps/<class>/<path> nest the class first.

    Args:
        step_dir: Directory holding the step.yaml.

    Returns:
        (step path, environment label), e.g. ('bfcl-eval', 'skypilot - lsf (ibm-bluevela)').
    """
    if step_dir.startswith(CONFIG_STEPS_ROOT):
        env_class, _, step_path = step_dir[len(CONFIG_STEPS_ROOT):].partition("/")
        return step_path, env_class
    env_dir, _, step_path = step_dir[len(CONFIG_ENV_ROOT):].partition("/steps/")
    env_class, *rest = env_dir.split("/")
    if not rest:
        return step_path, env_class
    cluster = f" ({'/'.join(rest[1:])})" if len(rest) > 1 else ""
    return step_path, f"{env_class} - {rest[0]}{cluster}"


def collect_config_impl(step_yaml):
    """Gather table facts for a step that exists only under configurations/.

    Args:
        step_yaml: configurations/.../steps/<path>/step.yaml.

    Returns:
        Dict shaped like collect_impl's, with framework False and no test cases.

    Raises:
        yaml.YAMLError: If the step.yaml does not parse.
    """
    step_dir = os.path.dirname(step_yaml)
    step_path, env = config_step_location(step_dir)
    with open(step_yaml) as f:
        step = yaml.safe_load(f)
    declared = declared_environments(step)
    envs = declared if any(" - " in e for e in declared) or " - " not in env else [env]
    readme = os.path.join(step_dir, "README.md")
    return {
        "step_path": step_path, "env": env, "envs": envs, "image": image_for(step_dir, step),
        "cases": {}, "readme": readme if os.path.exists(readme) else None,
        "published": True, "framework": False,
    }


def published_dir(impl):
    """Return the configurations/ directory a framework implementation publishes to.

    Args:
        impl: A collect_impl dict.

    Returns:
        Repo-relative directory path.
    """
    return f"{CONFIG_ENV_ROOT}{impl['env']}/steps/{impl['step_path']}"


def collect_impl(template_path):
    """Gather the per-implementation facts the table needs.

    Args:
        template_path: steps/<group>/<name>/<env>/step-template.yaml.

    Returns:
        Dict with step_path, env, envs, image (cell parts), cases, readme, published,
        framework (True: built from the steps/ tree).

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
        "readme": readme, "published": published, "framework": True,
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
            if parts[0] == "skypilot" and len(parts) > 1:
                clusters.setdefault(parts[1], set()).add("/".join(parts[2:]))
    return {sub for sub, names in clusters.items() if len(names) > 1}


def joined(parts, empty=NONE):
    """Interleave parts with ', ' separators.

    Args:
        parts: List of parts (str, Code, Link).
        empty: Cell to return when parts is empty.

    Returns:
        List of parts forming one cell.
    """
    if not parts:
        return [empty] if empty else []
    out = [parts[0]]
    for part in parts[1:]:
        out += [", ", part]
    return out


def build_row(step_path, impls, multi_cluster):
    """Build one table row (a list of cells) for a step from all of its implementations.

    Args:
        step_path: '<group>/<name>'.
        impls: collect_impl dicts for this step.
        multi_cluster: See multi_cluster_subtypes.

    Returns:
        List of cells, one per cells.COLUMNS entry.
    """
    multi = len(impls) > 1
    envs = list(dict.fromkeys(e for i in impls for e in i["envs"]))
    tested = sorted({env_label(env, multi_cluster) for i in impls for env, _ in i["cases"].values()})
    readmes = [Link(i["env"] if multi else "README.md", i["readme"]) for i in impls if i["readme"]]
    readme_cell = joined(readmes) + ([] if all(i["published"] for i in impls) else [" ‡"])
    images = list(dict.fromkeys(part for i in impls for part in i["image"]))
    samples = []
    for i in impls:
        if i["cases"]:
            case = min(i["cases"], key=lambda c: sample_rank(i["cases"][c][0], c))
            samples.append(Link(case, i["cases"][case][1]))
    return [[step_path], ["; ".join(envs) or NONE], joined(tested), readme_cell,
            joined(images, empty=""), joined(samples)]


def build_tables():
    """Collect every tracked step: framework steps by category, then the non-framework steps.

    Framework steps (steps/ tree) get one table per category. Non-framework steps (a
    configurations/ step.yaml that no framework step publishes) all go in one final table
    titled NON_FRAMEWORK_TITLE, uncategorised.

    Returns:
        List of (title, rows): categories alphabetical, then NON_FRAMEWORK_TITLE if any;
        rows sorted by step path.

    Raises:
        subprocess.CalledProcessError: If git fails (not run inside the repo).
    """
    impls = [collect_impl(p) for p in tracked_templates()]
    published = {published_dir(i) for i in impls}
    impls += [collect_config_impl(p) for p in tracked_config_steps()
              if os.path.dirname(p) not in published]
    by_step = {}
    for impl in impls:
        by_step.setdefault((impl["step_path"], impl["framework"]), []).append(impl)
    multi_cluster = multi_cluster_subtypes(impls)
    by_category, non_framework = {}, []
    for step_path, framework in sorted(by_step):
        row = build_row(step_path, by_step[(step_path, framework)], multi_cluster)
        if framework:
            by_category.setdefault(category_for(step_path), []).append(row)
        else:
            non_framework.append(row)
    tables = sorted(by_category.items())
    return tables + ([(NON_FRAMEWORK_TITLE, non_framework)] if non_framework else [])


def default_link_base():
    """Derive a GitHub blob URL prefix for the default branch of the repo's main remote.

    Prefers the 'upstream' remote (the shared repo) over 'origin' (often a fork).

    Returns:
        e.g. 'https://github.com/ibm-granite/granite.build/blob/main/', or '' when no
        GitHub remote is found.
    """
    for remote in ("upstream", "origin"):
        res = subprocess.run(["git", "remote", "get-url", remote], capture_output=True, text=True)
        m = re.match(r"(?:git@github\.com:|https://github\.com/)(.+?)(?:\.git)?$", res.stdout.strip())
        if m:
            return f"https://github.com/{m.group(1)}/blob/main/"
    return ""


def parse_args(argv=None):
    """Parse the command line.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        argparse.Namespace with formats, output, link_base.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-f", "--format", dest="formats", action="append", choices=FORMATS,
                        help="output format; repeat for several (default: markdown)")
    parser.add_argument("-o", "--output", metavar="PATH",
                        help="output path without extension; the format's extension is added "
                             "(default: Markdown to stdout, others to ./steps-inventory.<ext>)")
    parser.add_argument("--link-base", metavar="URL",
                        help="prefix for repo-relative links (default: relative for Markdown, "
                             "the GitHub blob URL of the upstream/origin remote for xlsx/pptx)")
    args = parser.parse_args(argv)
    args.formats = list(dict.fromkeys(args.formats or ["markdown"]))
    return args


def write_format(fmt, tables, args):
    """Render the tables in one format to stdout or a file.

    Args:
        fmt: A FORMATS key.
        tables: Output of build_tables.
        args: Parsed command line.

    Raises:
        ImportError: If an optional renderer dependency (xlsxwriter) is missing.
    """
    link_base = args.link_base
    if link_base is None:
        link_base = "" if fmt == "markdown" else default_link_base()
    if fmt == "markdown":
        import render_md
        text = render_md.render(tables, link_base)
        if not args.output:
            sys.stdout.write(text)
            return
        path = f"{args.output}.md"
        with open(path, "w") as f:
            f.write(text)
    else:
        path = f"{args.output or 'steps-inventory'}.{FORMATS[fmt]}"
        renderer = __import__(f"render_{fmt}")
        renderer.write(tables, path, link_base)
    print(f"wrote {path}", file=sys.stderr)


def main():
    """Build the per-category tables and write them in each requested format."""
    args = parse_args()
    tables = build_tables()
    for fmt in args.formats:
        try:
            write_format(fmt, tables, args)
        except ImportError as e:
            req = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
            sys.exit(f"{fmt}: missing dependency ({e.name}); install with "
                     f"pip install --require-hashes -r {req}")


if __name__ == "__main__":
    main()
