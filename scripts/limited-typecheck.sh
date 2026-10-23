#!/usr/bin/env bash
#
# Run pylint + mypy on the python files changed vs main, and only those.
#
# Not a full static check: files nobody touched are not linted, so their existing
# errors don't fail or bury the report, and every file gets checked the next time
# it changes. (`make staticcheck` / `make lint` are the whole-package checks.)
#
# Changed python files are grouped by how their imports resolve:
#   * steps/ files: each step (the directory holding step-template.yaml) is an
#     independent tree whose src/ modules are imported by flat name (the step's
#     test/conftest.py puts src/ on sys.path), and module names such as conftest.py
#     repeat across steps. So the changed files of each step are checked from that
#     step's root.
#   * configurations/ and test-data/ files: loose scripts (often bash_scripts/*/run.py)
#     whose module names repeat across directories, so each directory is checked
#     on its own.
#   * everything else (src/, test/, scripts/, ...): checked together from the repo
#     root with src/ and test/ on the path, matching pytest's `pythonpath`.
# autotunex/ is skipped: it has its own ruff config and CI.

MY_LINE_BREAK="---------------------------"
REPO_ROOT="$(git rev-parse --show-toplevel)"

# Print the step root (nearest ancestor dir holding step-template.yaml) of a
# repo-relative path, or nothing if the path is not inside a step.
#   $1 - repo-relative file path
step_root_of() {
    local d
    d="$(dirname "$1")"
    while [[ "${d}" == steps/* ]]; do
        if [[ -f "${d}/step-template.yaml" ]]; then
            echo "${d}"
            return
        fi
        d="$(dirname "${d}")"
    done
}

# Run pylint on some files, from the current directory.
#   $1    - import path (PYTHONPATH) for these files
#   $2    - pylint --disable list for these files ("" for none)
#   $3... - files to check, relative to the current directory
pylint_files() {
    local path="$1" disable="$2"
    shift 2
    [[ $# -gt 0 ]] || return 0
    PYTHONPATH="${path}" \
        pylint --rcfile="${REPO_ROOT}/.pylintrc" ${disable:+--disable="${disable}"} "$@"
}

# Run pylint + mypy on a group of files that share an import root, and print the
# results with paths rewritten to be repo-relative.
#   $1    - the group's root dir (repo-relative, "." for the repo root); the
#           checks run from there
#   $2    - header label for the group
#   $3    - extra import path entries, prepended to the repo's src/ and test/
#           ("" for none)
#   $4... - files to check, relative to the group root
# Imports of gbserver etc. (src/) and the test harness (test/) resolve like they
# do under pytest. Imports of runtime-only deps (torch, transformers, ...) usually
# aren't installed in the dev venv, so import-not-found is suppressed.
# --follow-imports=silent keeps errors in imported, unchanged files out of the
# report. Files under a test/ dir get redefined-outer-name disabled: every pytest
# fixture used as a test argument triggers it, burying real warnings (~2/3 of all
# messages in steps/).
check_group() {
    local root="$1" label="$2" extra="$3" f path
    local tests=() others=()
    shift 3
    path="${extra:+${extra}:}${REPO_ROOT}/src:${REPO_ROOT}/test"
    for f in "$@"; do
        if [[ "${f}" == test/* || "${f}" == */test/* ]]; then
            tests+=("${f}")
        else
            others+=("${f}")
        fi
    done
    echo "${MY_LINE_BREAK}"
    echo -e "\033[0;36m${label}\033[0m"
    (
        cd "${REPO_ROOT}/${root}" || exit 1
        pylint_files "${path}" "" "${others[@]+"${others[@]}"}"
        pylint_files "${path}" "redefined-outer-name" "${tests[@]+"${tests[@]}"}"
        MYPYPATH="${path}" \
            mypy --explicit-package-bases --follow-imports=silent \
            --disable-error-code=import-untyped \
            --disable-error-code=import-not-found "$@"
    ) 2>&1 | prefix_paths "${root}"
}

# Rewrite "<relative path>.py:" at the start of each stdin line to be relative to
# the repo root. Absolute paths are left alone.
#   $1 - dir the paths are relative to (repo-relative, "." for the repo root)
prefix_paths() {
    if [[ "$1" == "." ]]; then
        cat
    else
        sed -E "s#^([^/ :][^ :]*\.py:)#$1/\1#"
    fi
}

# Succeed if $1 is one of the remaining arguments.
#   $1    - value to look for
#   $2... - list to search
contains() {
    local want="$1" x
    shift
    for x in "$@"; do
        [[ "${x}" == "${want}" ]] && return 0
    done
    return 1
}

# Check changed python files in groups, one group per root dir, each from its
# root with the root's src/ (if any) on the import path.
#   $1    - "step" to group by step root (files outside any step fall back to
#           their own dir), or "dir" to group by each file's own dir
#   $2... - repo-relative existing changed python files
# Grouping by dir is for trees of loose scripts whose module names (e.g. run.py)
# repeat across directories, which mypy refuses in a single run.
check_grouped() {
    local mode="$1" x root roots=() files label
    shift
    for x in "$@"; do
        root="$(group_root_of "${mode}" "${x}")"
        contains "${root}" "${roots[@]+"${roots[@]}"}" || roots+=("${root}")
    done
    for root in "${roots[@]+"${roots[@]}"}"; do
        files=()
        for x in "$@"; do
            [[ "$(group_root_of "${mode}" "${x}")" == "${root}" ]] \
                && files+=("${x#"${root}"/}")
        done
        label="Dir: ${root}"
        [[ -f "${root}/step-template.yaml" ]] && label="Step: ${root}"
        check_group "${root}" "${label}" "src" "${files[@]}"
    done
}

# Print the group root check_grouped puts a file in.
#   $1 - "step" or "dir" (see check_grouped)
#   $2 - repo-relative file path
group_root_of() {
    local root=""
    [[ "$1" == step ]] && root="$(step_root_of "$2")"
    [[ -n "${root}" ]] || root="$(dirname "$2")"
    echo "${root}"
}

# Sort the changed files into groups by import root and check each group.
#   $@ - repo-relative changed file paths
# Returns 1 if there was no python file to check, else 0.
check_changed() {
    local x steps=() by_dir=() repo=()
    for x in "$@"; do
        # Skip paths that no longer exist on disk (deleted or renamed-away files
        # still appear in `git diff --name-only`); there's nothing to report for them.
        [[ -f "${x}" && "${x}" == *.py ]] || continue
        case "${x}" in
            autotunex/*) ;;
            steps/*) steps+=("${x}") ;;
            configurations/* | test-data/*) by_dir+=("${x}") ;;
            *) repo+=("${x}") ;;
        esac
    done
    [[ $((${#steps[@]} + ${#by_dir[@]} + ${#repo[@]})) -gt 0 ]] || return 1
    if [[ ${#repo[@]} -gt 0 ]]; then
        check_group "." "Repo: ${#repo[@]} file(s)" "" "${repo[@]}"
    fi
    check_grouped dir "${by_dir[@]+"${by_dir[@]}"}"
    check_grouped step "${steps[@]+"${steps[@]}"}"
}

echo 'typecheck start'
cd "${REPO_ROOT}" || exit 1

# Collect the changed files, NUL-delimited. Read with `read -d ''` rather than
# `mapfile -d`, which needs bash 4 (macOS ships bash 3.2); see limited-format.sh.
difflist="$(mktemp)"
trap 'rm -f "${difflist}"' EXIT
if ! git diff main...HEAD --name-only -z --format= > "${difflist}"; then
    echo "ERROR: 'git diff main...HEAD' failed — cannot determine changed files." >&2
    exit 1
fi
files=()
while IFS= read -r -d '' x; do
    files+=("${x}")
done < "${difflist}"

if ! check_changed "${files[@]+"${files[@]}"}"; then
    echo "no changed python files vs main; nothing to check"
fi

echo "${MY_LINE_BREAK}"
echo 'typecheck end'
