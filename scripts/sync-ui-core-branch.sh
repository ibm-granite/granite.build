#!/bin/bash
# Keeps a rolling `ui-core` branch in sync with frontend/packages/ui-core's current
# contents, so external consumers can depend on the package as it is on main rather
# than only at tagged releases.
#
# Usage: scripts/sync-ui-core-branch.sh [<source-ref>] [<target-branch>] [<remote>]
#   Defaults: main ui-core origin
#   A fork validating this script before proposing changes upstream should pass
#   `upstream/main` as <source-ref> (see .github/workflows/sync-ui-core-branch.yml).
#
# Why a subtree split is needed at all: Yarn Classic v1 has no subdirectory support
# for git dependencies — it clones the whole repo and expects package.json at the
# root, but this repo's root is a Python project. `git subtree split` produces a ref
# whose *root* is the package, which Yarn can consume:
#   "@granite-build/ui-core": "ibm-granite/granite.build#ui-core"
#
# Relationship to publish-ui-core.sh: that script does the same split but emits an
# immutable per-release tag (ui-core-v0.3.4), invoked by tag-main.sh at release time.
# The two are complementary and coexist — tags for pinned releases, this branch for
# "current". Neither replaces the other.
set -euo pipefail

source_ref=${1:-main}
target_branch=${2:-ui-core}
remote=${3:-origin}

prefix="frontend/packages/ui-core"
tmp_branch="dist/ui-core-sync-tmp"
remote_ref="refs/remotes/${remote}/${target_branch}"

if ! git rev-parse -q --verify "$source_ref" >/dev/null; then
    echo "error: source ref '$source_ref' not found" >&2
    exit 1
fi

if [ -z "$(git ls-tree -d --name-only "$source_ref" "$prefix")" ]; then
    echo "error: '$prefix' does not exist in '$source_ref'" >&2
    exit 1
fi

# Fetch the target branch so its current tree can be compared below. Tolerates
# failure: on the first ever run the branch doesn't exist yet, which is not an error.
git fetch --quiet "$remote" "+refs/heads/${target_branch}:${remote_ref}" 2>/dev/null || true

# Clear any stray temp branch left behind by a prior failed run, so this run doesn't
# fail on `git subtree split -b` before even getting started.
git branch -D "$tmp_branch" >/dev/null 2>&1 || true
trap 'git branch -D "$tmp_branch" >/dev/null 2>&1 || true' EXIT

# --quiet suppresses the per-commit progress counter, which is one line per commit
# walked (~225 and growing) and pure noise in a CI log. Real errors still surface,
# and `set -e` aborts on a non-zero exit either way.
# stdout is just the resulting SHA, which is reported explicitly below.
git subtree split --quiet --prefix="$prefix" "$source_ref" -b "$tmp_branch" >/dev/null

new_tree=$(git rev-parse "${tmp_branch}^{tree}")

# Skip the push when the content is byte-identical to what's already published.
# Not just tidiness: yarn resolves and caches git dependencies by commit SHA, so a
# force-push that changed the SHA without changing the tree — which is exactly what
# an upstream rebase or an amended commit produces — would invalidate every
# consumer's resolved dependency for no reason.
if git rev-parse -q --verify "$remote_ref" >/dev/null; then
    if [ "$new_tree" = "$(git rev-parse "${remote_ref}^{tree}")" ]; then
        echo "$target_branch is already up to date (tree $new_tree) — nothing to push"
        exit 0
    fi
fi

# --force because each split rewrites history from scratch: the commits are derived
# from the source ref's history, not appended to whatever the branch held before.
git push --force "$remote" "${tmp_branch}:refs/heads/${target_branch}"

echo "Pushed $prefix ($source_ref) to $remote/$target_branch"
echo "  commit: $(git rev-parse "$tmp_branch")"
echo "  tree:   $new_tree"
