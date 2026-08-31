#!/bin/bash
# Publishes frontend/packages/ui-core as a standalone git ref that external consumers
# (e.g. the internal deployment repo) can depend on directly via yarn's git dependency
# shorthand, without granite.build needing to publish to an npm registry.
#
# Usage: scripts/publish-ui-core.sh <tag>
#   <tag> is the same release tag passed to tag-main.sh, e.g. v0.3.4
#
# Produces a tag named ui-core-<tag> whose sole content is frontend/packages/ui-core's
# tree at that point in history (via `git subtree split`), and pushes it to origin.
# Consumers pin to it with, e.g.:
#   "@granite-build/ui-core": "ibm-granite/granite.build#ui-core-v0.3.4"
set -euo pipefail

tag=${1:-}
if [ -z "$tag" ]; then
    echo "usage: $0 <tag>"
    exit 1
fi

ui_core_tag="ui-core-${tag}"
tmp_branch="dist/ui-core-tmp-${tag}"

if git rev-parse -q --verify "refs/tags/$ui_core_tag" >/dev/null; then
    echo "error: tag $ui_core_tag already exists locally — delete it first (git tag -d $ui_core_tag) if you intend to re-publish" >&2
    exit 1
fi

# Clear any stray temp branch left behind by a prior failed run, so this run
# doesn't fail on `git subtree split -b` before even getting started.
git branch -D "$tmp_branch" >/dev/null 2>&1 || true
trap 'git branch -D "$tmp_branch" >/dev/null 2>&1 || true' EXIT

git subtree split --prefix=frontend/packages/ui-core "$tag" -b "$tmp_branch"
git tag "$ui_core_tag" "$tmp_branch"
git push origin "$ui_core_tag"

echo "Published $ui_core_tag"
