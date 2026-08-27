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

git subtree split --prefix=frontend/packages/ui-core "$tag" -b "$tmp_branch"
git tag "$ui_core_tag" "$tmp_branch"
git push origin "$ui_core_tag"
git branch -D "$tmp_branch"

echo "Published $ui_core_tag"
