#!/bin/bash
# Tags the main branch with a tag provided on the command line.
# tags are generally of the form vX.Y.z, for example v0.2.36
# Warning: there is not check for tag collision
#
# In addition to the immutable vX.Y.Z tag, this maintains two moving tags:
#   stable         - always advanced to the latest release; what the documented
#                    upgrade command pins (git+https://...granite.build.git@stable).
#   min-supported  - the floor below which the CLI hard-blocks. NOT advanced
#                    automatically (moving it drops support for older clients);
#                    advance it only with the --move-min-supported flag.
#
# Usage: scripts/tag-main.sh vX.Y.Z [--move-min-supported]
set -euo pipefail

tag=${1:-}
if [ -z "$tag" ]; then
    echo tag value must be provided
    exit 1
fi
# After PR to main is merged
git checkout main
git pull --ff-only
# List existing tags
git tag
# Define a new tag
git tag $tag
git push origin $tag

# Advance the rolling `stable` tag to this release. Force re-point + force push so
# re-running the script is idempotent. Lightweight tag => shares this commit's SHA.
git tag -f stable "$tag"
git push -f origin stable

# `min-supported` marks the floor below which the CLI refuses to run. Advance it only
# when explicitly requested, since moving it drops support for older clients. As a
# lightweight tag it shares the vX.Y.Z commit SHA, which is how the CLI resolves the
# floor version from the tags listing.
if [ "${2:-}" = "--move-min-supported" ]; then
    echo "Advancing min-supported floor to $tag"
    git tag -f min-supported "$tag"
    git push -f origin min-supported
fi

# Publish frontend/packages/ui-core as its own git ref at this tag, so external
# consumers (e.g. the internal deployment repo) can depend on it directly.
"$(dirname "$0")/publish-ui-core.sh" "$tag" 
