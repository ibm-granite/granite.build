#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Clone lm-engine at the UPSTREAM sha and check every patch applies, in order.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
sha="$(awk '/^sha:/{print $2}' "$here/UPSTREAM")"
repo="$(awk '/^repo:/{print $2}' "$here/UPSTREAM")"
work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
git clone --quiet --filter=blob:none "$repo" "$work/lm-engine"
git -C "$work/lm-engine" checkout --quiet "$sha"
for p in "$here"/patches/*.patch; do
  git -C "$work/lm-engine" apply --check "$p"
  git -C "$work/lm-engine" apply "$p"
  echo "ok $(basename "$p")"
done
