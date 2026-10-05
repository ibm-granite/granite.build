#!/usr/bin/env bash
# Preload the Docker images CI tests need, so the tests themselves never pull.
#
#   ci-docker-images.sh load   load the cached tarball if present, then pull any
#                              image still missing, retrying with backoff
#   ci-docker-images.sh save   write all listed images to the cached tarball
#
# The tarball directory is what actions/cache restores and saves (see ci.yml).
# Tests use pull_policy if-not-present, so a preloaded image is never re-pulled.
set -euo pipefail

IMAGE_LIST="${CI_DOCKER_IMAGE_LIST:-test-data/ci-docker-images.env}"
CACHE_DIR="${CI_DOCKER_IMAGE_CACHE_DIR:-$HOME/.cache/ci-docker-images}"
TARBALL="$CACHE_DIR/images.tar"
MAX_ATTEMPTS="${CI_DOCKER_PULL_ATTEMPTS:-6}"
# Backoff doubles from BASE up to CAP, plus up to BASE of jitter so the two
# matrix jobs don't retry in lockstep against the same per-IP limit.
BASE_DELAY="${CI_DOCKER_PULL_BASE_DELAY:-15}"
MAX_DELAY="${CI_DOCKER_PULL_MAX_DELAY:-120}"

# Manifest lines are NAME=image; print just the images.
read_images() {
    grep -v -E '^[[:space:]]*(#|$)' "$IMAGE_LIST" | cut -d= -f2- |
        sed -E 's/^[[:space:]]+|[[:space:]]+$//g' | grep .
}

pull_with_retry() {
    local image="$1" attempt=1 delay="$BASE_DELAY" out
    while true; do
        if out=$(docker pull --quiet "$image" 2>&1); then
            echo "Pulled $image"
            return 0
        fi
        echo "Pull of $image failed (attempt $attempt/$MAX_ATTEMPTS): $out" >&2
        # A missing tag won't appear on retry; fail now rather than after minutes.
        if grep -q -i -E 'manifest unknown|repository does not exist' <<<"$out"; then
            return 1
        fi
        if ((attempt >= MAX_ATTEMPTS)); then
            return 1
        fi
        local sleep_for=$((delay + RANDOM % (BASE_DELAY + 1)))
        echo "Retrying in ${sleep_for}s" >&2
        sleep "$sleep_for"
        attempt=$((attempt + 1))
        delay=$((delay * 2 > MAX_DELAY ? MAX_DELAY : delay * 2))
    done
}

cmd_load() {
    if [[ -f "$TARBALL" ]]; then
        echo "Loading cached images from $TARBALL"
        # A corrupt tarball shouldn't fail the job; fall through to pulling.
        docker load --input "$TARBALL" || echo "docker load failed; pulling instead" >&2
    else
        echo "No cached images at $TARBALL"
    fi

    local image failed=0
    while read -r image; do
        if docker image inspect "$image" >/dev/null 2>&1; then
            echo "Present: $image"
        elif ! pull_with_retry "$image"; then
            echo "::error::Could not pull $image" >&2
            failed=1
        fi
    done < <(read_images)
    return "$failed"
}

cmd_save() {
    mkdir -p "$CACHE_DIR"
    local image images=()
    while read -r image; do images+=("$image"); done < <(read_images)
    docker save --output "$TARBALL" "${images[@]}"
    echo "Saved ${#images[@]} images to $TARBALL ($(du -h "$TARBALL" | cut -f1))"
}

case "${1:-}" in
load) cmd_load ;;
save) cmd_save ;;
*)
    echo "usage: $0 {load|save}" >&2
    exit 2
    ;;
esac
