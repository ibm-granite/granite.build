#!/usr/bin/env bash
# Preload the Docker images CI tests use, so the tests themselves never pull.
#
#   load   load the cached tarball, then pull missing images with retry
#   save   write all manifest images to the tarball
#
# CACHE_DIR is what actions/cache restores and saves (see ci.yml). Tests pull
# if-not-present, so a preloaded image is used as-is.
set -euo pipefail

IMAGE_LIST="${CI_DOCKER_IMAGE_LIST:-test-data/ci-docker-images.env}"
CACHE_DIR="${CI_DOCKER_IMAGE_CACHE_DIR:-$HOME/.cache/ci-docker-images}"
TARBALL="$CACHE_DIR/images.tar"
MAX_ATTEMPTS="${CI_DOCKER_PULL_ATTEMPTS:-6}"
# Backoff doubles from BASE_DELAY to MAX_DELAY, plus up to BASE_DELAY of jitter
# so matrix jobs don't retry in lockstep.
BASE_DELAY="${CI_DOCKER_PULL_BASE_DELAY:-15}"
MAX_DELAY="${CI_DOCKER_PULL_MAX_DELAY:-120}"

# Print the image of each NAME=image line; exit 1 on a malformed line.
read_images() {
    awk -F= '
        /^[[:space:]]*(#|$)/ { next }
        {
            name = $1; image = substr($0, index($0, "=") + 1)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", image)
            if (NF < 2 || name == "" || image == "") {
                printf "%s:%d: expected NAME=image, got: %s\n", FILENAME, NR, $0 > "/dev/stderr"
                exit 1
            }
            print image
        }' "$IMAGE_LIST"
}

pull_with_retry() {
    local image="$1" attempt=1 delay="$BASE_DELAY" out
    while true; do
        if out=$(docker pull --quiet "$image" 2>&1); then
            echo "Pulled $image"
            return 0
        fi
        echo "Pull of $image failed (attempt $attempt/$MAX_ATTEMPTS): $out" >&2
        # A missing or invalid reference won't fix itself on retry; fail fast.
        if grep -q -i -E 'manifest unknown|repository does not exist|invalid reference format' <<<"$out"; then
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
        # A corrupt tarball falls through to pulling.
        docker load --input "$TARBALL" || echo "docker load failed; pulling instead" >&2
    else
        echo "No cached images at $TARBALL"
    fi

    local image failed=0 images
    # Assigned apart from `local` so set -e catches a malformed manifest.
    images=$(read_images)
    for image in $images; do
        if docker image inspect "$image" >/dev/null 2>&1; then
            echo "Present: $image"
        elif ! pull_with_retry "$image"; then
            echo "::error::Could not pull $image" >&2
            failed=1
        fi
    done
    return "$failed"
}

cmd_save() {
    mkdir -p "$CACHE_DIR"
    local images
    images=$(read_images)
    # shellcheck disable=SC2086  # one image reference per word
    docker save --output "$TARBALL" $images
    echo "Saved $(wc -w <<<"$images" | tr -d ' ') images to $TARBALL ($(du -h "$TARBALL" | cut -f1))"
}

case "${1:-}" in
load) cmd_load ;;
save) cmd_save ;;
*)
    echo "usage: $0 {load|save}" >&2
    exit 2
    ;;
esac
