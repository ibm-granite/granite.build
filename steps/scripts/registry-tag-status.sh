#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
#
# registry-tag-status.sh [--repository | --config-digest] <image-ref>
#
# Does this image tag (or, with --repository, this repository) exist on its
# registry? Prints one word and exits with its code:
#   present  0   the registry returned the manifest (or the repository's tag list)
#   absent   1   the registry said the tag, or the repository, does not exist
#   unknown  2   anything else: network error, auth failure, an unexpected answer
# Callers must treat `unknown` as neither answer: `make publish-image` refuses to
# overwrite a `present` tag, and `make publish-step` refuses an image that is not
# `present`.
#
# --config-digest prints the published image's config digest instead of `present`
# (exit 0); `absent`/`unknown` as above. For a single-platform image that digest is
# the local image ID (`<docker> image inspect --format '{{.Id}}'`), so `make test`
# can tell whether the image it built is the one already published under the tag.
# A multi-platform index has no single config, so it is `unknown`.
#
# Asks the registry only (no layer pull):
#   1. tag mode: `skopeo inspect --raw`, when skopeo is installed. skopeo is
#      OPTIONAL: nothing installs it (not the repo's venv, not CI); install it
#      yourself if you prefer it (brew/dnf/apt install skopeo);
#   2. otherwise the registry's v2 API with curl, which is all this script needs
#      (the manifest, or with --repository the tag list), following a Bearer
#      token challenge. The token request uses, first match wins:
#        - REGISTRY_USER and REGISTRY_PASSWORD from the environment, when both set;
#        - else a stored `podman login`: the `auths` entry for the image's
#          host/path, its namespace or its host, in $REGISTRY_AUTH_FILE or
#          ~/.config/containers/auth.json ($XDG_RUNTIME_DIR/containers/auth.json
#          on Linux). Logins kept in a credential helper are not read;
#        - else anonymous.
#      Credentials matter for a private repo, and on quay.io, which answers an
#      anonymous probe of a repository that does not exist yet with 401.
#      They go to curl as config on stdin, never on its command line.
# `podman/docker manifest inspect` is not used: podman fails on a single-platform
# image ("Treating single images as manifest lists is not implemented").
set -u

mode=tag
case "${1:-}" in
  --repository) mode=repository; shift ;;
  --config-digest) mode=config; shift ;;
esac
ref="${1:-}"
if [ -z "$ref" ]; then
  echo "usage: registry-tag-status.sh [--repository | --config-digest] <image-ref>" >&2
  echo unknown
  exit 2
fi

say() {
  echo "$1"
  case "$1" in present) exit 0 ;; absent) exit 1 ;; *) exit 2 ;; esac
}

# --- 1. skopeo (tags only) ----------------------------------------------------------
if [ "$mode" = tag ] && command -v skopeo >/dev/null 2>&1; then
  if err="$(skopeo inspect --raw "docker://${ref}" 2>&1 >/dev/null)"; then
    say present
  fi
  case "$err" in
    *"manifest unknown"*|*"MANIFEST_UNKNOWN"*|*"name unknown"*|*"NAME_UNKNOWN"*) say absent ;;
  esac
  # Any other skopeo failure: ask the registry directly before giving up.
fi

command -v curl >/dev/null 2>&1 || say unknown

# --- 2. registry v2 API --------------------------------------------------------------
# Split <host>/<path>[:<tag>|@<digest>], with Docker Hub's defaults.
name="$ref"
reference=""
case "$name" in
  *@*) reference="${name#*@}"; name="${name%%@*}" ;;
esac
if [ -z "$reference" ]; then
  last="${name##*/}"
  case "$last" in
    *:*) reference="${last##*:}"; name="${name%:*}" ;;
    *) reference="latest" ;;
  esac
fi
first="${name%%/*}"
if [ "$first" != "$name" ] && { case "$first" in *.*|*:*|localhost) true ;; *) false ;; esac; }; then
  host="$first"
  repo="${name#*/}"
else
  host="registry-1.docker.io"
  repo="$name"
  case "$repo" in */*) ;; *) repo="library/${repo}" ;; esac
fi
[ "$host" = "docker.io" ] && host="registry-1.docker.io"

accept="application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json"
if [ "$mode" = repository ]; then
  url="https://${host}/v2/${repo}/tags/list"
else
  url="https://${host}/v2/${repo}/manifests/${reference}"
fi
tmp="$(mktemp -d)" || say unknown
trap 'rm -rf "$tmp"' EXIT

# "user:password" of a stored podman login for this image, or nothing. Tries the
# longest auths key first: <host>/<repo path...>, then each shorter prefix, then
# <host>. The auth value is base64("user:password").
stored_login() {
  for f in "${REGISTRY_AUTH_FILE:-}" "${XDG_RUNTIME_DIR:-/nonexistent}/containers/auth.json" \
           "${HOME:-/nonexistent}/.config/containers/auth.json"; do
    [ -n "$f" ] && [ -f "$f" ] || continue
    flat="$(tr -d '\n' < "$f")"
    key="${first}/${repo}"
    [ "$host" = "registry-1.docker.io" ] && key="docker.io/${repo}"
    while :; do
      b64="$(printf '%s' "$flat" | sed -n "s|.*\"${key}\" *: *{[^}]*\"auth\" *: *\"\([A-Za-z0-9+/=]*\)\".*|\1|p")"
      if [ -n "$b64" ]; then
        printf '%s' "$b64" | base64 -d 2>/dev/null || printf '%s' "$b64" | base64 --decode 2>/dev/null
        return 0
      fi
      case "$key" in */*) key="${key%/*}" ;; *) break ;; esac
    done
  done
  return 0
}

# Ask the registry; $1 is curl config (an Authorization header) for stdin. A HEAD
# for a manifest, a GET for the tag list (body discarded) or for the manifest
# itself (--config-digest, body kept).
probe() {
  if [ "$mode" = repository ]; then
    printf '%s' "$1" | curl -s -o /dev/null -D "$tmp/headers" -w '%{http_code}' \
      --max-time 30 --config - "$url" 2>/dev/null
  elif [ "$mode" = config ]; then
    printf '%s' "$1" | curl -s -o "$tmp/manifest" -D "$tmp/headers" -w '%{http_code}' \
      --max-time 30 -H "Accept: ${accept}" --config - "$url" 2>/dev/null
  else
    printf '%s' "$1" | curl -s -I -o /dev/null -D "$tmp/headers" -w '%{http_code}' \
      --max-time 30 -H "Accept: ${accept}" --config - "$url" 2>/dev/null
  fi
}

code="$(probe "")" || say unknown
if [ "$code" = "401" ]; then
  challenge="$(tr -d '\r' < "$tmp/headers" | sed -n '/^[Ww][Ww][Ww]-[Aa]uthenticate: *[Bb]earer */{s///p;q;}')"
  [ -n "$challenge" ] || say unknown
  param() { printf '%s' "$challenge" | sed -n "s/.*$1=\"\([^\"]*\)\".*/\1/p"; }
  realm="$(param realm)"
  service="$(param service)"
  scope="$(param scope)"
  [ -n "$realm" ] || say unknown
  [ -n "$scope" ] || scope="repository:${repo}:pull"
  creds=""
  if [ -n "${REGISTRY_USER:-}" ] && [ -n "${REGISTRY_PASSWORD:-}" ]; then
    creds="user = \"${REGISTRY_USER}:${REGISTRY_PASSWORD}\""
  else
    stored="$(stored_login)"
    [ -n "$stored" ] && creds="user = \"${stored}\""
  fi
  tcode="$(printf '%s' "$creds" | curl -s -o "$tmp/token" -w '%{http_code}' --max-time 30 \
    --config - "${realm}?service=${service}&scope=${scope}" 2>/dev/null)" || say unknown
  [ "$tcode" = "200" ] || say unknown
  token="$(tr -d '\n' < "$tmp/token" | sed -n 's/.*"\(access_\)\{0,1\}token" *: *"\([^"]*\)".*/\2/p')"
  [ -n "$token" ] || say unknown
  code="$(probe "header = \"Authorization: Bearer ${token}\"")" || say unknown
fi

if [ "$mode" = config ] && [ "$code" = "200" ]; then
  # The config digest of a single-image manifest: "config":{...,"digest":"sha256:..."}.
  digest="$(tr -d '\n' < "$tmp/manifest" | sed -n 's/.*"config" *: *{[^}]*"digest" *: *"\(sha256:[0-9a-f]*\)".*/\1/p')"
  [ -n "$digest" ] || say unknown
  echo "$digest"
  exit 0
fi

case "$code" in
  200) say present ;;
  404) say absent ;;
  *) say unknown ;;
esac
