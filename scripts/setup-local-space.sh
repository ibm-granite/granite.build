#!/usr/bin/env bash
# Turn a $${KEY}-templated SkyPilot environment and a private site file into a
# usable environment in a gbserver space.
#
#   scripts/setup-local-space.sh --template T --site S --name NAME
#                                [--space-dir DIR] [--out DIR] [--env-key KEY]
#
#   --template T   the environment template; $${KEY} values and <% %> blocks,
#                  the gb CLI's build.yaml parameter syntax. gbserver's own
#                  {{ ... }} and {% %} templates are left as they are.
#   --site S       the site file: a flat YAML mapping of every site-specific
#                  value, kept outside the public repo (a private repo, or
#                  $GB_HOME/local-env/). The same file is later passed to
#                  `gb build start --parameters-path S`.
#   --name NAME    the environment's name, [a-z0-9][a-z0-9-]*
#   --space-dir D  the standalone space to write to; default $GB_HOME/local-space
#   --out DIR      write DIR/environments/skypilot-local/NAME/environment.yaml
#                  instead, for a private repo backing a shared gbserver; GB_HOME
#                  and space.yaml are not touched
#   --env-key KEY  the site file key set to the environment's URI; default
#                  ENVIRONMENT_URI
#
# Without --out it writes:
#
#   <space-dir>/overlay/environments/skypilot-local/NAME/environment.yaml
#       T rendered with S, under a skypilot-local/ name so it never shadows a
#       shipped environment
#   <space-dir>/space.yaml
#       a standalone space: that overlay first, then this checkout's
#       configurations/assets (an existing one that lists the overlay is kept)
#
# and in both cases sets KEY in S to space://environments/skypilot-local/NAME,
# leaving every other line of S as it was.
#
# GB_HOME is $GB_HOME_DIR, or ~/.granite.build. The Python used for rendering is
# $GB_PYTHON, else this checkout's .venv/bin/python, else python3; it must be
# able to import gbcli. The script reads no secret: the site file only names
# secrets. It is safe to re-run.
set -euo pipefail
umask 077

PROG="setup-local-space"
die() {
  echo "$PROG: $*" >&2
  exit 1
}
warn() { echo "$PROG: warning: $*" >&2; }
usage() {
  cat >&2 <<EOF
usage: $0 --template T --site S --name NAME [--space-dir DIR] [--out DIR] [--env-key KEY]
EOF
  [ $# -eq 0 ] || echo "$PROG: $*" >&2
  exit 2
}

TEMPLATE="" SITE="" NAME="" SPACE_DIR="" OUT="" ENV_KEY="ENVIRONMENT_URI"
HAVE_NAME=0
while [ $# -gt 0 ]; do
  case "$1" in
    --template | --site | --name | --space-dir | --out | --env-key)
      [ $# -ge 2 ] || usage "$1 needs a value"
      case "$1" in
        --template) TEMPLATE="$2" ;;
        --site) SITE="$2" ;;
        --name) NAME="$2" HAVE_NAME=1 ;;
        --space-dir) SPACE_DIR="$2" ;;
        --out) OUT="$2" ;;
        --env-key) ENV_KEY="$2" ;;
      esac
      shift 2
      ;;
    -h | --help) usage ;;
    *) usage "unknown argument: $1" ;;
  esac
done
[ -n "$TEMPLATE" ] || usage "--template is required"
[ -n "$SITE" ] || usage "--site is required"
[ "$HAVE_NAME" -eq 1 ] || usage "--name is required"
[[ "$NAME" =~ ^[a-z0-9][a-z0-9-]*$ ]] ||
  die "--name '$NAME' is not a valid name: use [a-z0-9][a-z0-9-]*"
[[ "$ENV_KEY" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] ||
  die "--env-key '$ENV_KEY' is not a valid key: use [A-Za-z_][A-Za-z0-9_]*"
[ -z "$OUT" ] || [ -z "$SPACE_DIR" ] || die "--out and --space-dir exclude each other"
[ -f "$TEMPLATE" ] || die "$TEMPLATE: no such template"
[ -f "$SITE" ] || die "$SITE: no such site file"
[ -w "$SITE" ] || die "$SITE: not writable"

abspath() { echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }
TEMPLATE="$(abspath "$TEMPLATE")"
SITE="$(abspath "$SITE")"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
GB_HOME="${GB_HOME_DIR:-$HOME/.granite.build}"

if [ -n "${GB_PYTHON:-}" ]; then
  PY="$GB_PYTHON"
elif [ -x "$REPO/.venv/bin/python" ]; then
  PY="$REPO/.venv/bin/python"
else
  PY="python3"
fi
"$PY" -c 'import gbcli.utils.buildutil, gbcli.services.service_build' 2>/dev/null ||
  die "$PY cannot import gbcli; set GB_PYTHON to a Python with granite.build" \
    "installed, or run 'make venv' in $REPO"

# 1. The environment: rendered into a scratch dir first, so a template that
#    fails to render leaves nothing behind, not even an empty directory.
SCRATCH="$(mktemp -d)"
trap 'rm -rf -- "$SCRATCH"' EXIT
RC=0
"$PY" "$HERE/render_env_template.py" "$TEMPLATE" "$SITE" "$SCRATCH/rendered.yaml" || RC=$?
[ "$RC" -eq 0 ] || die "could not render $TEMPLATE from $SITE; nothing written"

if [ -n "$OUT" ]; then
  mkdir -p "$OUT"
  OUT="$(cd "$OUT" && pwd)"
  DEST_DIR="$OUT/environments/skypilot-local/$NAME"
else
  S="${SPACE_DIR:-$GB_HOME/local-space}"
  mkdir -p "$S"
  S="$(cd "$S" && pwd)"
  chmod 700 "$S"
  DEST_DIR="$S/overlay/environments/skypilot-local/$NAME"
fi
DEST="$DEST_DIR/environment.yaml"
mkdir -p "$DEST_DIR"
TMP="$(mktemp "$DEST_DIR/.environment.yaml.XXXXXX")"
{
  echo "# Rendered from $TEMPLATE"
  echo "# with $SITE;"
  echo "# written by scripts/setup-local-space.sh; holds site values, keep it out of git."
  cat "$SCRATCH/rendered.yaml"
} >"$TMP"
chmod 600 "$TMP"
mv -f "$TMP" "$DEST"

# 2. The space: the overlay first, so its environments win.
if [ -z "$OUT" ]; then
  OVERLAY="file://$S/overlay"
  ASSETS="file://$REPO/configurations/assets"
  if [ ! -f "$S/space.yaml" ]; then
    TMP="$(mktemp "$S/.space.yaml.XXXXXX")"
    cat >"$TMP" <<EOF
# Written by scripts/setup-local-space.sh. A standalone space: the environments
# in the overlay, which hold your site values, win over the checkout's
# configurations/assets. Named public so it reads the space_secrets/public/
# secrets of the default standalone space.
name: public
secret_manager:
  type: local
  config: {}
base_uris:
  - $OVERLAY
  - $ASSETS
EOF
    chmod 600 "$TMP"
    mv -f "$TMP" "$S/space.yaml"
    SPACE_NOTE="written"
  elif grep -qF -- "- $OVERLAY" "$S/space.yaml"; then
    SPACE_NOTE="kept (it already lists the overlay)"
    grep -qF -- "- $ASSETS" "$S/space.yaml" ||
      warn "$S/space.yaml does not list this checkout's assets, $ASSETS"
  else
    SPACE_NOTE="NOT changed"
    warn "$S/space.yaml exists but does not list the overlay; left as is." \
      "Add '- $OVERLAY' as the first entry of its base_uris."
  fi
fi

# 3. The site file: KEY set in place, or appended; every other byte kept.
URI="space://environments/skypilot-local/$NAME"
"$PY" - "$SITE" "$ENV_KEY" "$URI" <<'PYEOF'
import os
import re
import sys
import tempfile

path, key, uri = sys.argv[1:]
with open(path, "rb") as f:
    data = f.read()
# NAME is [a-z0-9-] only, so single quotes need no escaping.
line = f"{key}: '{uri}'".encode()
pattern = re.compile(rb"^" + re.escape(key.encode()) + rb"[ \t]*:")
lines = data.splitlines(keepends=True)
found = False
for i, old in enumerate(lines):
    if pattern.match(old):
        body = old.rstrip(b"\r\n")
        lines[i] = line + old[len(body):]
        found = True
if not found:
    if lines and not lines[-1].endswith((b"\n", b"\r")):
        lines[-1] += b"\n"
    lines.append(line + b"\n")
new = b"".join(lines)
if new != data:
    mode = os.stat(path).st_mode & 0o7777
    directory = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=directory)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as f:
            f.write(new)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
PYEOF

echo "Set up environment $NAME from $TEMPLATE:"
echo "  environment  $DEST"
[ -n "$OUT" ] || echo "  space        $S/space.yaml: $SPACE_NOTE"
echo "  site         $SITE: $ENV_KEY=$URI"
echo
if [ -n "$OUT" ]; then
  echo "Commit $OUT to the private repo your gbserver's space reads, then run a recipe:"
else
  echo "Then start gbserver on the space and run a recipe:"
  echo "  gbserver standalone --space-dir $S"
fi
echo "  gb build start -f <recipe>/build.yaml --parameters-path $SITE"
