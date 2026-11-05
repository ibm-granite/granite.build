#!/usr/bin/env bash
# One-time local setup for the granite-switch recipes on AWS (README.md, step 4).
#
#   recipes/granite-switch/skypilot/aws/setup-local-space.sh <region>
#
# Reads FS, your EFS file system id, from $GB_HOME/local-env/gb-shared-<region>.ids
# (README.md, step 2) and writes, all outside the repo:
#
#   $GB_HOME/local-space/overlay/environments/skypilot-local/<name>/environment.yaml
#       the shipped environment template for <region>, with your EFS id
#   $GB_HOME/local-space/space.yaml
#       a standalone space: that overlay first, then this checkout's
#       configurations/assets (an existing one that lists the overlay is kept)
#   $GB_HOME/local-env/granite-switch.params
#       ENVIRONMENT_URI for `gb build start`; its other lines are kept
#
# GB_HOME is $GB_HOME_DIR, or ~/.granite.build. The script reads no secret and
# calls no AWS API. It is safe to re-run; ENVIRONMENT_URI points at the region
# set up last.
set -euo pipefail
umask 077

die() {
  echo "setup-local-space: $*" >&2
  exit 1
}
warn() { echo "setup-local-space: warning: $*" >&2; }

[ $# -eq 1 ] || die "usage: $0 <region>   (for example us-east-1)"
REGION="$1"
[[ "$REGION" =~ ^[a-z]{2}(-gov)?-[a-z]+-[0-9]+$ ]] || die "not an AWS region: $REGION"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
GB_HOME="${GB_HOME_DIR:-$HOME/.granite.build}"

# The ids file: gb-shared-<region>.ids, else the one *.ids file whose REGION= is
# <region>, whatever its name.
IDS="$GB_HOME/local-env/gb-shared-$REGION.ids"
if [ ! -f "$IDS" ]; then
  MATCHES="$(grep -lx "REGION=$REGION" "$GB_HOME"/local-env/*.ids 2>/dev/null || true)"
  case "$(printf '%s' "$MATCHES" | grep -c .)" in
    0) die "no ids file for $REGION in $GB_HOME/local-env: create the EFS and record its ids first (README.md, step 2)" ;;
    1) IDS="$MATCHES" ;;
    *) die "several ids files for $REGION; keep one:"$'\n'"$MATCHES" ;;
  esac
fi
FS="$(sed -n 's/^FS=//p' "$IDS" | tail -n 1 | tr -d "\"' ")"
[[ "$FS" =~ ^fs-[0-9a-f]{8,17}$ ]] ||
  die "$IDS: FS=${FS:-<missing>} is not an EFS file system id (fs-<hex>)"

# The shipped template for the region, and the region it is written for. Any
# other region starts from the us-east-1 one.
case "$REGION" in
  us-east-1) TEMPLATE=gb-shared FROM=us-east-1 NAME=aws-gb-shared ;;
  us-west-2) TEMPLATE=gb-shared-usw2 FROM=us-west-2 NAME=aws-gb-shared-usw2 ;;
  *) TEMPLATE=gb-shared FROM=us-east-1 NAME="aws-gb-shared-$REGION" ;;
esac
SRC="$REPO/configurations/assets/environments/skypilot/aws/$TEMPLATE/environment.yaml"
[ -f "$SRC" ] || die "$SRC not found: run this script from a granite.build checkout"
grep -q 'file_system_id: fs-REPLACE-ME$' "$SRC" || die "$SRC has no fs-REPLACE-ME placeholder"

# 1. The environment, with the EFS id, under a skypilot-local/ name so it never
#    shadows a shipped environment.
S="$GB_HOME/local-space"
DEST_DIR="$S/overlay/environments/skypilot-local/$NAME"
DEST="$DEST_DIR/environment.yaml"
mkdir -p "$DEST_DIR"
chmod 700 "$GB_HOME/local-env" "$S"
TMP="$(mktemp "$DEST_DIR/.environment.yaml.XXXXXX")"
{
  echo "# Written by recipes/granite-switch/skypilot/aws/setup-local-space.sh from"
  echo "# skypilot/aws/$TEMPLATE for $REGION. Holds your EFS id: keep it out of git."
  sed -e "s/fs-REPLACE-ME/$FS/" \
    -e "s/^name: .*/name: skypilot-local-$NAME/" \
    -e "s/$FROM/$REGION/g" "$SRC"
} >"$TMP"
grep -q "file_system_id: $FS\$" "$TMP" || {
  rm -f "$TMP"
  die "could not write the EFS id into $DEST"
}
chmod 600 "$TMP"
mv "$TMP" "$DEST"

# 2. The space: the overlay first, so its environments win.
OVERLAY="file://$S/overlay"
ASSETS="file://$REPO/configurations/assets"
if [ ! -f "$S/space.yaml" ]; then
  cat >"$S/space.yaml" <<EOF
# Written by recipes/granite-switch/skypilot/aws/setup-local-space.sh. A
# standalone space: the environments in the overlay, which hold your resource
# ids, win over the checkout's configurations/assets. Named public so it reads
# the space_secrets/public/ secrets of the default standalone space.
name: public
secret_manager:
  type: local
  config: {}
base_uris:
  - $OVERLAY
  - $ASSETS
EOF
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

# 3. The parameters file: ENVIRONMENT_URI replaced, other lines kept.
PARAMS="$GB_HOME/local-env/granite-switch.params"
URI="space://environments/skypilot-local/$NAME"
TMP="$(mktemp "$PARAMS.XXXXXX")"
{
  if [ -f "$PARAMS" ]; then grep -v '^ENVIRONMENT_URI=' "$PARAMS" || true; fi
  echo "ENVIRONMENT_URI=$URI"
} >"$TMP"
mv "$TMP" "$PARAMS"

cat <<EOF
Set up $REGION, with the EFS id from $IDS:
  environment  $DEST
  space        $S/space.yaml: $SPACE_NOTE
  parameters   $PARAMS: ENVIRONMENT_URI=$URI

Still to do by hand (this script never handles secrets; README.md, step 3):
  - the AWS key, as the secrets GB_AWS_ACCESS_KEY_ID and GB_AWS_SECRET_ACCESS_KEY;
  - an HF_TOKEN secret, if your data is in a private Hugging Face repo.
EOF
if ! grep -q '^SOURCE_DATA_URI=' "$PARAMS" || ! grep -q '^DATA_URI=' "$PARAMS"; then
  cat <<EOF
  - your data, as lines in $PARAMS:
      SOURCE_DATA_URI=<parquet dir>   for answerability-lora
      DATA_URI=<data root>            for answerability-sr
EOF
fi
cat <<EOF

Then start gbserver on the space and run a recipe:
  gbserver standalone --space-dir $S
  cd $HERE/answerability-lora
  gb build start -f build.yaml \$(sed 's/^/--param /' $PARAMS) --param RUN_ID=<unique>
EOF
