#!/usr/bin/env bash
# One-time local setup for the granite-switch recipes on AWS (README.md, step 4).
#
#   recipes/granite-switch/skypilot/aws/setup-local-space.sh <region>
#
# Reads FS, your EFS file system id, from $GB_HOME/local-env/gb-shared-<region>.ids
# (README.md, step 2), looks up the GPU host AMI, and writes, all outside the repo:
#
#   $GB_HOME/local-space/overlay/environments/skypilot-local/<name>/environment.yaml
#       environment-template.yaml (next to this script) with your EFS id, region
#       and host AMI; <name> is aws-gb-shared for us-east-1, aws-gb-shared-usw2
#       for us-west-2, aws-gb-shared-<region> otherwise
#   $GB_HOME/local-space/space.yaml
#       a standalone space: that overlay first, then this checkout's
#       configurations/assets (an existing one that lists the overlay is kept)
#   $GB_HOME/local-env/granite-switch.params
#       ENVIRONMENT_URI and HOST_AMI for `gb build start`; its other lines are kept
#
# The host AMI is the AWS Deep Learning Base GPU AMI for the region: without it an
# 8-GPU H200 instance does not see its GPUs (NVSwitch needs the DLAMI's NVIDIA
# Fabric Manager). An AMI=ami-... line in the ids file pins one; otherwise one AWS
# call reads the latest from AWS's public SSM parameter, with the AWS CLI and
# profile of README.md steps 1-2.
#
# GB_HOME is $GB_HOME_DIR, or ~/.granite.build. The script reads no secret. It is
# safe to re-run; ENVIRONMENT_URI and HOST_AMI point at the region set up last.
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
# <region>, whatever its name; the value may be quoted, as FS's may.
IDS="$GB_HOME/local-env/gb-shared-$REGION.ids"
if [ ! -f "$IDS" ]; then
  MATCHES="$(grep -lE "^REGION=[\"']?$REGION[\"']?[[:space:]]*\$" "$GB_HOME"/local-env/*.ids 2>/dev/null || true)"
  case "$(printf '%s' "$MATCHES" | grep -c .)" in
    0) die "no ids file for $REGION in $GB_HOME/local-env: create the EFS and record its ids first (README.md, step 2)" ;;
    1) IDS="$MATCHES" ;;
    *) die "several ids files for $REGION; keep one:"$'\n'"$MATCHES" ;;
  esac
fi
FS="$(sed -n 's/^FS=//p' "$IDS" | tail -n 1 | tr -d "\"' ")"
[[ "$FS" =~ ^fs-[0-9a-f]{8,17}$ ]] ||
  die "$IDS: FS=${FS:-<missing>} is not an EFS file system id (fs-<hex>)"

# The GPU host image: the AWS Deep Learning Base GPU AMI for this region. An AMI=
# line in the ids file pins one; otherwise AWS's public SSM parameter names the
# latest build.
SSM_AMI=/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id
AMI="$(sed -n 's/^AMI=//p' "$IDS" | tail -n 1 | tr -d "\"' ")"
if [ -n "$AMI" ]; then
  [[ "$AMI" =~ ^ami-[0-9a-f]{8,17}$ ]] || die "$IDS: AMI=$AMI is not an AMI id (ami-<hex>)"
  AMI_FROM="AMI= in $IDS"
else
  AMI="$(aws ssm get-parameter --region "$REGION" --name "$SSM_AMI" \
    --query Parameter.Value --output text 2>/dev/null)" || AMI=""
  [[ "$AMI" =~ ^ami-[0-9a-f]{8,17}$ ]] ||
    die "could not look up the Deep Learning Base GPU AMI for $REGION (aws ssm get-parameter $SSM_AMI)." \
      "Check the AWS CLI and its credentials, or pin one with an AMI=ami-... line in $IDS."
  AMI_FROM="$SSM_AMI"
fi

# One template for every region; the environment's name says which region its
# EFS is in (us-east-1 keeps the plain name the recipes default to).
case "$REGION" in
  us-east-1) NAME=aws-gb-shared ;;
  us-west-2) NAME=aws-gb-shared-usw2 ;;
  *) NAME="aws-gb-shared-$REGION" ;;
esac
SRC="$HERE/environment-template.yaml"
[ -f "$SRC" ] || die "$SRC not found: run this script from a granite.build checkout"
grep -q 'file_system_id: fs-REPLACE-ME$' "$SRC" || die "$SRC has no fs-REPLACE-ME placeholder"
grep -q 'host_image_id: AMI-REPLACE-ME' "$SRC" || die "$SRC has no AMI-REPLACE-ME placeholder"

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
  echo "# environment-template.yaml for $REGION. Holds your EFS id: keep it out of git."
  echo "# Host AMI: $AMI (from $AMI_FROM)."
  sed -e "s/fs-REPLACE-ME/$FS/" \
    -e "s/^name: .*/name: skypilot-local-$NAME/" \
    -e "s/AMI-REPLACE-ME/$AMI/g" \
    -e "s/REGION-REPLACE-ME/$REGION/g" "$SRC"
} >"$TMP"
grep -q "file_system_id: $FS\$" "$TMP" && grep -q "host_image_id: $AMI" "$TMP" &&
  ! grep -q "REPLACE-ME" "$TMP" || {
  rm -f "$TMP"
  die "could not fill in the EFS id, region and AMI in $DEST"
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

# 3. The parameters file: ENVIRONMENT_URI and HOST_AMI (the same region's, for the
#    answerability-sr recipe) replaced, other lines kept.
PARAMS="$GB_HOME/local-env/granite-switch.params"
URI="space://environments/skypilot-local/$NAME"
TMP="$(mktemp "$PARAMS.XXXXXX")"
{
  if [ -f "$PARAMS" ]; then grep -vE '^(ENVIRONMENT_URI|HOST_AMI)=' "$PARAMS" || true; fi
  echo "ENVIRONMENT_URI=$URI"
  echo "HOST_AMI=$AMI"
} >"$TMP"
mv "$TMP" "$PARAMS"

cat <<EOF
Set up $REGION, with the EFS id from $IDS:
  environment  $DEST
  host AMI     $AMI (from $AMI_FROM)
  space        $S/space.yaml: $SPACE_NOTE
  parameters   $PARAMS: ENVIRONMENT_URI=$URI, HOST_AMI=$AMI

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
