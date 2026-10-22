#!/bin/bash
# Verify that each checkpoint dir for an SFT run has the GRANITE chat_template.jinja
# (not the tulu default SFT bakes in). Confirms the chat-template injection in the
# openinstruct-sft step landed in every epoch_hf_*/step_hf_* and the final checkpoint.
# See docs/granite4_nano.md "Chat Template Injection into AWS Checkpoints".
#
# Usage: recipes/granite4-350m/aws/sft/verify_chat_template.sh <NAME> [region]
#   e.g. recipes/granite4-350m/aws/sft/verify_chat_template.sh test4-l40s-8
# Requires valid AWS credentials for the checkpoint bucket's region.
set -uo pipefail

NAME="${1:-smoke-ct-test}"
REGION="${2:-us-east-2}"
BUCKET="s3://granite-build-checkpoints"

# --- Preflight: AWS auth ---------------------------------------------------
if ! aws sts get-caller-identity --region "$REGION" >/dev/null 2>&1; then
  echo "ERROR: AWS credentials invalid/expired. Re-authenticate (e.g. 'aws sso login'" >&2
  echo "       or refresh your keys) and retry." >&2
  exit 1
fi

echo "=== Checkpoint tree for ${NAME} ==="
listing=$(aws s3 ls "${BUCKET}/${NAME}/" --recursive --region "$REGION" 2>/dev/null)
if [ -z "$listing" ]; then
  echo "  (no objects under ${BUCKET}/${NAME}/ — run may not have produced a checkpoint yet)"
  exit 2
fi
echo "$listing" | grep -E "chat_template\.jinja|config\.json|tokenizer_config\.json" | sort || true

# --- Collect chat_template.jinja keys (portable: macOS bash 3.2 has no mapfile) ---
keys=()
while IFS= read -r k; do
  [ -n "$k" ] && keys+=("$k")
done < <(echo "$listing" | awk '{print $4}' | grep "chat_template\.jinja$")

if [ "${#keys[@]}" -eq 0 ]; then
  echo
  echo "  No chat_template.jinja found in any checkpoint dir ❌"
  echo "  (Expected copy_chat_template_to_checkpoint to place one in each"
  echo "   epoch_hf_*/step_hf_* and the final dir.)"
  exit 3
fi

echo
echo "=== chat_template.jinja verdict per checkpoint dir ==="
fail=0
for key in "${keys[@]}"; do
  tmp=$(mktemp)
  if ! aws s3 cp "${BUCKET}/${key}" "$tmp" --region "$REGION" >/dev/null 2>&1; then
    echo "  ${key} -> ERROR downloading ❌"
    fail=1; rm -f "$tmp"; continue
  fi
  # -F = fixed strings: the markers contain '|' which grep treats as regex
  # alternation otherwise (matching '<' OR 'user' OR '>' — nearly everything).
  g=$(grep -cF '<|start_of_role|>' "$tmp"); g=${g:-0}
  t=$(grep -cF '<|user|>' "$tmp");         t=${t:-0}
  if [ "$g" -gt 0 ] && [ "$t" -eq 0 ]; then
    echo "  ${key} -> GRANITE ✅ (start_of_role=$g)"
  else
    echo "  ${key} -> NOT GRANITE ❌ (start_of_role=$g, tulu-user=$t)"
    fail=1
  fi
  rm -f "$tmp"
done

echo
if [ "$fail" -eq 0 ]; then
  echo "RESULT: PASS — all ${#keys[@]} checkpoint(s) carry the granite template ✅"
else
  echo "RESULT: FAIL — at least one checkpoint is missing/incorrect ❌"
  exit 4
fi
