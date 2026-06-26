#!/usr/bin/env bash
# Cleanup the standalone SQLite metadata DB by deleting builds and everything
# linked to them (targets, steps, events, artifact-registry rows). Spaces are
# preserved. Underlying artifact files (COS/disk) are NOT touched - only the DB
# references. NO BACKUP is taken.
#
# By default only terminal builds (SUCCESS/FAILED/CANCELLED) are removed so a
# running build is never deleted out from under the server. Pass --all to wipe
# every build regardless of status.
#
# Usage:
#   scripts/cleanup-standalone-db.sh            # delete terminal builds
#   scripts/cleanup-standalone-db.sh --all      # delete ALL builds
#   scripts/cleanup-standalone-db.sh --dry-run  # show what would be deleted
#
# DB location: $GB_HOME_DIR/llmb-server.db (default ~/.granite.build/llmb-server.db)
set -euo pipefail

DB="${GB_HOME_DIR:-$HOME/.granite.build}/llmb-server.db"

ALL=false
DRY_RUN=false
for arg in "$@"; do
  case "$arg" in
    --all)     ALL=true ;;
    --dry-run) DRY_RUN=true ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [[ ! -f "$DB" ]]; then
  echo "DB not found: $DB" >&2
  exit 1
fi

if pgrep -f "gbserver standalone" >/dev/null 2>&1; then
  echo "WARNING: a 'gbserver standalone' process is running." >&2
  echo "Stop it first to avoid DB locks / stale in-memory state." >&2
  exit 1
fi

if $ALL; then
  WHERE="1=1"
else
  WHERE="status IN ('SUCCESS','FAILED','CANCELLED')"
fi

echo "DB: $DB"
echo "Targeting builds WHERE: $WHERE"
sqlite3 -header -column "$DB" \
  "SELECT status, COUNT(*) AS n FROM gb_builds WHERE $WHERE GROUP BY status;"

if $DRY_RUN; then
  echo "(dry-run) no changes made."
  exit 0
fi

sqlite3 "$DB" <<SQL
BEGIN;
CREATE TEMP TABLE _del AS SELECT uuid FROM gb_builds WHERE $WHERE;
DELETE FROM gb_events    WHERE build_id            IN (SELECT uuid FROM _del);
DELETE FROM gb_steps     WHERE build_id            IN (SELECT uuid FROM _del);
DELETE FROM gb_targets   WHERE build_id            IN (SELECT uuid FROM _del);
DELETE FROM gb_artifacts WHERE created_by_build_id IN (SELECT uuid FROM _del);
DELETE FROM gb_builds    WHERE uuid                IN (SELECT uuid FROM _del);
COMMIT;
VACUUM;
SQL

echo
echo "=== remaining row counts ==="
for t in gb_builds gb_targets gb_steps gb_artifacts gb_events gb_spaces; do
  printf "%-15s %s\n" "$t" "$(sqlite3 "$DB" "SELECT COUNT(*) FROM $t;")"
done
echo "Done."
