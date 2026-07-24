-- Migration: Add config_snapshot column to jobs table
-- Date: 2026-04-16
-- Description: Snapshot full configuration at job creation time to preserve
--              historical config data when configurations are later updated.

USE autotune;

ALTER TABLE jobs ADD COLUMN config_snapshot JSON DEFAULT NULL AFTER config_id;

-- Backfill existing jobs with current config data (best-effort).
-- For jobs whose configs have already been overwritten, this captures the
-- current version — imperfect but the best available data.
UPDATE jobs j
INNER JOIN configurations c ON j.config_id = c.id
SET j.config_snapshot = JSON_OBJECT(
    'name', c.name,
    'tuner_type', c.tuner_type,
    'rl_tuner_type', c.rl_tuner_type,
    'config_data', c.config_data
)
WHERE j.config_snapshot IS NULL;
