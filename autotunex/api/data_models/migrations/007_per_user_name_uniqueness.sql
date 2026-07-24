-- Migration: Make configuration/dataset names unique per user instead of globally.
-- Date: 2026-05-04
-- Description: Drops the global UNIQUE constraint on configurations.name and
--              datasets.name and replaces it with a composite UNIQUE (user_id, name),
--              so two different users can own resources with the same name while
--              preserving uniqueness within a single user.

USE autotune;

-- Pre-flight: surface any existing cross-user name collisions before relaxing
-- the constraint. If any row appears in these result sets, resolve manually
-- (rename one side) before proceeding — they are technically impossible under
-- the old global UNIQUE constraint, but included defensively.
SELECT 'configurations' AS table_name, name, COUNT(*) AS dup_count
FROM configurations GROUP BY name HAVING dup_count > 1;
SELECT 'datasets' AS table_name, name, COUNT(*) AS dup_count
FROM datasets GROUP BY name HAVING dup_count > 1;

-- configurations: swap global UNIQUE(name) for UNIQUE(user_id, name)
ALTER TABLE configurations DROP INDEX `name`;
ALTER TABLE configurations ADD CONSTRAINT `uq_configurations_user_name` UNIQUE (`user_id`, `name`);

-- datasets: swap global UNIQUE(name) for UNIQUE(user_id, name)
ALTER TABLE datasets DROP INDEX `name`;
ALTER TABLE datasets ADD CONSTRAINT `uq_datasets_user_name` UNIQUE (`user_id`, `name`);
