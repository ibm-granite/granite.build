-- Migration: Add rl_tuner_type column to configurations table
-- Date: 2026-02-17
-- Description: Add support for tracking RL algorithm type (DPO, ORPO, KTO, PPO, GRPO, DAPO)

USE autotune;

-- Add rl_tuner_type column to configurations table
ALTER TABLE configurations
ADD COLUMN rl_tuner_type VARCHAR(50) DEFAULT NULL
AFTER tuner_type;

-- Add comment for documentation
ALTER TABLE configurations
MODIFY COLUMN rl_tuner_type VARCHAR(50) DEFAULT NULL
COMMENT 'RL algorithm type: dpo, orpo, kto, ppo, grpo, dapo, or NULL for non-RL configs';
