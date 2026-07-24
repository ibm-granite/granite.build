-- Migration: Add model_source column to jobs table
-- Date: 2026-01-17
-- Description: Add support for tracking model source (HuggingFace or DMF)

USE autotune;

-- Add model_source column to jobs table
ALTER TABLE jobs
ADD COLUMN model_source VARCHAR(50) DEFAULT 'huggingface' NOT NULL
AFTER model;

-- Add comment for documentation
ALTER TABLE jobs
MODIFY COLUMN model_source VARCHAR(50) DEFAULT 'huggingface' NOT NULL
COMMENT 'Source of the model: huggingface or dmf';
