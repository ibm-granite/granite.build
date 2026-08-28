CREATE DATABASE IF NOT EXISTS autotune;
USE autotune;

CREATE TABLE users (
    `id` VARCHAR(36) DEFAULT(uuid()) PRIMARY KEY,
    `email` VARCHAR(255) NOT NULL UNIQUE,
    `role` VARCHAR(50) DEFAULT 'user',
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE configurations (
    `id` VARCHAR(36) DEFAULT(uuid()) PRIMARY KEY,
    `user_id` VARCHAR(255) NOT NULL,
    `name` VARCHAR(255) NOT NULL,
    `tuner_type` VARCHAR(50),
    `rl_tuner_type` VARCHAR(50) DEFAULT NULL,
    `config_data` JSON,
	`created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY `uq_configurations_user_name` (`user_id`, `name`),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- datasets table
CREATE TABLE datasets (
    `id` VARCHAR(36) DEFAULT (uuid()) PRIMARY KEY,
    `user_id` VARCHAR(255) NOT NULL,
    `name` VARCHAR(255) NOT NULL,
    `description` TEXT NOT NULL,
    `train_file` VARCHAR(255) GENERATED ALWAYS AS (CONCAT(name, '_train')) STORED,
    `train_records` INT DEFAULT NULL,
    `train_file_size` INT DEFAULT NULL,
    `validation_file` VARCHAR(255) GENERATED ALWAYS AS (CONCAT(name, '_validation')) STORED,
    `validation_records` INT DEFAULT NULL,
    `validation_file_size` INT DEFAULT NULL,
    `data_format` VARCHAR(10) NOT NULL DEFAULT 'jsonl',
    `artifact_id` VARCHAR(36) DEFAULT NULL,
    `artifact_url` TEXT DEFAULT NULL,
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY `uq_datasets_user_name` (`user_id`, `name`),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Jobs Table
CREATE TABLE jobs (
    `id` VARCHAR(36) DEFAULT(uuid()) PRIMARY KEY,
    `user_id` VARCHAR(255) NOT NULL,
    `status` ENUM('PENDING', 'RUNNING', 'PAUSED', 'TERMINATED', 'ERROR', 'COMPLETED') NOT NULL DEFAULT 'PENDING',
    `seed` INT DEFAULT 42,
    `config_id` CHAR(36) NOT NULL,
    `config_snapshot` JSON DEFAULT NULL,
    `dataset_id` CHAR(36) NOT NULL,
    `model` VARCHAR(255) NOT NULL,
    `model_source` VARCHAR(50) DEFAULT 'huggingface' NOT NULL,
    `experiment_name` VARCHAR(255) NOT NULL,
    `tuning_type` VARCHAR(100) DEFAULT NULL,
    `precision` VARCHAR(50) NOT NULL,
    `ray_address` VARCHAR(50) DEFAULT NULL,
    `cleanup` BOOLEAN DEFAULT TRUE,
    `autotune` BOOLEAN DEFAULT TRUE,
    `output_artifacts` JSON DEFAULT NULL,
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (config_id) REFERENCES configurations(id) ON DELETE RESTRICT,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE RESTRICT
);

-- Trials Table
CREATE TABLE trials (
    `id` VARCHAR(16) PRIMARY KEY,
    `job_id` CHAR(36) NOT NULL,
    `status` ENUM('PENDING', 'RUNNING', 'PAUSED', 'TERMINATED', 'ERROR', 'COMPLETED') NOT NULL DEFAULT 'PENDING',
    `config` JSON,
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

-- log entries table
CREATE TABLE log_entries (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `job_id` CHAR(36) NOT NULL,
    `trial_id` CHAR(36) DEFAULT NULL,
    `level` VARCHAR(50),
    `filename` VARCHAR(255),
    `message` MEDIUMTEXT,
    `iteration` INT DEFAULT NULL,  
    `epoch` FLOAT DEFAULT NULL, 
    `timestamp` DATETIME,
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

CREATE TABLE results(
    `id` VARCHAR(36) DEFAULT (uuid()) PRIMARY KEY,
    `job_id` varchar(36) NOT NULL,
    `trial_id` varchar(16) NOT NULL UNIQUE,
    `metric` VARCHAR(255) NOT NULL,
    `metrics` JSON,
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
    FOREIGN KEY (trial_id) REFERENCES trials(id) ON DELETE CASCADE
);

CREATE TABLE gb_tasks (
    `id` CHAR(36) NOT NULL DEFAULT (UUID()) PRIMARY KEY,
    `job_id` CHAR(36) NOT NULL,
    `build_id` CHAR(36) DEFAULT NULL,
    `status` ENUM('PENDING', 'RUNNING', 'PAUSED', 'TERMINATED', 'ERROR', 'COMPLETED') NOT NULL DEFAULT 'PENDING',
    `type` ENUM('RITS', 'TUNING', 'DOWNLOAD') NOT NULL,
    `pr_url` TEXT DEFAULT NULL,
    `artifact_id` CHAR(36) DEFAULT NULL,
    `artifact_uri` TEXT DEFAULT NULL,
    `build_status` JSON DEFAULT NULL,
    `started_at` VARCHAR(255) DEFAULT NULL,
    `updated_at` VARCHAR(255) DEFAULT NULL,
    `rits_url` TEXT DEFAULT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

CREATE VIEW autotunex_jobs AS SELECT
    j.*,
    u.email AS user,
    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(j.config_snapshot, '$.rl_tuner_type')), c.rl_tuner_type) AS rl_tuner_type,
    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(j.config_snapshot, '$.name')), c.name) AS config_name,
    d.name AS dataset,
    COUNT(DISTINCT t.id) AS num_trials,
    gt.id AS task_id,
    gt.build_id,
    gt.status AS task_status,
    gt.type AS task_type,
    gt.pr_url AS github_pr_url,
    gt.artifact_id,
    gt.artifact_uri,
    gt.build_status,
    gt.started_at AS task_started_at,
    gt.updated_at AS task_updated_at,
    gt.rits_url
FROM jobs j
INNER JOIN users u ON j.user_id = u.id
INNER JOIN configurations c ON j.config_id = c.id
INNER JOIN datasets d ON j.dataset_id = d.id
LEFT JOIN trials t ON j.id = t.job_id
LEFT JOIN gb_tasks gt ON j.id = gt.job_id
GROUP BY j.id, gt.id, gt.build_id, gt.status, gt.pr_url, gt.artifact_id, 
         gt.artifact_uri, gt.build_status, gt.started_at, gt.updated_at, gt.rits_url
ORDER BY j.created_at DESC;