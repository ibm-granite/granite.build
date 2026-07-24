-- 005: Add DOWNLOAD task type to gb_tasks enum
ALTER TABLE `gb_tasks` MODIFY COLUMN `type` ENUM('RITS', 'TUNING', 'DOWNLOAD') NOT NULL;
