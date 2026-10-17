PRAGMA foreign_keys=OFF;--> statement-breakpoint
CREATE TABLE `__new_analysis_jobs` (
	`id` text PRIMARY KEY NOT NULL,
	`owner` text NOT NULL,
	`inference_job_id` text NOT NULL,
	`created_at` integer NOT NULL,
	`expires_at` integer NOT NULL
);
--> statement-breakpoint
INSERT INTO `__new_analysis_jobs`("id", "owner", "inference_job_id", "created_at", "expires_at") SELECT "id", "owner", "inference_job_id", "created_at", "expires_at" FROM `analysis_jobs`;--> statement-breakpoint
DROP TABLE `analysis_jobs`;--> statement-breakpoint
ALTER TABLE `__new_analysis_jobs` RENAME TO `analysis_jobs`;--> statement-breakpoint
PRAGMA foreign_keys=ON;--> statement-breakpoint
CREATE INDEX `analysis_jobs_owner` ON `analysis_jobs` (`owner`);--> statement-breakpoint
CREATE INDEX `analysis_jobs_expiry` ON `analysis_jobs` (`expires_at`);