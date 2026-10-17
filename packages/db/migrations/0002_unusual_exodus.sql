CREATE TABLE `analysis_jobs` (
	`id` text PRIMARY KEY NOT NULL,
	`owner` text NOT NULL,
	`created_at` integer NOT NULL,
	`expires_at` integer NOT NULL
);
--> statement-breakpoint
CREATE INDEX `analysis_jobs_owner` ON `analysis_jobs` (`owner`);--> statement-breakpoint
CREATE INDEX `analysis_jobs_expiry` ON `analysis_jobs` (`expires_at`);