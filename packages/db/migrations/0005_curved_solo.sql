CREATE TABLE `analysis_reviews` (
	`job_id` text NOT NULL,
	`event_index` integer NOT NULL,
	`event_type` text NOT NULL,
	`timecode_sec` real NOT NULL,
	`status` text NOT NULL,
	`comment` text DEFAULT '' NOT NULL,
	`updated_at` integer NOT NULL,
	PRIMARY KEY(`job_id`, `event_index`),
	FOREIGN KEY (`job_id`) REFERENCES `analysis_jobs`(`id`) ON UPDATE no action ON DELETE cascade
);
