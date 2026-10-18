import { sqliteTable, text, integer, real, index, primaryKey } from "drizzle-orm/sqlite-core";

// Anonymous auth uses these tables; credential accounts and verification are not exposed.
export { user, session, rateLimit } from "./auth-schema.ts";

export const analysisJobs = sqliteTable(
  "analysis_jobs",
  {
    id: text("id").primaryKey(),
    owner: text("owner").notNull(),
    inferenceJobId: text("inference_job_id").notNull(),
    createdAt: integer("created_at").notNull(),
    expiresAt: integer("expires_at").notNull(),
  },
  (table) => [
    index("analysis_jobs_owner").on(table.owner),
    index("analysis_jobs_expiry").on(table.expiresAt),
  ],
);

export const analysisReviews = sqliteTable(
  "analysis_reviews",
  {
    jobId: text("job_id")
      .notNull()
      .references(() => analysisJobs.id, { onDelete: "cascade" }),
    eventIndex: integer("event_index").notNull(),
    eventType: text("event_type").notNull(),
    timecodeSec: real("timecode_sec").notNull(),
    status: text("status", { enum: ["confirmed", "rejected", "needs_review"] }).notNull(),
    comment: text("comment").notNull().default(""),
    updatedAt: integer("updated_at", { mode: "timestamp_ms" }).notNull(),
  },
  (table) => [primaryKey({ columns: [table.jobId, table.eventIndex] })],
);
