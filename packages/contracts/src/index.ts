import { z } from "zod";

export const apiErrorSchema = z.object({
  error: z.object({
    code: z.string(),
    message: z.string(),
    issues: z.array(z.object({ path: z.array(z.string()), message: z.string() })).optional(),
  }),
});

export type ApiError = z.infer<typeof apiErrorSchema>;

export const UPLOAD_LIMIT_BYTES = 64 * 1024 * 1024;
export const JOB_LIFETIME_MS = 24 * 60 * 60 * 1000;
export const jobIdSchema = z.uuid();
export const submissionSchema = z.object({ job_id: jobIdSchema });
export const inferenceJobIdSchema = z
  .string()
  .min(1)
  .max(128)
  .regex(/^[A-Za-z0-9_-]+$/);
export const inferenceSubmissionSchema = z.object({
  job_id: inferenceJobIdSchema,
  status: z.literal("processing"),
});
export const evidenceSchema = z.object({
  timestamp_sec: z.number().finite().nonnegative(),
  image_base64: z
    .string()
    .max(4_000_000)
    .refine((value) => {
      if (!/^[A-Za-z0-9+/]+={0,2}$/.test(value) || value.length % 4 !== 0) return false;
      try {
        const bytes = atob(value);
        return bytes.startsWith("\xff\xd8\xff") && bytes.endsWith("\xff\xd9");
      } catch {
        return false;
      }
    }, "Expected base64 JPEG bytes"),
});
export const violationSchema = z.object({
  type: z.string().min(1),
  timecode_sec: z.number().finite().nonnegative(),
  description: z.string().min(1),
  evidence_frames: z.array(evidenceSchema).min(1).max(3),
});
const processingSchema = z.object({ job_id: jobIdSchema, status: z.literal("processing") });
const failedSchema = z.object({
  job_id: jobIdSchema,
  status: z.literal("failed"),
  error: apiErrorSchema.shape.error,
});
const doneSchema = z.object({
  job_id: jobIdSchema,
  status: z.literal("done"),
  verdict: z.enum(["violation", "no_violation"]),
  violations: z.array(violationSchema),
});
function verdictMatches(job: z.infer<typeof doneSchema> | { status: "processing" | "failed" }) {
  return job.status !== "done" || (job.verdict === "violation") === job.violations.length > 0;
}
export const jobSchema = z
  .discriminatedUnion("status", [processingSchema, failedSchema, doneSchema])
  .refine(verdictMatches, "Verdict must match the event list");
export const upstreamJobSchema = z
  .discriminatedUnion("status", [
    processingSchema.extend({ job_id: inferenceJobIdSchema }),
    failedSchema.extend({
      job_id: inferenceJobIdSchema,
      error: z.string().transform(() => ({
        code: "ANALYSIS_FAILED",
        message: "Сервис не смог завершить анализ.",
      })),
    }),
    doneSchema.extend({ job_id: inferenceJobIdSchema }),
  ])
  .refine(verdictMatches, "Verdict must match the event list");
export type Job = z.infer<typeof jobSchema>;
export type UpstreamJob = z.infer<typeof upstreamJobSchema>;
export const reviewDecisionSchema = z.enum(["confirmed", "rejected", "needs_review"]);
export const saveReviewSchema = z.object({
  status: reviewDecisionSchema,
  comment: z.string().trim().max(4000).default(""),
});
export const eventReviewSchema = saveReviewSchema.extend({
  event_index: z.number().int().nonnegative(),
  updated_at: z.iso.datetime(),
});
export type EventReview = z.infer<typeof eventReviewSchema>;
export type SaveReview = z.infer<typeof saveReviewSchema>;
export const reviewJobSchema = jobSchema.and(
  z.object({
    expires_at: z.iso.datetime(),
    reviews: z.array(eventReviewSchema).default([]),
  }),
);
export type ReviewJob = z.infer<typeof reviewJobSchema>;
