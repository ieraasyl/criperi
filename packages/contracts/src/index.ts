import { z } from "zod";

export const createNoteSchema = z.object({
  title: z.string().trim().min(1).max(200),
});

export type CreateNote = z.infer<typeof createNoteSchema>;

export const apiErrorSchema = z.object({
  error: z.object({
    code: z.string(),
    message: z.string(),
    job_id: z.uuid().optional(),
    issues: z.array(z.object({ path: z.array(z.string()), message: z.string() })).optional(),
  }),
});

export type ApiError = z.infer<typeof apiErrorSchema>;

export const UPLOAD_LIMIT_BYTES = 64 * 1024 * 1024;
export const JOB_LIFETIME_MS = 24 * 60 * 60 * 1000;
export const jobIdSchema = z.uuid();
export const submissionSchema = z.object({ job_id: jobIdSchema });
export const ruleSchema = z.object({
  id: z.string().min(1),
  document: z.string().min(1),
  point: z.string(),
  url: z.url().refine((url) => url.startsWith("https://")),
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
export const jobSchema = z
  .discriminatedUnion("status", [processingSchema, failedSchema, doneSchema])
  .refine(
    (job) => job.status !== "done" || (job.verdict === "violation") === job.violations.length > 0,
    "Verdict must match the event list",
  );
export const upstreamJobSchema = z
  .discriminatedUnion("status", [
    processingSchema,
    failedSchema,
    doneSchema.extend({
      duration_sec: z.number().finite().positive(),
      violations: z.array(violationSchema.extend({ rule: ruleSchema })),
    }),
  ])
  .superRefine((job, ctx) => {
    if (job.status !== "done") return;
    if ((job.verdict === "violation") !== job.violations.length > 0) {
      ctx.addIssue({ code: "custom", message: "Verdict must match the event list" });
    }
    for (const event of job.violations) {
      if (
        event.timecode_sec >= job.duration_sec ||
        event.evidence_frames.some((frame) => frame.timestamp_sec >= job.duration_sec)
      ) {
        ctx.addIssue({ code: "custom", message: "Timestamp outside video" });
      }
    }
  });
export type Job = z.infer<typeof jobSchema>;
export type UpstreamJob = z.infer<typeof upstreamJobSchema>;
export type Rule = z.infer<typeof ruleSchema>;
export const reviewJobSchema = upstreamJobSchema.and(z.object({ expires_at: z.iso.datetime() }));
export type ReviewJob = z.infer<typeof reviewJobSchema>;
