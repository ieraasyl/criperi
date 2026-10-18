import { z } from "zod";

export const createNoteSchema = z.object({
  title: z.string().trim().min(1).max(200),
});

export type CreateNote = z.infer<typeof createNoteSchema>;

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
export const reviewJobSchema = jobSchema.and(z.object({ expires_at: z.iso.datetime() }));
export type ReviewJob = z.infer<typeof reviewJobSchema>;

export const LIVE_WINDOW_FRAMES = 12;
export const LIVE_STRIDE_FRAMES = 4;
export const LIVE_REQUEST_LIMIT_BYTES = 4 * 1024 * 1024;
export const LIVE_IMAGE_LIMIT = 350_000;
export const LIVE_TIMEOUT_MS = 180_000;

const liveEvidenceSchema = evidenceSchema.extend({
  image_base64: evidenceSchema.shape.image_base64.refine(
    (image) => image.length <= LIVE_IMAGE_LIMIT,
  ),
});

export const liveWindowSchema = z
  .object({ frames: z.array(liveEvidenceSchema).min(2).max(LIVE_WINDOW_FRAMES) })
  .superRefine(({ frames }, ctx) => {
    for (let i = 1; i < frames.length; i++) {
      const gap = frames[i]!.timestamp_sec - frames[i - 1]!.timestamp_sec;
      if (gap < 0.5 || gap > 1.75)
        ctx.addIssue({
          code: "custom",
          message: "Expected consecutive sampled frames",
          path: ["frames", i],
        });
    }
  });

export const liveEventSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("stage"), stage: z.enum(["layout", "detecting", "verifying"]) }),
  liveEvidenceSchema.extend({
    type: z.literal("frame"),
    people_count: z.number().int().nonnegative().max(100),
    objects_count: z.number().int().nonnegative().max(1000),
  }),
  z.object({
    type: z.literal("result"),
    from_sec: z.number().finite().nonnegative(),
    to_sec: z.number().finite().nonnegative(),
    violations: z
      .array(violationSchema.extend({ evidence_frames: z.array(liveEvidenceSchema).length(1) }))
      .max(21),
  }),
  z.object({ type: z.literal("error"), error: apiErrorSchema.shape.error }),
]);

export type LiveWindow = z.infer<typeof liveWindowSchema>;
export type LiveEvent = z.infer<typeof liveEventSchema>;
export type LiveFrame = Extract<LiveEvent, { type: "frame" }>;

// The service sends one JSON data line per SSE message. Bound both a message and
// the full window response, including when a peer never sends a line break.
export async function* readLiveEvents(body: ReadableStream<Uint8Array>): AsyncGenerator<LiveEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let pending = "";
  let bytes = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.byteLength;
      if (bytes > 24 * 1024 * 1024) throw new Error("Live response exceeded its limit");
      pending += decoder.decode(value, { stream: true });
      let end: number;
      while ((end = pending.indexOf("\n")) !== -1) {
        const line = pending.slice(0, end).replace(/\r$/, "");
        pending = pending.slice(end + 1);
        if (line.length > 8 * 1024 * 1024) throw new Error("Live event exceeded its limit");
        if (!line || line.startsWith(":")) continue;
        if (!line.startsWith("data:")) throw new Error("Invalid live stream");
        const event = liveEventSchema.parse(JSON.parse(line.slice(5)));
        yield event;
        if (event.type === "result" || event.type === "error") return;
      }
      if (pending.length > 8 * 1024 * 1024) throw new Error("Live event exceeded its limit");
    }
    throw new Error("Live stream ended without a result");
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
