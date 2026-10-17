import { Hono, type Context } from "hono";
import { createMiddleware } from "hono/factory";
import { createDb, analysisJobs, and, eq } from "@app/db";
import { JOB_LIFETIME_MS, UPLOAD_LIMIT_BYTES, jobIdSchema, jobSchema } from "@app/contracts";
import { requireSession, type AuthenticatedEnv } from "../middleware/session.ts";
import {
  AnalysisError,
  getInferenceJob,
  inferenceConfig,
  submitInferenceJob,
} from "../inference.ts";

type AnalysisEnv = AuthenticatedEnv & { Variables: { owner: string } };

const requireKey = createMiddleware<AnalysisEnv>(async (c, next) => {
  const supplied = c.req.header("X-API-Key") ?? "";
  const configured = c.env.EVALUATION_API_KEY;
  if (!configured) throw new AnalysisError("EVALUATION_UNAVAILABLE", "Доступ к API не настроен.");
  const digest = (value: string) =>
    crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  const [left, right] = await Promise.all([digest(supplied), digest(configured)]);
  const a = new Uint8Array(left),
    b = new Uint8Array(right);
  let difference = 0;
  for (let i = 0; i < a.length; i++) difference |= a[i]! ^ b[i]!;
  if (!supplied || difference !== 0)
    throw new AnalysisError("UNAUTHORIZED", "Неверный ключ API.", 401);
  c.set("owner", `api:${c.env.EVALUATION_OWNER_ID ?? "evaluation"}`);
  await next();
});

async function submit(c: Context<AnalysisEnv>) {
  const startedAt = Date.now();
  inferenceConfig(c.env);
  if (
    !c.req.header("Content-Type")?.toLowerCase().startsWith("multipart/form-data;") ||
    !c.req.raw.body
  ) {
    throw new AnalysisError("INVALID_UPLOAD", "Выберите MOV-файл в поле file.", 400);
  }
  const configured = Number(c.env.UPLOAD_LIMIT_BYTES ?? UPLOAD_LIMIT_BYTES);
  const limit =
    Number.isSafeInteger(configured) && configured > 0
      ? Math.min(configured, UPLOAD_LIMIT_BYTES)
      : UPLOAD_LIMIT_BYTES;
  const length = Number(c.req.header("Content-Length"));
  if (length > limit)
    throw new AnalysisError("UPLOAD_TOO_LARGE", "Файл превышает допустимый размер.", 413);
  const id = crypto.randomUUID();
  const inferenceJobId = await submitInferenceJob(c.env, c.req.raw, limit);
  try {
    await createDb(c.env.DB)
      .insert(analysisJobs)
      .values({
        id,
        owner: c.get("owner"),
        inferenceJobId,
        createdAt: startedAt,
        expiresAt: startedAt + JOB_LIFETIME_MS,
      });
  } catch {
    throw new AnalysisError(
      "SUBMISSION_UNCONFIRMED",
      "Видео принято, но не удалось сохранить идентификатор анализа. Повторная отправка отключена.",
    );
  }
  return c.json({ job_id: id }, 202);
}

async function getJob(c: Context<AnalysisEnv>) {
  const parsed = jobIdSchema.safeParse(c.req.param("job_id"));
  if (!parsed.success) throw new AnalysisError("NOT_FOUND", "Анализ не найден.", 404);
  const [record] = await createDb(c.env.DB)
    .select()
    .from(analysisJobs)
    .where(and(eq(analysisJobs.id, parsed.data), eq(analysisJobs.owner, c.get("owner"))))
    .limit(1);
  if (!record) throw new AnalysisError("NOT_FOUND", "Анализ не найден.", 404);
  if (record.expiresAt <= Date.now())
    throw new AnalysisError("EXPIRED", "Срок хранения анализа истёк.", 410);
  const upstream = await getInferenceJob(c.env, record.inferenceJobId, c.req.raw.signal);
  const job = { ...upstream, job_id: record.id };
  if (c.get("owner").startsWith("api:")) return c.json(jobSchema.parse(job));
  return c.json({ ...job, expires_at: new Date(record.expiresAt).toISOString() });
}

export const evaluationRoutes = new Hono<AnalysisEnv>()
  .use("*", requireKey)
  .post("/analyze", submit)
  .get("/jobs/:job_id", getJob);

export const analysisRoutes = new Hono<AnalysisEnv>()
  .use("*", requireSession)
  .use("*", async (c, next) => {
    c.set("owner", `user:${c.get("session").user.id}`);
    await next();
  })
  .post("/", submit)
  .get("/:job_id", getJob);
