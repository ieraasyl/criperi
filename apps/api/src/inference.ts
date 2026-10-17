import { upstreamJobSchema, type UpstreamJob } from "@app/contracts";
import type { Bindings } from "./env.ts";

export class AnalysisError extends Error {
  jobId?: string;
  readonly code: string;
  readonly status: 400 | 401 | 404 | 410 | 413 | 415 | 422 | 429 | 502 | 503;
  constructor(code: string, message: string, status: AnalysisError["status"] = 503) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

export function inferenceConfig(env: Bindings) {
  if (!env.INFERENCE_URL || !env.INFERENCE_API_KEY) {
    throw new AnalysisError("INFERENCE_UNAVAILABLE", "Сервис анализа не настроен.");
  }
  let url: URL;
  try {
    url = new URL(env.INFERENCE_URL);
  } catch {
    throw new AnalysisError("INFERENCE_UNAVAILABLE", "Некорректный адрес сервиса анализа.");
  }
  if (
    (url.protocol !== "https:" &&
      !(url.protocol === "http:" && ["127.0.0.1", "localhost"].includes(url.hostname))) ||
    url.username ||
    url.password ||
    url.pathname !== "/" ||
    url.search ||
    url.hash
  ) {
    throw new AnalysisError("INFERENCE_UNAVAILABLE", "Некорректный адрес сервиса анализа.");
  }
  return { url: url.origin, key: env.INFERENCE_API_KEY };
}

async function readJson(response: Response): Promise<unknown> {
  // Bound enriched results before parsing them in a Worker isolate.
  if (!response.body)
    throw new AnalysisError("INVALID_RESPONSE", "Сервис вернул пустой ответ.", 502);
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > 16 * 1024 * 1024)
        throw new AnalysisError("INVALID_RESPONSE", "Ответ сервиса слишком большой.", 502);
      chunks.push(value);
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  try {
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    throw new AnalysisError("INVALID_RESPONSE", "Сервис вернул некорректный ответ.", 502);
  }
}

export async function getInferenceJob(
  env: Bindings,
  id: string,
  signal?: AbortSignal,
): Promise<UpstreamJob> {
  const config = inferenceConfig(env);
  let response: Response | undefined;
  try {
    response = await fetch(`${config.url}/api/v1/jobs/${id}`, {
      headers: { "X-API-Key": config.key },
      signal: AbortSignal.any([AbortSignal.timeout(10_000), ...(signal ? [signal] : [])]),
      redirect: "manual",
    });
    if (response.status === 404) throw new AnalysisError("NOT_FOUND", "Анализ не найден.", 404);
    if (response.status === 410)
      throw new AnalysisError("EXPIRED", "Срок хранения анализа истёк.", 410);
    if (!response.ok)
      throw new AnalysisError("INFERENCE_UNAVAILABLE", "Сервис анализа временно недоступен.");
    const parsed = upstreamJobSchema.safeParse(await readJson(response));
    if (!parsed.success || parsed.data.job_id !== id)
      throw new AnalysisError("INVALID_RESPONSE", "Сервис вернул некорректный результат.", 502);
    return parsed.data;
  } catch (error) {
    signal?.throwIfAborted();
    if (error instanceof AnalysisError) throw error;
    throw new AnalysisError("INFERENCE_UNAVAILABLE", "Не удалось связаться с сервисом анализа.");
  } finally {
    await response?.body?.cancel().catch(() => {});
  }
}

export async function submitInferenceJob(
  env: Bindings,
  id: string,
  request: Request,
  limit: number,
  startedAt: number,
) {
  const config = inferenceConfig(env);
  let bytes = 0;
  let exceeded = false;
  const stream = request.body?.pipeThrough(
    new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, controller) {
        bytes += chunk.byteLength;
        if (bytes > limit) {
          exceeded = true;
          controller.error(new Error("Upload limit exceeded"));
        } else controller.enqueue(chunk);
      },
    }),
  );
  let response: Response | undefined;
  try {
    response = await fetch(`${config.url}/api/v1/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": request.headers.get("Content-Type")!,
        "X-API-Key": config.key,
        "X-Job-ID": id,
        "X-Request-Started-At": String(startedAt),
      },
      body: stream,
      signal: AbortSignal.any([request.signal, AbortSignal.timeout(90_000)]),
      redirect: "manual",
    });
    if (exceeded)
      throw new AnalysisError("UPLOAD_TOO_LARGE", "Файл превышает допустимый размер.", 413);
    if (response.status === 202) {
      // We allocated the identifier. A malformed acknowledgement never triggers another write.
      await response.body?.cancel().catch(() => {});
      return;
    }
    if ([400, 413, 415, 422, 429].includes(response.status)) {
      const status = response.status as 400 | 413 | 415 | 422 | 429;
      await response.body?.cancel().catch(() => {});
      throw new AnalysisError(
        status === 429 ? "CAPACITY_EXCEEDED" : "INVALID_UPLOAD",
        status === 429
          ? "Сервис занят. Попробуйте позже."
          : "Не удалось принять видео. Проверьте формат и размер файла.",
        status,
      );
    }
  } catch (error) {
    if (exceeded)
      throw new AnalysisError("UPLOAD_TOO_LARGE", "Файл превышает допустимый размер.", 413);
    if (error instanceof AnalysisError) throw error;
  } finally {
    await response?.body?.cancel().catch(() => {});
  }
  // The acknowledgement may have been lost after a successful durable write.
  try {
    await getInferenceJob(env, id);
    return;
  } catch {
    /* Do not resubmit. */
  }
  throw new AnalysisError(
    "SUBMISSION_UNCONFIRMED",
    `Не удалось подтвердить приём видео. Идентификатор: ${id}. Проверьте страницу результата.`,
    503,
  );
}
