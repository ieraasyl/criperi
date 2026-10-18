import { LIVE_TIMEOUT_MS, type LiveEvent, type LiveWindow } from "@app/contracts";
import { AnalysisError, inferenceConfig } from "./inference.ts";
import type { Bindings } from "./env.ts";

export async function openLiveInference(env: Bindings, window: LiveWindow, signal: AbortSignal) {
  const config = inferenceConfig(env);
  let response: Response;
  try {
    response = await fetch(`${config.url}/api/v1/live`, {
      method: "POST",
      headers: {
        "X-API-Key": config.key,
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(window),
      signal: AbortSignal.any([signal, AbortSignal.timeout(LIVE_TIMEOUT_MS)]),
      redirect: "manual",
    });
  } catch {
    signal.throwIfAborted();
    throw new AnalysisError("INFERENCE_UNAVAILABLE", "Не удалось подключиться к сервису анализа.");
  }
  if (
    !response.ok ||
    !response.body ||
    !response.headers.get("Content-Type")?.startsWith("text/event-stream")
  ) {
    await response.body?.cancel().catch(() => {});
    throw new AnalysisError(
      response.status === 404 ? "LIVE_UNAVAILABLE" : "INFERENCE_UNAVAILABLE",
      response.status === 404
        ? "Анализ при воспроизведении пока недоступен."
        : response.status === 429
          ? "Сервис занят. Остановите воспроизведение и попробуйте позже."
          : "Сервис анализа временно недоступен.",
      response.status === 429 ? 429 : 503,
    );
  }
  return response.body;
}

export function validateLiveEvent(event: LiveEvent, window: LiveWindow, previousTime: number) {
  const first = window.frames[0]!.timestamp_sec;
  const last = window.frames.at(-1)!.timestamp_sec;
  const within = (time: number) => time >= first - 0.01 && time <= last + 0.01;
  if (event.type === "frame") {
    if (!within(event.timestamp_sec) || event.timestamp_sec <= previousTime)
      throw new Error("Live frame is outside its window or out of order");
  }
  if (event.type === "result") {
    if (
      Math.abs(event.from_sec - first) > 0.01 ||
      Math.abs(event.to_sec - last) > 0.01 ||
      event.violations.some(
        (finding) =>
          !within(finding.timecode_sec) ||
          finding.evidence_frames.some((frame) => !within(frame.timestamp_sec)),
      )
    )
      throw new Error("Live result is outside its window");
  }
}
