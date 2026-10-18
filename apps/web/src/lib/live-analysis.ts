import {
  apiErrorSchema,
  LIVE_STRIDE_FRAMES,
  LIVE_TIMEOUT_MS,
  LIVE_WINDOW_FRAMES,
  readLiveEvents,
  type LiveEvent,
  type LiveWindow,
} from "@app/contracts";
import { api } from "./api";
import { ApiRequestError } from "./request";

export type LiveFinding = Extract<LiveEvent, { type: "result" }>["violations"][number] & {
  last_seen_sec: number;
};

export function mergeLiveFindings(
  current: LiveFinding[],
  result: Extract<LiveEvent, { type: "result" }>,
) {
  const next = [...current];
  for (const finding of result.violations) {
    const index = next.findIndex(
      (item) =>
        item.type === finding.type &&
        finding.timecode_sec >= item.timecode_sec - 10 &&
        finding.timecode_sec <= item.last_seen_sec + 10,
    );
    if (index === -1) next.push({ ...finding, last_seen_sec: result.to_sec });
    else next[index] = { ...next[index]!, last_seen_sec: result.to_sec };
  }
  return next.sort((a, b) => a.timecode_sec - b.timecode_sec);
}

export async function analyzeLiveWindow(
  window: LiveWindow,
  signal: AbortSignal,
  onEvent: (event: LiveEvent) => void,
) {
  const response = await api.live.$post(
    { json: window },
    {
      headers: { Accept: "text/event-stream" },
      init: {
        signal: AbortSignal.any([signal, AbortSignal.timeout(LIVE_TIMEOUT_MS)]),
      },
    },
  );
  signal.throwIfAborted();
  if (!response.ok) {
    const parsed = apiErrorSchema.safeParse(await response.json().catch(() => null));
    throw new ApiRequestError(
      response.status === 401
        ? "Сеанс завершён. Войдите снова."
        : parsed.success
          ? parsed.data.error.message
          : "Не удалось начать анализ.",
      response.status,
      parsed.success ? parsed.data.error.code : "LIVE_FAILED",
    );
  }
  if (!response.body) throw new Error("Сервис вернул пустой ответ.");
  for await (const event of readLiveEvents(response.body)) {
    signal.throwIfAborted();
    if (event.type === "error")
      throw new ApiRequestError(event.error.message, 200, event.error.code);
    onEvent(event);
  }
}

type Callbacks = {
  analyze?: typeof analyzeLiveWindow;
  onEvent: (event: LiveEvent) => void;
  onBuffering: (buffering: boolean) => void;
  onBusy: (busy: boolean) => void;
  onComplete: () => void;
  onError: (message: string) => void;
};

// One request at a time and at most twenty buffered frames, including eight
// frames of context. Pause playback before the buffer can lose coverage.
export class LiveAnalysisSession {
  private readonly controller = new AbortController();
  private frames: LiveWindow["frames"] = [];
  private through = -1;
  private busy = false;
  private active = true;
  private ended = false;
  private buffering = false;

  private readonly callbacks: Callbacks;

  constructor(callbacks: Callbacks) {
    this.callbacks = callbacks;
  }

  add(frame: LiveWindow["frames"][number]) {
    if (!this.active || this.ended) return;
    if (this.frames.length >= LIVE_WINDOW_FRAMES + 8) {
      this.fail("Очередь кадров заполнена. Запустите анализ снова с текущего момента.");
      return;
    }
    const last = this.frames.at(-1);
    if (last && frame.timestamp_sec - last.timestamp_sec > 1.75) {
      this.fail(
        "Между кадрами возник пропуск. Воспроизведение остановлено. Запустите анализ снова с этого момента.",
      );
      return;
    }
    if (last && frame.timestamp_sec <= last.timestamp_sec) return;
    this.frames.push(frame);
    if (this.frames.length >= LIVE_WINDOW_FRAMES + 8) this.setBuffering(true);
    void this.drain();
  }

  finish() {
    this.ended = true;
    void this.drain();
  }

  stop() {
    this.active = false;
    this.controller.abort();
    this.frames = [];
  }

  private setBuffering(buffering: boolean) {
    if (this.buffering === buffering) return;
    this.buffering = buffering;
    this.callbacks.onBuffering(buffering);
  }

  private fail(message: string) {
    this.stop();
    this.callbacks.onBusy(false);
    this.callbacks.onError(message);
  }

  private async drain() {
    if (!this.active || this.busy) return;
    const pending = this.frames.filter((frame) => frame.timestamp_sec > this.through);
    if (!pending.length) {
      if (this.ended) {
        this.active = false;
        this.callbacks.onComplete();
      }
      return;
    }
    if (pending.length < LIVE_STRIDE_FRAMES && !this.ended) return;
    const context = this.frames.filter((frame) => frame.timestamp_sec <= this.through).slice(-8);
    const frames = [...context, ...pending.slice(0, LIVE_STRIDE_FRAMES)];
    if (frames.length < 2) {
      this.fail("Для анализа нужно хотя бы два кадра. Выберите более длинное видео.");
      return;
    }
    this.busy = true;
    this.callbacks.onBusy(true);
    try {
      await (this.callbacks.analyze ?? analyzeLiveWindow)(
        { frames },
        this.controller.signal,
        (event) => {
          if (this.active) this.callbacks.onEvent(event);
        },
      );
      if (!this.active) return;
      this.through = frames.at(-1)!.timestamp_sec;
      this.frames = [
        ...frames.slice(-8),
        ...this.frames.filter((frame) => frame.timestamp_sec > this.through),
      ];
      if (this.frames.length < LIVE_WINDOW_FRAMES + 4) this.setBuffering(false);
    } catch (error) {
      if (this.active)
        this.fail(
          error instanceof ApiRequestError
            ? error.message
            : "Не удалось завершить анализ. Найденные события остаются на странице. Запустите анализ снова, чтобы продолжить.",
        );
    } finally {
      this.busy = false;
      if (this.active) {
        this.callbacks.onBusy(false);
        void this.drain();
      }
    }
  }
}

export function liveTimecode(seconds: number) {
  const tenths = Math.round(seconds * 10);
  return `${Math.floor(tenths / 600)
    .toString()
    .padStart(2, "0")}:${((tenths % 600) / 10).toFixed(1).padStart(4, "0")}`;
}
