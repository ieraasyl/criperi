import { expect, test, vi } from "vite-plus/test";
import { liveWindowSchema, readLiveEvents, type LiveEvent, type LiveWindow } from "@app/contracts";
import { LiveAnalysisSession, mergeLiveFindings } from "./live-analysis";
import evidence from "../../../../tests/fixtures/evidence.json";

const frame = (timestamp_sec: number) => ({ timestamp_sec, image_base64: evidence.jpeg });
const result: Extract<LiveEvent, { type: "result" }> = {
  type: "result",
  from_sec: 0,
  to_sec: 3,
  violations: [],
};
function callbacks() {
  return {
    onEvent: vi.fn(),
    onBuffering: vi.fn(),
    onBusy: vi.fn(),
    onComplete: vi.fn(),
    onError: vi.fn(),
  };
}

test("serial windows cover every captured frame, keep overlap, and pause a full buffer", async () => {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const windows: LiveWindow[] = [];
  const analyze = vi.fn(async (window: LiveWindow) => {
    windows.push(window);
    if (windows.length === 1) await gate;
  });
  const events = callbacks();
  const session = new LiveAnalysisSession({ ...events, analyze });
  for (let t = 0; t < 20; t++) session.add(frame(t));
  expect(windows).toHaveLength(1);
  expect(events.onBuffering).toHaveBeenCalledWith(true);
  release();
  await vi.waitFor(() => expect(windows).toHaveLength(5));
  expect(windows.map((window) => window.frames.map((item) => item.timestamp_sec))).toEqual([
    [0, 1, 2, 3],
    [0, 1, 2, 3, 4, 5, 6, 7],
    Array.from({ length: 12 }, (_, i) => i),
    Array.from({ length: 12 }, (_, i) => i + 4),
    Array.from({ length: 12 }, (_, i) => i + 8),
  ]);
  expect(events.onBuffering).toHaveBeenLastCalledWith(false);
  session.finish();
  expect(events.onComplete).toHaveBeenCalledTimes(1);
});

test("ending playback flushes the short final window", async () => {
  const analyze = vi.fn().mockResolvedValue(undefined);
  const events = callbacks();
  const session = new LiveAnalysisSession({ ...events, analyze });
  session.add(frame(2));
  session.add(frame(3));
  expect(analyze).not.toHaveBeenCalled();
  session.finish();
  await vi.waitFor(() => expect(events.onComplete).toHaveBeenCalledTimes(1));
  expect(analyze.mock.calls[0]?.[0].frames).toHaveLength(2);
});

test("stopping cancels inference and ignores events arriving from an old playback or account", async () => {
  let release!: () => void;
  let emit!: (event: LiveEvent) => void;
  let signal!: AbortSignal;
  const events = callbacks();
  const session = new LiveAnalysisSession({
    ...events,
    analyze: async (_window, inputSignal, onEvent) => {
      signal = inputSignal;
      emit = onEvent;
      await new Promise<void>((resolve) => {
        release = resolve;
      });
    },
  });
  for (let t = 0; t < 4; t++) session.add(frame(t));
  session.stop();
  expect(signal.aborted).toBe(true);
  emit(result);
  release();
  await Promise.resolve();
  expect(events.onEvent).not.toHaveBeenCalled();
  expect(events.onComplete).not.toHaveBeenCalled();
});

test("a failed window stops without retrying or reporting a clean result", async () => {
  const analyze = vi.fn().mockRejectedValue(new Error("GPU failed"));
  const events = callbacks();
  const session = new LiveAnalysisSession({ ...events, analyze });
  for (let t = 0; t < 8; t++) session.add(frame(t));
  await vi.waitFor(() => expect(events.onError).toHaveBeenCalledTimes(1));
  expect(analyze).toHaveBeenCalledTimes(1);
  expect(events.onComplete).not.toHaveBeenCalled();
});

test("missing playback samples stop analysis instead of claiming uninterrupted coverage", () => {
  const events = callbacks();
  const session = new LiveAnalysisSession({ ...events, analyze: vi.fn() });
  session.add(frame(1));
  session.add(frame(5));
  expect(events.onError).toHaveBeenCalledWith(expect.stringContaining("пропуск"));
});

test("repeated overlapping findings merge while a later episode stays separate", () => {
  const finding = {
    type: "V17. Еда",
    timecode_sec: 2,
    description: "Еда на столе",
    evidence_frames: [frame(2)],
  };
  const first = mergeLiveFindings([], { ...result, to_sec: 11, violations: [finding] });
  const repeated = mergeLiveFindings(first, {
    ...result,
    to_sec: 15,
    violations: [{ ...finding, timecode_sec: 6 }],
  });
  expect(repeated).toHaveLength(1);
  expect(repeated[0]?.timecode_sec).toBe(2);
  expect(
    mergeLiveFindings(repeated, {
      ...result,
      to_sec: 40,
      violations: [{ ...finding, timecode_sec: 36 }],
    }),
  ).toHaveLength(2);
});

test("live windows reject oversized, invalid JPEG and discontinuous inputs", () => {
  expect(liveWindowSchema.safeParse({ frames: [frame(0), frame(1)] }).success).toBe(true);
  expect(liveWindowSchema.safeParse({ frames: [frame(0), frame(3)] }).success).toBe(false);
  expect(liveWindowSchema.safeParse({ frames: [frame(1), frame(0)] }).success).toBe(false);
  expect(
    liveWindowSchema.safeParse({
      frames: [frame(0), { ...frame(1), image_base64: btoa("not JPEG") }],
    }).success,
  ).toBe(false);
  expect(
    liveWindowSchema.safeParse({ frames: Array.from({ length: 13 }, (_, i) => frame(i)) }).success,
  ).toBe(false);
});

test("the event reader handles split UTF-8 and returns events before inference finishes", async () => {
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  const cancel = vi.fn();
  const body = new ReadableStream<Uint8Array>({
    start(value) {
      controller = value;
    },
    cancel,
  });
  const iterator = readLiveEvents(body);
  const bytes = new TextEncoder().encode(
    `: heartbeat\r\ndata: ${JSON.stringify({ type: "stage", stage: "layout" })}\r\n\r\n`,
  );
  controller.enqueue(bytes.slice(0, 10));
  controller.enqueue(bytes.slice(10));
  expect((await iterator.next()).value).toEqual({ type: "stage", stage: "layout" });
  const end = new TextEncoder().encode(
    `data: ${JSON.stringify({ ...result, violations: [{ type: "Еда", timecode_sec: 1, description: "Напиток", evidence_frames: [frame(1)] }] })}\n\n`,
  );
  for (const byte of end) controller.enqueue(new Uint8Array([byte]));
  expect((await iterator.next()).value).toMatchObject({
    type: "result",
    violations: [{ type: "Еда" }],
  });
  expect((await iterator.next()).done).toBe(true);
  expect(cancel).toHaveBeenCalledTimes(1);
});

test("a truncated stream is an error, never an empty findings list", async () => {
  const body = new Response('data: {"type":"stage","stage":"verifying"}\n\n').body!;
  const iterator = readLiveEvents(body);
  await iterator.next();
  await expect(iterator.next()).rejects.toThrow("without a result");
  expect(body.locked).toBe(false);
});
