// @vitest-environment jsdom
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vite-plus/test";
import { LiveAnalysis } from "./live-analysis";
import type { LiveEvent } from "@app/contracts";
import evidence from "../../../../tests/fixtures/evidence.json";

let root: Root;
let host: HTMLDivElement;
let tick: FrameRequestCallback;
let paused: boolean;
let streams: ReadableStreamDefaultController<Uint8Array>[];
const revokeUrl = vi.fn();

beforeEach(() => {
  paused = true;
  streams = [];
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal(
    "requestAnimationFrame",
    vi.fn((callback) => {
      tick = callback;
      return 1;
    }),
  );
  vi.stubGlobal("cancelAnimationFrame", vi.fn());
  vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:test-video");
  revokeUrl.mockClear();
  vi.spyOn(URL, "revokeObjectURL").mockImplementation(revokeUrl);
  vi.spyOn(HTMLMediaElement.prototype, "play").mockImplementation(async () => {
    paused = false;
  });
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {
    paused = true;
  });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
    drawImage: vi.fn(),
  } as unknown as CanvasRenderingContext2D);
  vi.spyOn(HTMLCanvasElement.prototype, "toDataURL").mockReturnValue(
    `data:image/jpeg;base64,${evidence.jpeg}`,
  );
  vi.stubGlobal(
    "fetch",
    vi.fn().mockImplementation(
      async () =>
        new Response(
          new ReadableStream({
            start(controller) {
              streams.push(controller);
            },
          }),
          { headers: { "Content-Type": "text/event-stream" } },
        ),
    ),
  );
});

afterEach(async () => {
  await act(async () => root?.unmount());
  host?.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function button(text: string) {
  return Array.from(host.querySelectorAll<HTMLButtonElement>("button")).find(
    (item) => item.textContent === text,
  )!;
}

async function chooseFile(name = "video.mp4") {
  const input = host.querySelector<HTMLInputElement>("#live-file")!;
  Object.defineProperty(input, "files", {
    configurable: true,
    value: [new File(["video"], name, { type: "video/mp4" })],
  });
  await act(async () => input.dispatchEvent(new Event("change", { bubbles: true })));
}

async function mount() {
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  await act(async () => root.render(createElement(LiveAnalysis)));
  await chooseFile();
  const video = host.querySelector("video")!;
  Object.defineProperties(video, {
    readyState: { value: 2 },
    videoWidth: { value: 400 },
    videoHeight: { value: 600 },
    paused: { get: () => paused },
  });
  return video;
}

async function capture(video: HTMLVideoElement, time: number) {
  await act(async () => {
    video.currentTime = time;
    video.dispatchEvent(new Event("timeupdate"));
    tick(time * 1000);
  });
}

async function emit(event: LiveEvent, index = 0) {
  await act(async () =>
    streams[index]!.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`)),
  );
}

const result: Extract<LiveEvent, { type: "result" }> = {
  type: "result",
  from_sec: 0,
  to_sec: 3,
  violations: [
    {
      type: "V17. Еда или напиток",
      timecode_sec: 2,
      description: "На столе напиток.",
      evidence_frames: [{ timestamp_sec: 2, image_base64: evidence.jpeg }],
    },
  ],
};

test("playback supplies actual frames and displays streamed detections and findings before the video ends", async () => {
  const video = await mount();
  expect(fetch).not.toHaveBeenCalled();
  await act(async () => button("Начать просмотр и анализ").click());
  for (let time = 0; time < 4; time++) await capture(video, time);
  expect(fetch).toHaveBeenCalledTimes(1);
  const [, init] = vi.mocked(fetch).mock.calls[0]!;
  expect(new Headers(init!.headers).get("Content-Type")).toBe("application/json");
  expect(
    JSON.parse(init!.body as string).frames.map(
      (frame: { timestamp_sec: number }) => frame.timestamp_sec,
    ),
  ).toEqual([0, 1, 2, 3]);
  await emit({
    type: "frame",
    timestamp_sec: 3,
    image_base64: evidence.jpeg,
    people_count: 2,
    objects_count: 1,
  });
  expect(host.textContent).toContain("Обработанный кадр: 00:03.0");
  expect(host.textContent).toContain("Найденные события: 0");
  await emit(result);
  expect(host.textContent).toContain("Найденные события: 1");
  expect(host.textContent).toContain("На столе напиток.");
  expect(video.currentTime).toBe(3);
  await act(async () => button("Перейти к событию").click());
  expect(video.currentTime).toBe(2);
  expect(button("Остановить анализ").disabled).toBe(true);
  expect(init!.signal!.aborted).toBe(true);
});

test("changing the file cancels the old request, revokes the local URL and ignores late findings", async () => {
  const video = await mount();
  await act(async () => button("Начать просмотр и анализ").click());
  for (let time = 0; time < 4; time++) await capture(video, time);
  const signal = vi.mocked(fetch).mock.calls[0]![1]!.signal!;
  await chooseFile("another.mp4");
  expect(signal.aborted).toBe(true);
  expect(revokeUrl).toHaveBeenCalledWith("blob:test-video");
  await emit(result);
  expect(host.textContent).not.toContain("На столе напиток.");
  expect(host.textContent).toContain("Найденные события: 0");
});

test("seeking stops the current inference without clearing findings already displayed", async () => {
  const video = await mount();
  await act(async () => button("Начать просмотр и анализ").click());
  for (let time = 0; time < 4; time++) await capture(video, time);
  await emit(result);
  await act(async () => video.dispatchEvent(new Event("seeking")));
  expect(host.textContent).toContain("Анализ остановлен после перемотки");
  expect(host.textContent).toContain("На столе напиток.");
  expect(button("Остановить анализ").disabled).toBe(true);
});

test("ending playback captures the final visible frame and flushes a partial window", async () => {
  const video = await mount();
  await act(async () => button("Начать просмотр и анализ").click());
  for (let time = 0; time < 4; time++) await capture(video, time);
  await emit(result);
  await capture(video, 4);
  await capture(video, 5);
  await act(async () => {
    video.currentTime = 6.4;
    video.dispatchEvent(new Event("ended"));
  });
  expect(fetch).toHaveBeenCalledTimes(2);
  const body = JSON.parse(vi.mocked(fetch).mock.calls[1]![1]!.body as string);
  expect(body.frames.at(-1).timestamp_sec).toBe(6.4);
  await emit({ type: "result", from_sec: 0, to_sec: 6.4, violations: [] }, 1);
  expect(host.textContent).toContain("Все полученные кадры обработаны");
});
