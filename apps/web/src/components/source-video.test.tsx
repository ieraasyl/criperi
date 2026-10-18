// @vitest-environment jsdom
import { act, createElement, useState } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vite-plus/test";
import { SourceVideo } from "./source-video";

let host: HTMLDivElement;
let root: Root;
const createUrl = vi.fn(),
  revokeUrl = vi.fn();
const play = vi.fn();
beforeEach(() => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  const NativeURL = URL;
  createUrl.mockReset().mockReturnValue("blob:test-video");
  revokeUrl.mockReset();
  vi.stubGlobal(
    "URL",
    class extends NativeURL {
      static createObjectURL = createUrl;
      static revokeObjectURL = revokeUrl;
    },
  );
  play.mockReset().mockResolvedValue(undefined);
  vi.spyOn(HTMLMediaElement.prototype, "play").mockImplementation(() => play());
  vi.stubGlobal("fetch", vi.fn());
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

test("plays the locally retained original from the event time and releases its URL", async () => {
  const file = new File(["test"], "original.mov", { type: "video/quicktime" });
  await act(async () =>
    root.render(createElement(SourceVideo, { file, onFile: vi.fn(), seconds: 8 })),
  );
  await act(async () => host.querySelector<HTMLButtonElement>("button")!.click());
  expect(host.textContent).not.toContain("Заменить видео");
  expect(host.querySelector("input")?.hidden).toBe(true);
  expect(host.querySelectorAll("button")).toHaveLength(1);
  const video = host.querySelector("video")!;
  Object.defineProperty(video, "duration", { value: 60 });
  await act(async () => video.dispatchEvent(new Event("loadedmetadata")));
  expect(video.currentTime).toBe(8);
  expect(play).toHaveBeenCalled();
  expect(createUrl).toHaveBeenCalledWith(file);
  expect(fetch).not.toHaveBeenCalled();
  await act(async () => root.render(null));
  expect(revokeUrl).toHaveBeenCalledWith("blob:test-video");
});

test("a reopened result can attach a local original without uploading it and reports unsupported codecs", async () => {
  function Player() {
    const [file, setFile] = useState<File | null>(null);
    return createElement(SourceVideo, { file, onFile: setFile, seconds: 2 });
  }
  await act(async () => root.render(createElement(Player)));
  const input = host.querySelector("input")!;
  const file = new File(["test"], "copy.mp4", { type: "video/mp4" });
  Object.defineProperty(input, "files", { value: [file] });
  await act(async () => input.dispatchEvent(new Event("change", { bubbles: true })));
  expect(host.querySelector("video")?.getAttribute("src")).toBe("blob:test-video");
  expect(fetch).not.toHaveBeenCalled();
  await act(async () => host.querySelector("video")!.dispatchEvent(new Event("error")));
  expect(host.querySelector('[role="alert"]')?.textContent).toContain(
    "Браузер не смог воспроизвести",
  );
});

test("a source that ends before the event is reported instead of playing a different moment", async () => {
  const file = new File(["test"], "short.mp4", { type: "video/mp4" });
  await act(async () =>
    root.render(createElement(SourceVideo, { file, onFile: vi.fn(), seconds: 20 })),
  );
  await act(async () => host.querySelector<HTMLButtonElement>("button")!.click());
  const video = host.querySelector("video")!;
  Object.defineProperty(video, "duration", { value: 12 });
  await act(async () => video.dispatchEvent(new Event("loadedmetadata")));
  expect(play).not.toHaveBeenCalled();
  expect(host.querySelector('[role="alert"]')?.textContent).toContain("нет этого момента");
});
