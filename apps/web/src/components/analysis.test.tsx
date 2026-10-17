// @vitest-environment jsdom
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { createMemoryHistory } from "@tanstack/react-router";
import { defaultScheduler, notifyManager } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, test, vi } from "vite-plus/test";
import App from "@/App";
import { createAppRouter } from "@/router";
import { analysisPollingInterval } from "@/lib/analysis-query";
import type { ReviewJob } from "@app/contracts";
import evidence from "../../../../tests/fixtures/evidence.json";

const auth = vi.hoisted(() => ({
  session: {
    data: { user: { id: "alice", email: "a@example.com" } },
    isPending: false,
    error: null,
  },
}));
vi.mock("@/lib/auth-client", () => ({
  authClient: { useSession: () => auth.session, signOut: vi.fn() },
}));
let root: Root;
let host: HTMLDivElement;
let router: ReturnType<typeof createAppRouter>;
const id = "a97aa7d9-d1d9-467b-980b-7a8b9fd4b251";
function completed(): ReviewJob {
  return {
    job_id: id,
    status: "done",
    verdict: "violation",
    expires_at: new Date(Date.now() + 86400_000).toISOString(),
    violations: [
      {
        type: "Личные вещи",
        timecode_sec: 1,
        description: "На столе видна личная вещь. Источник: НБРК №231, п. 12.",
        evidence_frames: [{ timestamp_sec: 1, image_base64: evidence.jpeg }],
      },
    ],
  };
}
async function render() {
  await act(async () => root.render(createElement(App, { router })));
}
async function mount(path = `/analyses/${id}`) {
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  router = createAppRouter(createMemoryHistory({ initialEntries: [path] }));
  await render();
}
function button(label: string) {
  return Array.from(host.querySelectorAll("button")).find((b) => b.textContent === label)!;
}
beforeEach(() => {
  auth.session.data = { user: { id: "alice", email: "a@example.com" } };
  notifyManager.setScheduler(queueMicrotask);
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("scrollTo", vi.fn());
});
afterEach(async () => {
  await act(async () => root?.unmount());
  host?.remove();
  notifyManager.setScheduler(defaultScheduler);
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

test("a result URL restores findings, citations and evidence without another upload", async () => {
  const fetch = vi.fn().mockResolvedValue(Response.json(completed()));
  vi.stubGlobal("fetch", fetch);
  await mount();
  expect(host.textContent).toContain("Обнаружены признаки возможных нарушений");
  expect(host.textContent).toContain("Источник: НБРК №231, п. 12.");
  expect(host.querySelector("img")?.getAttribute("src")).toBe(
    `data:image/jpeg;base64,${evidence.jpeg}`,
  );
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(fetch.mock.calls[0]?.[1]).toMatchObject({
    method: "GET",
    signal: expect.any(AbortSignal),
  });
});

test("a refresh failure retains findings and only retries GET", async () => {
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(Response.json(completed()))
    .mockResolvedValue(new Response(null, { status: 503 }));
  vi.stubGlobal("fetch", fetch);
  await mount();
  await act(async () => router.invalidate());
  // Reconnect forces a refresh without remounting the private cache.
  const { onlineManager } = await import("@tanstack/react-query");
  vi.spyOn(Date, "now").mockReturnValue(Date.now() + 31_000);
  await act(async () => {
    onlineManager.setOnline(false);
    onlineManager.setOnline(true);
  });
  expect(host.textContent).toContain("Обнаружены признаки возможных нарушений");
  expect(host.textContent).toContain("Обновить результат");
  await act(async () => button("Обновить результат").click());
  expect(fetch.mock.calls.every(([, init]) => init.method === "GET")).toBe(true);
});

test("switching accounts cancels the old request and hides its late response", async () => {
  let resolve!: (value: Response) => void;
  const pending = new Promise<Response>((done) => {
    resolve = done;
  });
  const fetch = vi
    .fn()
    .mockReturnValueOnce(pending)
    .mockResolvedValue(
      Response.json(
        { error: { code: "NOT_FOUND", message: "Анализ не найден." } },
        { status: 404 },
      ),
    );
  vi.stubGlobal("fetch", fetch);
  await mount();
  const signal = fetch.mock.calls[0]?.[1].signal;
  auth.session.data = { user: { id: "bob", email: "b@example.com" } };
  await render();
  await act(async () => resolve(Response.json(completed())));
  expect(signal.aborted).toBe(true);
  expect(host.textContent).not.toContain("На столе видна личная вещь.");
  expect(host.querySelector("img")).toBeNull();
});

test("polling stops on terminal or expired results", () => {
  const done = completed();
  expect(analysisPollingInterval(done)).toBe(false);
  const pending: ReviewJob = { job_id: id, status: "processing", expires_at: done.expires_at };
  expect(analysisPollingInterval(pending)).toBe(2000);
  expect(analysisPollingInterval({ ...pending, expires_at: "2000-01-01T00:00:00.000Z" })).toBe(
    false,
  );
});

test("expiry hides findings and evidence", async () => {
  vi.stubGlobal(
    "fetch",
    vi
      .fn()
      .mockResolvedValue(Response.json({ ...completed(), expires_at: "2000-01-01T00:00:00.000Z" })),
  );
  await mount();
  expect(host.textContent).toContain("Срок хранения анализа истёк");
  expect(host.querySelector("img")).toBeNull();
});

test("a malformed confirmed upload is never automatically resubmitted", async () => {
  const fetch = vi.fn().mockResolvedValue(new Response("not JSON", { status: 202 }));
  vi.stubGlobal("fetch", fetch);
  const NativeFormData = FormData;
  const file = new File(["synthetic"], "test.mov", { type: "video/quicktime" });
  vi.stubGlobal(
    "FormData",
    class extends NativeFormData {
      override get(name: string) {
        return name === "file" ? file : super.get(name);
      }
    },
  );
  await mount("/");
  await act(async () =>
    host
      .querySelector("form")!
      .dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })),
  );
  expect(host.textContent).toContain("Повторная отправка отключена");
  expect(button("Начать анализ").disabled).toBe(true);
  expect(fetch).toHaveBeenCalledTimes(1);
});

test("an uncertain ML submission shows the error and disables another upload", async () => {
  const message = "Видео могло быть принято; повторная отправка отключена.";
  const fetch = vi.fn().mockResolvedValue(
    Response.json(
      {
        error: { code: "SUBMISSION_UNCONFIRMED", message },
      },
      { status: 503 },
    ),
  );
  vi.stubGlobal("fetch", fetch);
  const NativeFormData = FormData;
  const file = new File(["synthetic"], "test.mov", { type: "video/quicktime" });
  vi.stubGlobal(
    "FormData",
    class extends NativeFormData {
      override get(name: string) {
        return name === "file" ? file : super.get(name);
      }
    },
  );
  await mount("/");
  await act(async () =>
    host
      .querySelector("form")!
      .dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })),
  );
  expect(host.textContent).toContain(message);
  expect(button("Начать анализ").disabled).toBe(true);
  expect(router.state.location.pathname).toBe("/");
  expect(fetch).toHaveBeenCalledTimes(1);
});

test("a late upload acknowledgement cannot navigate another account", async () => {
  let resolve!: (value: Response) => void;
  const pending = new Promise<Response>((done) => {
    resolve = done;
  });
  const fetch = vi.fn().mockReturnValue(pending);
  vi.stubGlobal("fetch", fetch);
  const NativeFormData = FormData;
  const file = new File(["synthetic"], "test.mov", { type: "video/quicktime" });
  vi.stubGlobal(
    "FormData",
    class extends NativeFormData {
      override get(name: string) {
        return name === "file" ? file : super.get(name);
      }
    },
  );
  await mount("/");
  await act(async () =>
    host
      .querySelector("form")!
      .dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })),
  );
  auth.session.data = { user: { id: "bob", email: "b@example.com" } };
  await render();
  await act(async () => resolve(Response.json({ job_id: id }, { status: 202 })));
  expect(router.state.location.pathname).toBe("/");
  expect(host.textContent).toContain("b@example.com");
  expect(fetch).toHaveBeenCalledTimes(1);
});
