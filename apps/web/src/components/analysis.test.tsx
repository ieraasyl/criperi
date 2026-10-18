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
    reviews: [],
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
  return Array.from(document.querySelectorAll("button")).find(
    (b) => b.textContent?.replace(/^[✓✕○]\s*/, "").trim() === label,
  )!;
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
  const result = completed();
  const fetch = vi.fn().mockResolvedValue(Response.json(result));
  vi.stubGlobal("fetch", fetch);
  await mount();
  expect(host.textContent).toContain("Обнаружено 1 событие для проверки");
  expect(button("Нормативное основание")).toBeDefined();
  expect(host.querySelector("img")?.getAttribute("src")).toBe(
    `data:image/jpeg;base64,${evidence.jpeg}`,
  );
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(fetch.mock.calls[0]?.[1]).toMatchObject({
    method: "GET",
    signal: expect.any(AbortSignal),
  });
  const summary = host.querySelector('section[aria-label="Сводка анализа"]')!;
  expect(summary.querySelector("time")?.getAttribute("datetime")).toBe(result.expires_at);
  expect(summary.textContent).toContain("Результат хранится до");
});

test("frame selection updates the large preview and opens the selected frame", async () => {
  const result = completed();
  if (result.status !== "done") throw new Error("Expected a completed result");
  result.violations[0]!.evidence_frames = [1, 2.5, 5].map((timestamp_sec) => ({
    timestamp_sec,
    image_base64: evidence.jpeg,
  }));
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(result)));
  await mount();

  const viewer = host.querySelector('section[aria-label="Кадры события"]')!;
  const choices = viewer.querySelectorAll<HTMLButtonElement>('[aria-label="Выбор кадра"] button');
  expect(viewer.querySelector("figure img")?.getAttribute("alt")).toBe("Кадр события: 00:01.0");
  await act(async () => choices[1]!.click());
  expect(viewer.querySelector("figure img")?.getAttribute("alt")).toBe("Кадр события: 00:02.5");
  expect(viewer.querySelector("figcaption")).toBeNull();
  expect(choices[0]!.getAttribute("aria-pressed")).toBe("false");
  expect(choices[1]!.getAttribute("aria-pressed")).toBe("true");

  await act(async () =>
    viewer.querySelector<HTMLButtonElement>('[aria-label="Открыть кадр 00:02.5"]')!.click(),
  );
  const dialog = document.querySelector('[role="dialog"]')!;
  expect(dialog.getAttribute("aria-label")).toBe("Личные вещи. Кадр 2 из 3");
  expect(dialog.querySelector("img")?.getAttribute("alt")).toBe("Кадр события: 00:02.5");
});

test("findings label video time, code and reference separately from the title", async () => {
  const result = completed();
  if (result.status !== "done") throw new Error("Expected a completed result");
  result.violations[0]!.type = "V18. Дверь кабинета оставлена открытой (НБРК № 14, п. 8)";
  result.violations[0]!.timecode_sec = 2;
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(result)));
  await mount();

  expect(host.querySelector("h3")?.textContent).toBe("Дверь кабинета оставлена открытой");
  expect(host.querySelector("article")?.textContent).toContain("Время на видео: 00:02.0");
  const code = host
    .querySelector("#finding-title")
    ?.parentElement?.querySelector('[data-slot="tooltip-trigger"]');
  expect(code?.textContent).toBe("V18");
  expect(host.querySelector('nav [data-slot="tooltip-trigger"]')?.textContent).toBe("V18");
  expect(
    Array.from(host.querySelectorAll("button")).some((item) =>
      item.textContent?.includes("НБРК № 14, п. 8"),
    ),
  ).toBe(true);
});

test("descriptions distinguish the observation, source, quotation and qualification", async () => {
  const result = completed();
  if (result.status !== "done") throw new Error("Expected a completed result");
  const observation = "С 2 с дверь кабинета остаётся открытой после прохода людей.";
  const source = "Название документа из ответа сервиса, п. 8";
  const quote = "Цитата из ответа сервиса.";
  const qualification =
    "Связь с нормой косвенная; результат является подсказкой для проверки человеком.";
  result.violations[0]!.description = `${observation} Источник: ${source}: «${quote}» ${qualification}`;
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(result)));
  await mount();

  expect(document.querySelector('[role="dialog"]')).toBeNull();
  await act(async () => button("Нормативное основание").click());
  const sections = Array.from(document.querySelectorAll("h4")).map((heading) => [
    heading.textContent,
    heading.nextElementSibling?.textContent,
  ]);
  expect(sections).toEqual(
    expect.arrayContaining([
      ["Наблюдение", "С 00:02 дверь кабинета остаётся открытой после прохода людей."],
      ["Нормативный источник", source],
      ["Текст нормы", `«${quote}»`],
      ["Ограничения вывода", qualification],
    ]),
  );
  expect(document.querySelector("blockquote")?.textContent).toBe(`«${quote}»`);
  expect(
    document.querySelector('[role="dialog"][aria-label="Нормативное основание"]'),
  ).not.toBeNull();
  expect(
    Array.from(host.querySelectorAll("h4"))
      .find((heading) => heading.textContent === "Ограничения вывода")
      ?.closest("details"),
  ).toBeNull();
});

test("selecting a finding replaces its details and keeps only one finding expanded", async () => {
  const result = completed();
  if (result.status !== "done") throw new Error("Expected a completed result");
  result.violations.push({
    ...result.violations[0]!,
    type: "V18. Дверь кабинета оставлена открытой (НБРК № 14, п. 8)",
    timecode_sec: 2,
    description: "С 2 с дверь кабинета остаётся открытой после прохода людей.",
  });
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(result)));
  await mount();

  const rows = host.querySelectorAll<HTMLButtonElement>('nav[aria-label="Выбор события"] button');
  expect(rows).toHaveLength(2);
  expect(rows[0]!.getAttribute("aria-pressed")).toBe("true");
  expect(host.querySelector("article")?.textContent).toContain("На столе видна личная вещь.");
  await act(async () => rows[1]!.click());

  expect(host.querySelectorAll("article")).toHaveLength(1);
  expect(host.querySelector("article")?.textContent).toContain(
    "С 00:02 дверь кабинета остаётся открытой",
  );
  expect(host.querySelector("article")?.textContent).not.toContain("На столе видна личная вещь.");
  expect(rows[0]!.getAttribute("aria-pressed")).toBe("false");
  expect(rows[1]!.getAttribute("aria-pressed")).toBe("true");
  expect(host.querySelector("h3")?.textContent).toBe("Дверь кабинета оставлена открытой");
  const selector = host.querySelector<HTMLButtonElement>("#finding-select")!;
  expect(selector.textContent).toContain("Дверь кабинета оставлена открытой");
  await act(async () => selector.click());
  const first = document.querySelector<HTMLElement>('[role="option"]')!;
  await act(async () => first.click());
  expect(host.querySelector("h3")?.textContent).toBe("Личные вещи");
  expect(rows[0]!.getAttribute("aria-pressed")).toBe("true");
});

test("new analysis uses an alert dialog and only leaves the review after confirmation", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(completed())));
  await mount();
  const action = button("Новый анализ");
  expect(host.querySelector("h1")?.textContent).toBe("Результат анализа");
  await act(async () => action.click());
  expect(router.state.location.pathname).toBe(`/analyses/${id}`);
  expect(document.querySelector('[role="alertdialog"]')?.textContent).toContain(
    "Есть события без окончательного решения",
  );
  await act(async () => button("Продолжить проверку").click());
  expect(router.state.location.pathname).toBe(`/analyses/${id}`);
  await act(async () => action.click());
  await act(async () => button("Начать новый анализ").click());
  expect(router.state.location.pathname).toBe("/");
  expect(host.querySelector('input[type="file"]')).not.toBeNull();
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
  expect(host.textContent).toContain("Обнаружено 1 событие для проверки");
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
  const pending: ReviewJob = {
    job_id: id,
    status: "processing",
    expires_at: done.expires_at,
    reviews: [],
  };
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

function enterComment(value: string) {
  const input = host.querySelector<HTMLTextAreaElement>("#review-comment")!;
  Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!.call(input, value);
  input.dispatchEvent(new Event("input", { bubbles: true }));
}

test("review decisions save comments, update progress and survive a result refresh", async () => {
  const result = completed();
  const fetch = vi.fn().mockImplementation((_url, init) => {
    if (init.method === "PUT") {
      const input = JSON.parse(init.body);
      const review = { ...input, event_index: 0, updated_at: new Date().toISOString() };
      result.reviews = [review];
      return Promise.resolve(Response.json({ review }));
    }
    return Promise.resolve(Response.json(result));
  });
  vi.stubGlobal("fetch", fetch);
  await mount();
  await act(async () => enterComment("Проверено по исходному видео."));
  await act(async () => button("Подтвердить").click());
  expect(button("Подтвердить").getAttribute("aria-pressed")).toBe("true");
  expect(fetch.mock.calls.filter(([, init]) => init.method === "PUT")).toHaveLength(0);
  await act(async () => button("Сохранить решение").click());
  expect(fetch.mock.calls.filter(([, init]) => init.method === "PUT")).toHaveLength(1);
  expect(result.reviews[0]?.comment).toBe("Проверено по исходному видео.");
  expect(host.textContent).toContain("Проверено 1 из 1");
  expect(host.querySelector('nav[aria-label="Выбор события"]')?.textContent).toContain(
    "Подтверждено",
  );
  expect(button("Экспорт отчёта").disabled).toBe(false);
  await act(async () => router.invalidate());
  expect(host.textContent).not.toContain("Сохранено:");
  expect(button("Подтвердить").getAttribute("aria-pressed")).toBe("true");
});

test("a malformed acknowledged review refreshes the result without repeating the write", async () => {
  const result = completed();
  const fetch = vi.fn().mockImplementation((_url, init) => {
    if (init.method === "PUT") {
      result.reviews = [
        { event_index: 0, status: "rejected", comment: "", updated_at: new Date().toISOString() },
      ];
      return Promise.resolve(new Response("not JSON", { status: 200 }));
    }
    return Promise.resolve(Response.json(result));
  });
  vi.stubGlobal("fetch", fetch);
  await mount();
  await act(async () => button("Отклонить").click());
  await act(async () => button("Сохранить решение").click());
  expect(host.textContent).toContain("Проверено 1 из 1");
  expect(fetch.mock.calls.filter(([, init]) => init.method === "PUT")).toHaveLength(1);
  expect(fetch.mock.calls.filter(([, init]) => init.method === "GET").length).toBeGreaterThan(1);
});

test("failed review writes retain saved decisions and leave the comment available", async () => {
  const result = completed();
  result.reviews = [
    {
      event_index: 0,
      status: "confirmed",
      comment: "Старый комментарий",
      updated_at: new Date().toISOString(),
    },
  ];
  const fetch = vi
    .fn()
    .mockImplementation((_url, init) =>
      Promise.resolve(
        init.method === "PUT" ? new Response(null, { status: 503 }) : Response.json(result),
      ),
    );
  vi.stubGlobal("fetch", fetch);
  await mount();
  await act(async () => enterComment("Новый комментарий"));
  await act(async () => button("Отклонить").click());
  await act(async () => button("Сохранить решение").click());
  expect(host.textContent).toContain("Проверено 1 из 1");
  expect(host.querySelector('nav[aria-label="Выбор события"]')?.textContent).toContain(
    "Подтверждено",
  );
  expect(host.querySelector<HTMLTextAreaElement>("#review-comment")?.value).toBe(
    "Новый комментарий",
  );
  expect(host.querySelector('[role="alert"]')).not.toBeNull();
  expect(fetch.mock.calls.filter(([, init]) => init.method === "PUT")).toHaveLength(1);
});

test("deep links, nearest frames and keyboard navigation preserve the selected event and frame", async () => {
  const result = completed();
  if (result.status !== "done") throw new Error("Expected done");
  result.violations[0]!.timecode_sec = 2;
  result.violations[0]!.evidence_frames = [0, 2.5, 5].map((timestamp_sec) => ({
    timestamp_sec,
    image_base64: evidence.jpeg,
  }));
  result.violations.push({
    ...result.violations[0]!,
    timecode_sec: 8,
    evidence_frames: [6, 8.5, 11].map((timestamp_sec) => ({
      timestamp_sec,
      image_base64: evidence.jpeg,
    })),
  });
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(result)));
  await mount(`/analyses/${id}?event=2`);
  expect(host.querySelector("figure img")?.getAttribute("alt")).toBe("Кадр события: 00:08.5");
  expect(host.textContent).not.toContain("· 2 эпизода");
  expect(host.textContent).toContain("Эпизод 2/2");
  await act(async () =>
    document.body.dispatchEvent(
      new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true, cancelable: true }),
    ),
  );
  expect(router.state.location.search).toMatchObject({ event: 2, frame: 3 });
  expect(host.querySelector("figure img")?.getAttribute("alt")).toBe("Кадр события: 00:11.0");
  await act(async () =>
    document.body.dispatchEvent(
      new KeyboardEvent("keydown", { key: "ArrowUp", bubbles: true, cancelable: true }),
    ),
  );
  expect(router.state.location.search).toMatchObject({ event: 1, frame: 2 });
  expect(host.querySelector("figure img")?.getAttribute("alt")).toBe("Кадр события: 00:02.5");
});

test("the norm chip opens one drawer with copying and an official source link", async () => {
  const result = completed();
  if (result.status !== "done") throw new Error("Expected done");
  result.violations[0]!.type = "V18. Дверь открыта (НБРК № 14, п. 8)";
  result.violations[0]!.description = "С 2 с дверь открыта. Источник: Документ: «Точная цитата.»";
  const writeText = vi.fn().mockResolvedValue(undefined);
  vi.stubGlobal("navigator", { clipboard: { writeText } });
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(result)));
  await mount();
  const trigger = Array.from(host.querySelectorAll<HTMLButtonElement>("button")).find((item) =>
    item.textContent?.includes("НБРК № 14, п. 8"),
  )!;
  await act(async () => trigger.click());
  const drawer = document.querySelector('[role="dialog"][aria-label="Нормативное основание"]')!;
  expect(drawer).not.toBeNull();
  await vi.waitFor(() =>
    expect(
      document.documentElement.style.overflowY === "hidden" ||
        document.body.style.overflowY === "hidden",
    ).toBe(true),
  );
  expect(drawer.querySelector("a")?.getAttribute("href")).toBe(
    "https://adilet.zan.kz/rus/docs/V2000020075",
  );
  await act(async () => button("Копировать цитату").click());
  expect(writeText).toHaveBeenCalledWith("Точная цитата.");
  await act(async () =>
    drawer.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true })),
  );
  expect(!drawer.isConnected || drawer.hasAttribute("data-closed")).toBe(true);
  await vi.waitFor(() => {
    expect(document.documentElement.style.overflowY).not.toBe("hidden");
    expect(document.body.style.overflowY).not.toBe("hidden");
  });
});

test("the dark viewer keeps frame selection after closing and locks background scrolling", async () => {
  const result = completed();
  if (result.status !== "done") throw new Error("Expected done");
  result.violations[0]!.evidence_frames = [0, 2.5, 5].map((timestamp_sec) => ({
    timestamp_sec,
    image_base64: evidence.jpeg,
  }));
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(result)));
  await mount();
  const open = host.querySelector<HTMLButtonElement>('figure button[aria-haspopup="dialog"]')!;
  await act(async () => open.click());
  const dialog = document.querySelector('[role="dialog"][data-slot="dialog-content"]')!;
  expect(dialog).not.toBeNull();
  await vi.waitFor(() =>
    expect(
      document.documentElement.style.overflowY === "hidden" ||
        document.body.style.overflowY === "hidden",
    ).toBe(true),
  );
  await act(async () =>
    dialog.querySelector<HTMLButtonElement>('[aria-label="Следующий кадр"]')!.click(),
  );
  expect(router.state.location.search.frame).toBe(2);
  await act(async () =>
    dialog.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft", bubbles: true })),
  );
  expect(router.state.location.search.frame).toBe(1);
  await act(async () =>
    dialog.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true })),
  );
  expect(router.state.location.search.frame).toBe(2);
  await act(async () => button("Закрыть кадр").click());
  expect(!dialog.isConnected || dialog.hasAttribute("data-closed")).toBe(true);
  expect(host.querySelector("figure img")?.getAttribute("alt")).toBe("Кадр события: 00:02.5");
  await vi.waitFor(() => {
    expect(document.documentElement.style.overflowY).not.toBe("hidden");
    expect(document.body.style.overflowY).not.toBe("hidden");
  });
});

test("save and next skips finalized episodes and export waits for every final decision", async () => {
  const result = completed();
  if (result.status !== "done") throw new Error("Expected done");
  result.violations.push(
    { ...result.violations[0]!, timecode_sec: 8 },
    { ...result.violations[0]!, timecode_sec: 20 },
  );
  result.reviews = [
    { event_index: 1, status: "confirmed", comment: "", updated_at: new Date().toISOString() },
  ];
  let acknowledge: () => void;
  const fetch = vi.fn().mockImplementation((_url: string, init: RequestInit) => {
    if (init.method === "PUT") {
      const input = JSON.parse(init.body as string);
      const index = Number(_url.split("/").at(-1));
      const review = { ...input, event_index: index, updated_at: new Date().toISOString() };
      result.reviews = [...result.reviews.filter((item) => item.event_index !== index), review];
      return new Promise<Response>((resolve) => {
        acknowledge = () => resolve(Response.json({ review }));
      });
    }
    return Promise.resolve(Response.json(result));
  });
  vi.stubGlobal("fetch", fetch);
  await mount();
  expect(host.querySelector("nav h2")?.textContent).toBe("Проверено 1 из 3");
  expect(button("Экспорт отчёта").disabled).toBe(true);
  const exportTrigger = host.querySelector<HTMLElement>('[data-slot="tooltip-trigger"]')!;
  await act(async () => {
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab", bubbles: true }));
    exportTrigger.focus();
    await new Promise((resolve) => setTimeout(resolve, 20));
  });
  expect(document.querySelector('[data-slot="tooltip-content"]')?.textContent).toContain(
    "Подтвердите или отклоните каждый эпизод",
  );
  await act(async () => exportTrigger.blur());
  await act(async () => button("Подтвердить").click());
  expect(button("Подтвердить").getAttribute("aria-pressed")).toBe("true");
  expect(fetch.mock.calls.filter(([, init]) => init.method === "PUT")).toHaveLength(0);
  await act(async () => button("Сохранить и перейти к следующему").click());
  expect(router.state.location.search.event).not.toBe(3);
  expect(button("Экспорт отчёта").disabled).toBe(true);
  await act(async () => acknowledge!());
  expect(router.state.location.search.event).toBe(3);
  expect(host.querySelector("nav h2")?.textContent).toBe("Проверено 2 из 3");
  await act(async () => button("Нужна проверка").click());
  expect(button("Нужна проверка").getAttribute("aria-pressed")).toBe("true");
  expect(button("Подтвердить").getAttribute("aria-pressed")).toBe("false");
  await act(async () => button("Сохранить решение").click());
  await act(async () => acknowledge!());
  expect(button("Экспорт отчёта").disabled).toBe(true);
  expect(host.querySelector("nav h2")?.textContent).toBe("Проверено 2 из 3");
  await act(async () => button("Отклонить").click());
  await act(async () => button("Сохранить решение").click());
  await act(async () => acknowledge!());
  expect(button("Экспорт отчёта").disabled).toBe(false);
  expect(host.querySelector("nav h2")?.textContent).toBe("Проверено 3 из 3");
});

test("the event header copies the selected frame link and keeps codes before both titles", async () => {
  const result = completed();
  if (result.status !== "done") throw new Error("Expected done");
  result.violations[0]!.type =
    "V17. Еда или напитки на рабочем месте или в сейфе (НБРК № 231, п. 12)";
  result.violations[0]!.evidence_frames = [6, 8.5, 11].map((timestamp_sec) => ({
    timestamp_sec,
    image_base64: evidence.jpeg,
  }));
  result.reviews = [
    { event_index: 0, status: "rejected", comment: "", updated_at: new Date().toISOString() },
  ];
  const writeText = vi.fn().mockResolvedValue(undefined);
  vi.stubGlobal("navigator", { clipboard: { writeText } });
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(result)));
  await mount(`/analyses/${id}?event=1&frame=3`);
  const heading = host.querySelector("#finding-title")!;
  expect(heading.parentElement?.textContent).toContain(
    "V17 Еда или напитки на рабочем месте или в сейфе",
  );
  expect(host.querySelector("nav")?.textContent).toContain(
    "V17 Еда или напитки на рабочем месте или в сейфе",
  );
  expect(host.querySelector("nav")?.textContent).not.toMatch(/· \d+ эпизод/);
  expect(host.querySelector('nav [data-slot="badge"][data-variant="rejected"]')?.textContent).toBe(
    "Отклонено",
  );
  expect(button("Отклонить").getAttribute("aria-pressed")).toBe("true");
  expect(host.querySelector('figure [aria-hidden="true"]')).toBeNull();
  const copy = host.querySelector<HTMLButtonElement>(
    '[aria-label="Копировать ссылку на событие"]',
  )!;
  expect(copy.parentElement?.previousElementSibling).toBe(heading);
  expect(copy.parentElement?.parentElement?.contains(heading)).toBe(true);
  await act(async () => copy.click());
  const shared = new URL(writeText.mock.calls[0]![0]);
  expect(shared.pathname).toBe(`/analyses/${id}`);
  expect(shared.searchParams.get("event")).toBe("1");
  expect(shared.searchParams.get("frame")).toBe("3");
  expect(host.textContent).toContain("Ссылка скопирована");
});
