// @vitest-environment jsdom
import { act, createElement, StrictMode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { createMemoryHistory } from "@tanstack/react-router";
import { afterEach, beforeEach, expect, test, vi } from "vite-plus/test";
import App from "./App";
import { createAppRouter } from "./router";
import { Route as AuthenticatedRoute } from "./routes/_authenticated";

const auth = vi.hoisted(() => ({
  session: {
    data: null as { user: { id: string; email: string; isAnonymous?: boolean } } | null,
    isPending: false,
    error: null as { message: string } | null,
    refetch: vi.fn(),
  },
  getSession: vi.fn(),
  anonymous: vi.fn(),
}));
vi.mock("@/lib/auth-client", () => ({
  authClient: {
    useSession: () => auth.session,
    getSession: auth.getSession,
    signIn: { anonymous: auth.anonymous },
  },
}));

const user = { id: "visitor-1", email: "visitor@anonymous.placeholder.invalid", isAnonymous: true };
let root: Root | undefined;
let host: HTMLDivElement;
let router: ReturnType<typeof createAppRouter>;
let fetchMock: ReturnType<typeof vi.fn>;

async function render() {
  await act(async () => {
    root!.render(createElement(StrictMode, null, createElement(App, { router })));
  });
}

async function mount(entries = ["/"], initialIndex = entries.length - 1) {
  router = createAppRouter(createMemoryHistory({ initialEntries: entries, initialIndex }));
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  await render();
}

function button(text: string) {
  return Array.from(host.querySelectorAll("button")).find((item) => item.textContent === text)!;
}

beforeEach(() => {
  auth.session.data = null;
  auth.session.isPending = false;
  auth.session.error = null;
  auth.session.refetch.mockReset().mockResolvedValue(undefined);
  auth.getSession
    .mockReset()
    .mockImplementation(async () => ({ data: auth.session.data, error: null }));
  auth.anonymous.mockReset().mockImplementation(async () => {
    auth.session.data = { user };
    return { data: { user }, error: null };
  });
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("scrollTo", vi.fn());
  vi.stubGlobal("navigator", { locks: { request: vi.fn(async (_name, callback) => callback()) } });
  fetchMock = vi.fn().mockImplementation(() => Promise.resolve(Response.json({})));
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(async () => {
  await act(async () => root?.unmount());
  root = undefined;
  host?.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

test("first visit prepares one anonymous session in StrictMode and keeps the requested URL", async () => {
  await mount(["/?filter=recent#results"]);
  await render();
  expect(auth.anonymous).toHaveBeenCalledTimes(1);
  expect(router.state.location.href).toBe("/?filter=recent#results");
  expect(host.textContent).toContain("Анализ кассового видео");
  expect(host.textContent).toContain("24 часа в этом браузере");
  expect(host.textContent).not.toContain("Аккаунт");
  expect(host.querySelector('[name="email"]')).toBeNull();
  expect(button("Выйти")).toBeUndefined();
  expect(fetchMock).not.toHaveBeenCalled();
});

test("routing and private requests wait for a slow anonymous session", async () => {
  let finish!: () => void;
  auth.anonymous.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        finish = () => {
          auth.session.data = { user };
          resolve({ data: { user }, error: null });
        };
      }),
  );
  const path = "/analyses/280312ee-e063-42b9-a846-4a18fe361682?event=2&frame=1";
  await mount([path]);
  expect(host.textContent).toContain("Подготовка сеанса");
  expect(router.state.location.href).toBe(path);
  expect(router.state.matches).toHaveLength(0);
  expect(fetchMock).not.toHaveBeenCalled();
  await act(async () => finish());
  await render();
  expect(router.state.location.href).toBe(path);
  expect(host.querySelector('[name="email"]')).toBeNull();
  expect(auth.anonymous).toHaveBeenCalledTimes(1);
  expect(fetchMock).toHaveBeenCalled();
  expect(fetchMock.mock.calls.every(([, init]) => init.method === "GET")).toBe(true);
});

test("an existing anonymous session is reused", async () => {
  auth.session.data = { user };
  await mount();
  expect(host.textContent).toContain("Анализ кассового видео");
  expect(auth.anonymous).not.toHaveBeenCalled();
  expect(auth.getSession).not.toHaveBeenCalled();
});

test("initial session loading and errors prevent private requests and offer retry", async () => {
  auth.session.isPending = true;
  await mount();
  expect(host.textContent).toContain("Подготовка сеанса");
  expect(auth.anonymous).not.toHaveBeenCalled();
  expect(fetchMock).not.toHaveBeenCalled();
  auth.session.isPending = false;
  auth.session.error = { message: "Network unavailable" };
  await render();
  expect(host.textContent).toContain("Не удалось проверить сеанс.");
  await act(async () => button("Повторить").click());
  expect(auth.session.refetch).toHaveBeenCalledTimes(1);
  expect(auth.anonymous).not.toHaveBeenCalled();
  auth.session.error = null;
  auth.session.data = { user };
  await render();
  expect(host.textContent).toContain("Анализ кассового видео");
});

test("failed session creation offers an explicit retry without a login form", async () => {
  auth.anonymous.mockResolvedValueOnce({ error: { status: 429 } });
  await mount();
  expect(host.textContent).toContain("Слишком много попыток");
  expect(auth.anonymous).toHaveBeenCalledTimes(1);
  expect(fetchMock).not.toHaveBeenCalled();
  await act(async () => button("Повторить").click());
  await render();
  expect(auth.anonymous).toHaveBeenCalledTimes(2);
  expect(host.textContent).toContain("Анализ кассового видео");
});

test("a rejected cookie prevents uploads and explains how to restore access", async () => {
  auth.anonymous.mockResolvedValue({ data: { user }, error: null });
  await mount();
  expect(host.textContent).toContain("Разрешите cookies");
  expect(host.querySelector('input[type="file"]')).toBeNull();
  expect(auth.anonymous).toHaveBeenCalledTimes(1);
});

test("auth refetch transitions cannot repeat creation or discard a cookie failure", async () => {
  let finish!: () => void;
  auth.anonymous.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        finish = () => resolve({ data: { user }, error: null });
      }),
  );
  await mount();
  auth.session.isPending = true;
  await render();
  auth.session.isPending = false;
  await render();
  await act(async () => finish());
  expect(host.textContent).toContain("Разрешите cookies");
  await render();
  expect(auth.anonymous).toHaveBeenCalledTimes(1);
  expect(host.textContent).toContain("Разрешите cookies");
  expect(fetchMock).not.toHaveBeenCalled();
});

test("a session created by another tab is reused before any creation request", async () => {
  auth.getSession.mockResolvedValue({ data: { user }, error: null });
  await mount();
  expect(auth.anonymous).not.toHaveBeenCalled();
  expect(auth.session.refetch).toHaveBeenCalledTimes(1);
});

test("unknown routes show a not-found screen with a working home link", async () => {
  auth.session.data = { user };
  await mount(["/missing"]);
  expect(host.textContent).toContain("Страница не найдена");
  expect(fetchMock).not.toHaveBeenCalled();
  await act(async () => {
    host.querySelector<HTMLAnchorElement>("a")!.click();
    await router.load();
  });
  expect(router.state.location.pathname).toBe("/");
});

test("route errors offer retry and recover when loading succeeds", async () => {
  auth.session.data = { user };
  const beforeLoad = AuthenticatedRoute.options.beforeLoad!;
  const failure = vi.spyOn(AuthenticatedRoute.options, "beforeLoad").mockImplementation(() => {
    throw new Error("Page unavailable");
  });
  vi.spyOn(console, "error").mockImplementation(() => {});
  vi.spyOn(console, "warn").mockImplementation(() => {});
  await mount();
  expect(host.textContent).toContain("Не удалось загрузить страницу.");
  failure.mockImplementation(beforeLoad);
  await act(async () => button("Повторить").click());
  expect(host.textContent).toContain("Анализ кассового видео");
  expect(host.textContent).not.toContain("Не удалось загрузить страницу.");
});
