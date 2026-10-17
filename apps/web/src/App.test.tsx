// @vitest-environment jsdom
import { act, createElement, StrictMode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { createMemoryHistory } from "@tanstack/react-router";
import { afterEach, beforeEach, expect, test, vi } from "vite-plus/test";
import App from "./App";
import { createAppRouter } from "./router";
import { Route as LoginRoute } from "./routes/login";

const auth = vi.hoisted(() => ({
  session: {
    data: null as { user: { id: string; email: string } } | null,
    isPending: false,
    error: null as { message: string } | null,
    refetch: vi.fn(),
  },
  signIn: vi.fn(),
  signUp: vi.fn(),
  signOut: vi.fn(),
}));
vi.mock("@/lib/auth-client", () => ({
  authClient: {
    useSession: () => auth.session,
    signIn: { email: auth.signIn },
    signUp: { email: auth.signUp },
    signOut: auth.signOut,
  },
}));

const user = { id: "user-1", email: "user@example.com" };
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

async function settle() {
  await act(async () => {
    await router.load();
  });
}

function button(text: string) {
  return Array.from(host.querySelectorAll("button")).find((item) => item.textContent === text)!;
}

beforeEach(() => {
  auth.session.data = null;
  auth.session.isPending = false;
  auth.session.error = null;
  auth.session.refetch.mockReset();
  auth.signIn.mockReset();
  auth.signUp.mockReset();
  auth.signOut.mockReset();
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("scrollTo", vi.fn());
  fetchMock = vi.fn().mockImplementation(() => Promise.resolve(Response.json({ notes: [] })));
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(async () => {
  await act(async () => root?.unmount());
  root = undefined;
  host?.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

test("signed-out navigation preserves path, search, and hash without fetching private data", async () => {
  await mount(["/?filter=recent#results"]);
  expect(router.state.location.pathname).toBe("/login");
  expect(router.state.location.search.redirect).toBe("/?filter=recent#results");
  expect(host.querySelector('[name="email"]')).not.toBeNull();
  expect(host.textContent).not.toContain("Notes");
  expect(fetchMock).not.toHaveBeenCalled();
});

test("initial session loading and errors prevent protected requests and offer retry", async () => {
  auth.session.isPending = true;
  await mount();
  expect(host.textContent).toContain("Loading session");
  expect(fetchMock).not.toHaveBeenCalled();
  auth.session.isPending = false;
  auth.session.error = { message: "Network unavailable" };
  await render();
  expect(host.textContent).toContain("Could not load session.");
  await act(async () => button("Retry loading session").click());
  expect(auth.session.refetch).toHaveBeenCalledTimes(1);
  expect(fetchMock).not.toHaveBeenCalled();
  auth.session.error = null;
  auth.session.data = { user };
  await render();
  expect(host.textContent).toContain("Notes");
});

test.each(["sign-in", "registration"])(
  "%s returns to the requested URL and replaces login history",
  async (flow) => {
    await mount(["/previous", "/?filter=recent#results"]);
    if (flow === "registration") {
      await act(async () => button("Create an account").click());
      host.querySelector<HTMLInputElement>('[name="name"]')!.value = "Test User";
    }
    host.querySelector<HTMLInputElement>('[name="email"]')!.value = user.email;
    host.querySelector<HTMLInputElement>('[name="password"]')!.value = "password123";
    const submit = flow === "registration" ? auth.signUp : auth.signIn;
    submit.mockResolvedValue({ error: null });
    await act(async () => {
      host
        .querySelector("form")!
        .dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    });
    expect(submit).toHaveBeenCalledWith(
      expect.objectContaining({ email: user.email, password: "password123" }),
    );
    auth.session.data = { user };
    await render();
    expect(router.state.location.href).toBe("/?filter=recent#results");
    expect(host.textContent).toContain("Notes");
    await act(async () => router.history.back());
    await settle();
    expect(router.state.location.pathname).toBe("/previous");
  },
);

test.each(["/login", "/login?redirect=%2F%3Ffilter%3Drecent%23results"])(
  "authenticated direct visit to %s leaves login",
  async (url) => {
    auth.session.data = { user };
    await mount([url]);
    expect(router.state.location.href).toBe(
      url.includes("redirect") ? "/?filter=recent#results" : "/",
    );
    expect(host.querySelector('[name="email"]')).toBeNull();
    expect(host.textContent).toContain("Notes");
  },
);

test.each([
  "https://evil.example",
  "//evil.example",
  "/\\evil.example",
  "/login?redirect=/",
  "/%6cogin",
  123,
])("an invalid return destination %s falls back home", async (destination) => {
  auth.session.data = { user };
  const query =
    typeof destination === "string" ? encodeURIComponent(destination) : String(destination);
  await mount([`/login?redirect=${query}`]);
  expect(router.state.location.href).toBe("/");
});

test("failed authentication stays on login and does not fetch private data", async () => {
  auth.signIn.mockResolvedValue({ error: { message: "Invalid credentials" } });
  await mount(["/login"]);
  host.querySelector<HTMLInputElement>('[name="email"]')!.value = user.email;
  host.querySelector<HTMLInputElement>('[name="password"]')!.value = "password123";
  await act(async () => {
    host
      .querySelector("form")!
      .dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
  });
  expect(host.textContent).toContain("Invalid credentials");
  expect(router.state.location.pathname).toBe("/login");
  expect(fetchMock).not.toHaveBeenCalled();
});

test("sign-out removes protected content and returns to login", async () => {
  auth.session.data = { user };
  auth.signOut.mockResolvedValue({ error: null });
  await mount(["/?view=all#top"]);
  await act(async () => button("Sign out").click());
  expect(auth.signOut).toHaveBeenCalledTimes(1);
  auth.session.data = null;
  await render();
  expect(router.state.location.pathname).toBe("/login");
  expect(router.state.location.search.redirect).toBe("/?view=all#top");
  expect(host.textContent).not.toContain("Signed in as");
  expect(host.querySelector('[name="title"]')).toBeNull();
});

test("unknown routes show a public not-found screen with a working home link", async () => {
  await mount(["/missing"]);
  expect(router.state.location.pathname).toBe("/missing");
  expect(host.textContent).toContain("Page not found");
  expect(fetchMock).not.toHaveBeenCalled();
  await act(async () => host.querySelector<HTMLAnchorElement>("a")!.click());
  await settle();
  expect(router.state.location.pathname).toBe("/login");
});

test("route errors offer retry and recover when loading succeeds", async () => {
  const beforeLoad = LoginRoute.options.beforeLoad!;
  const failure = vi.spyOn(LoginRoute.options, "beforeLoad").mockImplementation(() => {
    throw new Error("Page unavailable");
  });
  vi.spyOn(console, "error").mockImplementation(() => {});
  vi.spyOn(console, "warn").mockImplementation(() => {});
  await mount(["/login"]);
  expect(host.textContent).toContain("Could not load this page.");
  expect(host.querySelector("a")!.getAttribute("href")).toBe("/");
  failure.mockImplementation(beforeLoad);
  await act(async () => button("Retry").click());
  expect(host.querySelector('[name="email"]')).not.toBeNull();
  expect(host.textContent).not.toContain("Could not load this page.");
});
