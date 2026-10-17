// @vitest-environment jsdom
import { act, createElement, StrictMode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vite-plus/test";
import {
  defaultScheduler,
  focusManager,
  notifyManager,
  onlineManager,
} from "@tanstack/react-query";
import App from "@/App";
import { AuthenticatedQueryProvider } from "./authenticated-query-provider";
import { Notes } from "./notes";

const session = vi.hoisted(() => ({
  data: null as { user: { id: string; email: string } } | null,
  isPending: false,
}));
vi.mock("@/lib/auth-client", () => ({
  authClient: { signOut: vi.fn(), useSession: () => session },
}));

const user = { id: "user-1", email: "user@example.com" };
const otherUser = { id: "user-2", email: "other@example.com" };
const note = {
  id: "note-1",
  userId: user.id,
  title: "Saved note",
  createdAt: "2026-10-16T12:00:00.000Z",
};
let root: Root;
let host: HTMLDivElement;

function deferred() {
  let resolve!: (value: Response) => void;
  const promise = new Promise<Response>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

async function mount(fetch: ReturnType<typeof vi.fn>, app = false) {
  vi.stubGlobal("fetch", fetch);
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  if (app) return renderSession(user);
  await act(async () => {
    root.render(createElement(AuthenticatedQueryProvider, null, createElement(Notes, { user })));
  });
}

async function renderSession(next: typeof user | null) {
  session.data = next ? { user: next } : null;
  await act(async () => {
    root.render(createElement(StrictMode, null, createElement(App)));
  });
}

async function save() {
  host.querySelector<HTMLInputElement>('[name="title"]')!.value = note.title;
  await act(async () => {
    host
      .querySelector("form")!
      .dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
  });
}

beforeEach(() => {
  notifyManager.setScheduler(queueMicrotask);
});
afterEach(async () => {
  await act(async () => root?.unmount());
  host?.remove();
  notifyManager.setScheduler(defaultScheduler);
  focusManager.setFocused(undefined);
  onlineManager.setOnline(true);
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

test("saving refreshes the list; a failed refresh offers a GET-only retry", async () => {
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(Response.json({ notes: [] }))
    .mockResolvedValueOnce(Response.json({ note }, { status: 201 }))
    .mockResolvedValueOnce(new Response(null, { status: 500 }))
    .mockResolvedValueOnce(Response.json({ notes: [note] }));
  await mount(fetch);
  await save();
  expect(host.querySelector<HTMLInputElement>('[name="title"]')!.value).toBe("");
  expect(host.textContent).toContain("Note saved.");
  expect(host.textContent).toContain("Could not load notes.");
  expect(host.querySelector("ul")!.textContent).toBe(note.title);
  const retry = Array.from(host.querySelectorAll("button")).find(
    (button) => button.textContent === "Retry loading notes",
  )!;
  await act(async () => retry.click());
  expect(host.querySelector("ul")!.textContent).toBe(note.title);
  expect(fetch.mock.calls.filter(([, options]) => options?.method === "POST")).toHaveLength(1);
});

test.each([401, 500])("a failed save (%s) preserves input without retrying", async (status) => {
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(Response.json({ notes: [] }))
    .mockResolvedValueOnce(Response.json({}, { status }));
  await mount(fetch);
  await save();
  expect(host.querySelector<HTMLInputElement>('[name="title"]')!.value).toBe(note.title);
  expect(host.querySelector('[role="alert"]')).not.toBeNull();
  expect(host.textContent).not.toContain("Note saved.");
  expect(fetch).toHaveBeenCalledTimes(2);
});

test.each([
  { status: 201, body: "{}" },
  { status: 200, body: "not JSON" },
])(
  "a malformed $status save response clears the input and reloads without repeating POST",
  async ({ status, body }) => {
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(Response.json({ notes: [] }))
      .mockResolvedValueOnce(new Response(body, { status }))
      .mockResolvedValueOnce(Response.json({ notes: [note] }));
    await mount(fetch);
    await save();

    expect(host.querySelector<HTMLInputElement>('[name="title"]')!.value).toBe("");
    expect(host.textContent).toContain("Note saved.");
    expect(host.querySelector('[role="alert"]')).toBeNull();
    expect(host.querySelector("ul")!.textContent).toBe(note.title);
    expect(fetch.mock.calls.filter(([, options]) => options?.method === "POST")).toHaveLength(1);
    expect(fetch).toHaveBeenCalledTimes(3);
  },
);

test("a save cancels an older initial load before refreshing", async () => {
  const initial = deferred();
  const fetch = vi
    .fn()
    .mockReturnValueOnce(initial.promise)
    .mockResolvedValueOnce(Response.json({ note }, { status: 201 }))
    .mockResolvedValueOnce(Response.json({ notes: [note] }));
  await mount(fetch);
  await save();
  await act(async () => initial.resolve(Response.json({ notes: [] })));
  expect(fetch.mock.calls[0]![1].signal.aborted).toBe(true);
  expect(host.querySelector("ul")!.textContent).toBe(note.title);
});

test.each(["focus", "reconnect"])("%s refreshes only after the freshness window", async (event) => {
  const fetch = vi.fn().mockImplementation(() => Promise.resolve(Response.json({ notes: [note] })));
  await mount(fetch);
  const now = Date.now();
  async function trigger() {
    await act(async () => {
      if (event === "focus") {
        focusManager.setFocused(false);
        focusManager.setFocused(true);
      } else {
        onlineManager.setOnline(false);
        onlineManager.setOnline(true);
      }
    });
  }
  await trigger();
  expect(fetch).toHaveBeenCalledTimes(1);
  vi.spyOn(Date, "now").mockReturnValue(now + 30_001);
  await trigger();
  expect(fetch).toHaveBeenCalledTimes(2);
});

test("StrictMode session updates retain the cache; account changes and sign-out replace it", async () => {
  const fetch = vi.fn().mockImplementation(() => Promise.resolve(Response.json({ notes: [note] })));
  await mount(fetch, true);
  const calls = fetch.mock.calls.length;
  await renderSession({ ...user });
  expect(fetch).toHaveBeenCalledTimes(calls);
  for (const next of [otherUser, null, user]) {
    fetch.mockImplementation(() => Promise.resolve(Response.json({ notes: [] })));
    await renderSession(next);
    expect(host.textContent).not.toContain(note.title);
  }
  expect(host.textContent).toContain("No notes yet.");
});

test.each(["read", "write", "malformed write"])(
  "a late %s cannot affect another account",
  async (operation) => {
    const pending = deferred();
    const fetch = vi.fn().mockImplementation((_path, options) => {
      if (operation === "read" || options?.method === "POST") {
        return pending.promise.then((response) => {
          options?.signal?.throwIfAborted();
          return response;
        });
      }
      return Promise.resolve(Response.json({ notes: [] }));
    });
    await mount(fetch, true);
    if (operation !== "read") await save();
    fetch.mockImplementation(() => Promise.resolve(Response.json({ notes: [] })));
    await renderSession(otherUser);
    const calls = fetch.mock.calls.length;
    await act(async () =>
      pending.resolve(
        operation === "read"
          ? Response.json({ notes: [note] })
          : Response.json(operation === "write" ? { note } : {}, { status: 201 }),
      ),
    );
    expect(fetch).toHaveBeenCalledTimes(calls);
    expect(host.querySelector("ul")!.textContent).toBe("");
    expect(host.textContent).not.toContain("Note saved.");
  },
);
