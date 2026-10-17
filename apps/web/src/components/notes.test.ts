// @vitest-environment jsdom
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, expect, test, vi } from "vite-plus/test";
import { Notes } from "./notes";

vi.mock("@/lib/auth-client", () => ({ authClient: { signOut: vi.fn() } }));

const note = {
  id: "note-1",
  userId: "user-1",
  title: "Saved note",
  createdAt: "2026-10-16T12:00:00.000Z",
};
let root: Root | undefined;
let host: HTMLDivElement;

function deferred() {
  let resolve!: (value: Response) => void;
  const promise = new Promise<Response>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

async function mount(fetch: ReturnType<typeof vi.fn>) {
  vi.stubGlobal("fetch", fetch);
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  await act(async () => {
    root!.render(createElement(Notes, { user: { id: "user-1", email: "user@example.com" } }));
  });
}

async function save() {
  host.querySelector<HTMLInputElement>('[name="title"]')!.value = "Saved note";
  await act(async () => {
    host
      .querySelector("form")!
      .dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
  });
}

function failure() {
  return Response.json(
    { error: { code: "INTERNAL_ERROR", message: "Please try again." } },
    { status: 500 },
  );
}

afterEach(async () => {
  await act(async () => root?.unmount());
  root = undefined;
  host?.remove();
  vi.unstubAllGlobals();
});

test("a successful save survives refresh failure and retry only reloads the list", async () => {
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(Response.json({ notes: [] }))
    .mockResolvedValueOnce(Response.json({ note }, { status: 201 }))
    .mockResolvedValueOnce(failure())
    .mockResolvedValueOnce(Response.json({ notes: [note] }));
  await mount(fetch);
  await save();

  expect(host.querySelector<HTMLInputElement>('[name="title"]')!.value).toBe("");
  expect(host.querySelector("ul")!.textContent).toContain(note.title);
  expect(host.textContent).toContain("Note saved.");
  expect(host.querySelector('[role="alert"]')!.textContent).toContain("Could not load notes.");

  const retry = Array.from(host.querySelectorAll("button")).find(
    (button) => button.textContent === "Retry loading notes",
  )!;
  await act(async () => retry.click());

  expect(host.querySelector('[role="alert"]')).toBeNull();
  expect(fetch.mock.calls.filter(([, options]) => options?.method === "POST")).toHaveLength(1);
  expect(fetch).toHaveBeenCalledTimes(4);
});

test("an older initial load cannot overwrite the saved and refreshed list", async () => {
  const initial = deferred();
  const fetch = vi
    .fn()
    .mockReturnValueOnce(initial.promise)
    .mockResolvedValueOnce(Response.json({ note }, { status: 201 }))
    .mockResolvedValueOnce(Response.json({ notes: [note] }));
  await mount(fetch);
  expect(host.textContent).toContain("Loading notes…");
  await save();
  await act(async () => initial.resolve(Response.json({ notes: [] })));

  expect(fetch.mock.calls[0]![1].signal.aborted).toBe(true);
  expect(host.querySelector("ul")!.textContent).toBe(note.title);
  expect(host.textContent).not.toContain("Loading notes…");
});

test("initial load errors expose a loading retry", async () => {
  const retry = deferred();
  const fetch = vi.fn().mockResolvedValueOnce(failure()).mockReturnValueOnce(retry.promise);
  await mount(fetch);
  const button = Array.from(host.querySelectorAll("button")).find(
    (item) => item.textContent === "Retry loading notes",
  )!;
  await act(async () => button.click());
  expect(host.textContent).toContain("Loading notes…");
  expect(host.querySelector('[role="alert"]')).toBeNull();

  await act(async () => retry.resolve(Response.json({ notes: [note] })));
  expect(host.querySelector("ul")!.textContent).toBe(note.title);
});

test("a failed save retains the input and does not refresh the list", async () => {
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(Response.json({ notes: [] }))
    .mockResolvedValueOnce(failure());
  await mount(fetch);
  await save();

  expect(host.querySelector<HTMLInputElement>('[name="title"]')!.value).toBe("Saved note");
  expect(host.textContent).not.toContain("Note saved.");
  expect(fetch).toHaveBeenCalledTimes(2);
});

test("a confirmed write with an invalid response clears the input and reloads", async () => {
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(Response.json({ notes: [] }))
    .mockResolvedValueOnce(Response.json({}, { status: 201 }))
    .mockResolvedValueOnce(Response.json({ notes: [note] }));
  await mount(fetch);
  await save();

  expect(host.querySelector<HTMLInputElement>('[name="title"]')!.value).toBe("");
  expect(host.textContent).toContain("Note saved.");
  expect(host.querySelector("ul")!.textContent).toBe(note.title);
});
