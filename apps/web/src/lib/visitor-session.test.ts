import { afterEach, beforeEach, expect, test, vi } from "vite-plus/test";
import { ensureVisitorSession } from "./visitor-session";

const auth = vi.hoisted(() => ({ getSession: vi.fn(), anonymous: vi.fn() }));
vi.mock("./auth-client", () => ({
  authClient: { getSession: auth.getSession, signIn: { anonymous: auth.anonymous } },
}));

beforeEach(() => {
  vi.resetAllMocks();
  vi.stubGlobal("navigator", { locks: { request: vi.fn(async (_name, callback) => callback()) } });
});
afterEach(() => vi.unstubAllGlobals());

test("concurrent callers share creation and verify the session cookie", async () => {
  auth.getSession
    .mockResolvedValueOnce({ data: null, error: null })
    .mockResolvedValueOnce({ data: { user: { id: "visitor" } }, error: null });
  auth.anonymous.mockResolvedValue({ error: null });
  await Promise.all([ensureVisitorSession(), ensureVisitorSession()]);
  expect(auth.anonymous).toHaveBeenCalledTimes(1);
});

test("the tab holding the browser lock rechecks cookies before creating a visitor", async () => {
  const request = vi.fn(async (_name, callback) => callback());
  vi.stubGlobal("navigator", { locks: { request } });
  auth.getSession.mockResolvedValue({ data: { user: { id: "other-tab-visitor" } }, error: null });
  await ensureVisitorSession();
  expect(request).toHaveBeenCalledTimes(1);
  expect(auth.anonymous).not.toHaveBeenCalled();
});

test("a failed acknowledgement preserves the established identity without another write", async () => {
  auth.getSession
    .mockResolvedValueOnce({ data: null, error: null })
    .mockResolvedValueOnce({ data: { user: { id: "accepted-visitor" } }, error: null });
  auth.anonymous.mockResolvedValue({ error: { status: 503 } });
  await expect(ensureVisitorSession()).resolves.toBeUndefined();
  expect(auth.anonymous).toHaveBeenCalledTimes(1);
});

test("a failed cookie lookup cannot replace an existing identity", async () => {
  auth.getSession.mockResolvedValueOnce({ data: null, error: { status: 503 } });
  await expect(ensureVisitorSession()).rejects.toThrow("Не удалось подготовить сеанс.");
  expect(auth.anonymous).not.toHaveBeenCalled();
  auth.getSession.mockResolvedValueOnce({
    data: { user: { id: "restored-visitor" } },
    error: null,
  });
  await expect(ensureVisitorSession()).resolves.toBeUndefined();
  expect(auth.anonymous).not.toHaveBeenCalled();
});
