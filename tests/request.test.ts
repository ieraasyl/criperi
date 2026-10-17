import { afterEach, expect, test, vi } from "vite-plus/test";
import { api } from "../apps/web/src/lib/api.ts";
import { ApiRequestError, request } from "../apps/web/src/lib/request.ts";

afterEach(() => vi.unstubAllGlobals());

function loadNotes(options?: RequestInit) {
  return request((init) => api.notes.$get({}, { init }), options);
}

test("parses successful RPC responses and forwards request options", async () => {
  const fetch = vi.fn().mockResolvedValue(Response.json({ notes: [] }));
  vi.stubGlobal("fetch", fetch);
  const options = { signal: new AbortController().signal };

  expect(await loadNotes(options)).toEqual({ notes: [] });
  expect(fetch).toHaveBeenCalledWith(
    "/api/notes",
    expect.objectContaining({ ...options, method: "GET" }),
  );
});

test("preserves validation details and HTTP status", async () => {
  const error = {
    code: "VALIDATION_ERROR",
    message: "Check the submitted fields.",
    issues: [{ path: ["title"], message: "Too long" }],
  };
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json({ error }, { status: 400 })));

  await expect(loadNotes()).rejects.toMatchObject({
    ...error,
    status: 400,
  });
});

test.each([
  { status: 401, code: "UNAUTHORIZED", message: "Your session has expired. Please sign in again." },
  { status: 500, code: "HTTP_ERROR", message: "Something went wrong. Please try again." },
])("distinguishes a $status response even without JSON", async ({ status, code, message }) => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("failure", { status })));

  await expect(loadNotes()).rejects.toMatchObject({
    status,
    code,
    message,
  });
});

test("preserves errors from global handlers", async () => {
  const error = { code: "INTERNAL_ERROR", message: "Something went wrong. Please try again." };
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json({ error }, { status: 500 })));

  await expect(loadNotes()).rejects.toMatchObject({ ...error, status: 500 });
});

test("rejects successful responses containing invalid JSON", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("not JSON", { status: 201 })));

  await expect(loadNotes()).rejects.toMatchObject({ status: 201, code: "INVALID_RESPONSE" });
});

test("trusts successful JSON without validating its shape", async () => {
  const body = { notes: "not schema-validated" };
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(body)));

  expect(await loadNotes()).toEqual(body);
});

test("distinguishes connection failures from server failures", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

  await expect(loadNotes()).rejects.toBeInstanceOf(ApiRequestError);
  await expect(loadNotes()).rejects.toMatchObject({
    status: 0,
    code: "NETWORK_ERROR",
  });
});

test("preserves cancellation", async () => {
  const controller = new AbortController();
  const cause = new DOMException("Aborted", "AbortError");
  controller.abort();
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(cause));

  await expect(loadNotes({ signal: controller.signal })).rejects.toBe(cause);
});

test("preserves cancellation while reading the response body", async () => {
  const controller = new AbortController();
  const cause = new Error("Reading cancelled");
  const response = Response.json({ notes: [] });
  vi.spyOn(response, "json").mockImplementation(async () => {
    controller.abort(cause);
    throw cause;
  });
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response));

  await expect(loadNotes({ signal: controller.signal })).rejects.toBe(cause);
});
