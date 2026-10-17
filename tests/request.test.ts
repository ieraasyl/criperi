import { afterEach, expect, test, vi } from "vite-plus/test";
import { notesResponseSchema } from "../packages/contracts/src/index.ts";
import { ApiRequestError, request } from "../apps/web/src/lib/request.ts";

afterEach(() => vi.unstubAllGlobals());

test("parses successful responses using the supplied schema and forwards options", async () => {
  const fetch = vi.fn().mockResolvedValue(Response.json({ notes: [] }));
  vi.stubGlobal("fetch", fetch);
  const options = { signal: new AbortController().signal };

  expect(await request("/api/notes", notesResponseSchema, options)).toEqual({ notes: [] });
  expect(fetch).toHaveBeenCalledWith("/api/notes", options);
});

test("preserves validation details and HTTP status", async () => {
  const error = {
    code: "VALIDATION_ERROR",
    message: "Check the submitted fields.",
    issues: [{ path: ["title"], message: "Too long" }],
  };
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json({ error }, { status: 400 })));

  await expect(request("/api/notes", notesResponseSchema)).rejects.toMatchObject({
    ...error,
    status: 400,
  });
});

test.each([
  { status: 401, code: "UNAUTHORIZED", message: "Your session has expired. Please sign in again." },
  { status: 500, code: "HTTP_ERROR", message: "Something went wrong. Please try again." },
])("distinguishes a $status response even without JSON", async ({ status, code, message }) => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("failure", { status })));

  await expect(request("/api/notes", notesResponseSchema)).rejects.toMatchObject({
    status,
    code,
    message,
  });
});

test.each(["not JSON", JSON.stringify({ notes: "invalid" })])(
  "rejects invalid success bodies",
  async (body) => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body, { status: 201 })));

    await expect(request("/api/notes", notesResponseSchema)).rejects.toMatchObject({
      status: 201,
      code: "INVALID_RESPONSE",
    });
  },
);

test("distinguishes connection failures from server failures", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

  await expect(request("/api/notes", notesResponseSchema)).rejects.toBeInstanceOf(ApiRequestError);
  await expect(request("/api/notes", notesResponseSchema)).rejects.toMatchObject({
    status: 0,
    code: "NETWORK_ERROR",
  });
});

test("preserves cancellation", async () => {
  const controller = new AbortController();
  const cause = new DOMException("Aborted", "AbortError");
  controller.abort();
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(cause));

  await expect(
    request("/api/notes", notesResponseSchema, { signal: controller.signal }),
  ).rejects.toBe(cause);
});
