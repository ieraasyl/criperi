import { afterEach, expect, test, vi } from "vite-plus/test";
import { getInferenceJob, inferenceConfig, submitInferenceJob } from "./inference.ts";
import type { Bindings } from "./env.ts";

const id = "a97aa7d9-d1d9-467b-980b-7a8b9fd4b251";
const env = {
  INFERENCE_URL: "https://inference.example.com",
  INFERENCE_API_KEY: "service-key",
} as Bindings;

afterEach(() => vi.unstubAllGlobals());

test.each([
  "not a url",
  "http://inference.example.com",
  "https://user:password@example.com",
  "https://example.com/path",
  "https://example.com?key=value",
])("rejects invalid or ambiguous inference origins: %s", (url) => {
  expect(() => inferenceConfig({ ...env, INFERENCE_URL: url })).toThrow("Некорректный адрес");
});

test("cancels upstream error bodies and never follows redirects with the service key", async () => {
  let cancelled = false;
  const response = new Response(
    new ReadableStream({
      cancel() {
        cancelled = true;
      },
    }),
    { status: 302, headers: { Location: "https://other.example.com" } },
  );
  const fetch = vi.fn().mockResolvedValue(response);
  vi.stubGlobal("fetch", fetch);
  await expect(getInferenceJob(env, id)).rejects.toMatchObject({ code: "INFERENCE_UNAVAILABLE" });
  expect(cancelled).toBe(true);
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(fetch.mock.calls[0]?.[1]).toMatchObject({
    redirect: "manual",
    headers: { "X-API-Key": "service-key" },
  });
});

test("preserves cancellation while polling", async () => {
  const controller = new AbortController();
  const reason = new DOMException("Cancelled", "AbortError");
  vi.stubGlobal(
    "fetch",
    vi.fn().mockImplementation(() => {
      controller.abort(reason);
      throw reason;
    }),
  );
  await expect(getInferenceJob(env, id, controller.signal)).rejects.toBe(reason);
});

test("a malformed or oversized result is rejected and its stream is released", async () => {
  const invalid = new Response("not JSON");
  const fetch = vi.fn().mockResolvedValueOnce(invalid);
  vi.stubGlobal("fetch", fetch);
  await expect(getInferenceJob(env, id)).rejects.toMatchObject({ code: "INVALID_RESPONSE" });
  expect(invalid.body?.locked).toBe(false);
  let cancelled = false;
  const oversized = new Response(
    new ReadableStream({
      start(controller) {
        controller.enqueue(new Uint8Array(16 * 1024 * 1024 + 1));
      },
      cancel() {
        cancelled = true;
      },
    }),
  );
  fetch.mockResolvedValueOnce(oversized);
  await expect(getInferenceJob(env, id)).rejects.toMatchObject({ code: "INVALID_RESPONSE" });
  expect(cancelled).toBe(true);
  expect(oversized.body?.locked).toBe(false);
});

test("a streamed upload exceeding its limit cannot be acknowledged as accepted", async () => {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new Uint8Array(8));
      controller.close();
    },
  });
  const request = new Request("https://app.example.com/api/analyses", {
    method: "POST",
    headers: { "Content-Type": "multipart/form-data; boundary=test" },
    body,
    duplex: "half",
  } as RequestInit);
  const fetch = vi.fn().mockImplementation(async (_url, init) => {
    const reader = init.body.getReader();
    try {
      while (!(await reader.read()).done) {
        /* Drain the bounded stream. */
      }
    } catch {
      /* Simulate a server that still attempts to acknowledge a truncated upload. */
    } finally {
      reader.releaseLock();
    }
    return new Response(null, { status: 202 });
  });
  vi.stubGlobal("fetch", fetch);
  await expect(submitInferenceJob(env, id, request, 4, Date.now())).rejects.toMatchObject({
    status: 413,
  });
  expect(fetch).toHaveBeenCalledTimes(1);
});
