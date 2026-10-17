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
    return new Response(null, { status: 200 });
  });
  vi.stubGlobal("fetch", fetch);
  await expect(submitInferenceJob(env, request, 4)).rejects.toMatchObject({
    status: 413,
  });
  expect(fetch).toHaveBeenCalledTimes(1);
});

function uploadRequest() {
  return new Request("https://app.example.com/api/analyses", {
    method: "POST",
    headers: { "Content-Type": "multipart/form-data; boundary=test" },
    body: "video bytes",
  });
}

test("accepts the ML service's HTTP 200 acknowledgement", async () => {
  const jobId = "fc-01ABCDEFG123456789";
  const fetch = vi.fn().mockResolvedValue(Response.json({ job_id: jobId, status: "processing" }));
  vi.stubGlobal("fetch", fetch);
  expect(await submitInferenceJob(env, uploadRequest(), 4096)).toBe(jobId);
  expect(fetch).toHaveBeenCalledTimes(1);
  const headers = fetch.mock.calls[0]?.[1].headers;
  expect(headers).toMatchObject({ "X-API-Key": "service-key" });
  expect(headers).not.toHaveProperty("X-Job-ID");
});

test("an unsupported acknowledgement status is not treated as acceptance", async () => {
  const fetch = vi
    .fn()
    .mockResolvedValue(Response.json({ job_id: "fc-id", status: "processing" }, { status: 202 }));
  vi.stubGlobal("fetch", fetch);
  await expect(submitInferenceJob(env, uploadRequest(), 4096)).rejects.toMatchObject({
    code: "SUBMISSION_UNCONFIRMED",
  });
  expect(fetch).toHaveBeenCalledTimes(1);
});

test("rejects the old structured service failure format", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      Response.json({
        job_id: "fc-id",
        status: "failed",
        error: { code: "ANALYSIS_FAILED", message: "Old format" },
      }),
    ),
  );
  await expect(getInferenceJob(env, "fc-id")).rejects.toMatchObject({ code: "INVALID_RESPONSE" });
});

test.each([
  "invalid JSON",
  JSON.stringify({ job_id: "../../other-job", status: "processing" }),
  JSON.stringify({ status: "processing" }),
  JSON.stringify({ job_id: "fc-id", status: "done" }),
])("an unreadable acknowledgement cannot trigger a retry or speculative poll: %s", async (body) => {
  const fetch = vi.fn().mockResolvedValue(new Response(body));
  vi.stubGlobal("fetch", fetch);
  await expect(submitInferenceJob(env, uploadRequest(), 4096)).rejects.toMatchObject({
    code: "SUBMISSION_UNCONFIRMED",
  });
  expect(fetch).toHaveBeenCalledTimes(1);
});

test("a lost acknowledgement is not resubmitted", async () => {
  const fetch = vi.fn().mockRejectedValue(new TypeError("Connection lost"));
  vi.stubGlobal("fetch", fetch);
  await expect(submitInferenceJob(env, uploadRequest(), 4096)).rejects.toMatchObject({
    code: "SUBMISSION_UNCONFIRMED",
  });
  expect(fetch).toHaveBeenCalledTimes(1);
});

test("polls the service ID and normalizes ML failure strings", async () => {
  const jobId = "fc-01ABCDEFG123456789";
  const fetch = vi.fn().mockResolvedValue(
    Response.json({
      job_id: jobId,
      status: "failed",
      error: "RuntimeError: internal path /private/model",
    }),
  );
  vi.stubGlobal("fetch", fetch);
  expect(await getInferenceJob(env, jobId)).toEqual({
    job_id: jobId,
    status: "failed",
    error: { code: "ANALYSIS_FAILED", message: "Сервис не смог завершить анализ." },
  });
  expect(fetch.mock.calls[0]?.[0]).toBe(`${env.INFERENCE_URL}/api/v1/jobs/${jobId}`);
});

test("rejects a result for a different service job", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      Response.json({
        job_id: "fc-other",
        status: "processing",
      }),
    ),
  );
  await expect(getInferenceJob(env, "fc-requested")).rejects.toMatchObject({
    code: "INVALID_RESPONSE",
  });
});
