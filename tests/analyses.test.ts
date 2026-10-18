import { randomUUID } from "node:crypto";
import { expect, inject, test } from "vite-plus/test";
import {
  jobIdSchema,
  jobSchema,
  reviewJobSchema,
  readLiveEvents,
  LIVE_REQUEST_LIMIT_BYTES,
} from "@app/contracts";
import evidence from "./fixtures/evidence.json";

declare module "vite-plus/test" {
  interface ProvidedContext {
    evaluationKey: string;
    inferenceOrigin: string;
    inferenceKey: string;
  }
}

const origin = inject("origin");
const key = inject("evaluationKey");
const headers = { "X-API-Key": key };

function upload(
  name: string,
  customHeaders: HeadersInit = headers,
  path = "/api/v1/analyze",
  size = 16,
) {
  const body = new FormData();
  body.append("file", new Blob([new Uint8Array(size)]), name);
  return fetch(`${origin}${path}`, { method: "POST", headers: customHeaders, body });
}

async function submit(name: string) {
  const response = await upload(name);
  expect(response.status, await response.clone().text()).toBe(202);
  return ((await response.json()) as { job_id: string }).job_id;
}

const invalidKeys: HeadersInit[] = [{}, { "X-API-Key": "wrong" }];
test.each(invalidKeys)("evaluation endpoints require the evaluator key", async (invalid) => {
  expect((await upload("negative.mov", invalid)).status).toBe(401);
  expect((await fetch(`${origin}/api/v1/jobs/${randomUUID()}`, { headers: invalid })).status).toBe(
    401,
  );
});

test.each(["negative.mov", "positive.mov", "processing.mov", "failed.mov"])(
  "serves the jury contract for %s",
  async (name) => {
    const id = await submit(name);
    const response = await fetch(`${origin}/api/v1/jobs/${id}`, { headers });
    expect(response.status).toBe(200);
    expect(response.headers.get("cache-control")).toContain("no-store");
    const job = jobSchema.parse(await response.json());
    expect(job.job_id).toBe(id);
    if (job.status === "done") {
      expect(job.verdict).toBe(name === "negative.mov" ? "no_violation" : "violation");
      expect(job.violations).toHaveLength(name === "negative.mov" ? 0 : 2);
      if (job.violations.length) {
        expect(job.violations[0]?.description).toContain("НБРК №231, п. 12");
        expect(job.violations[0]).not.toHaveProperty("rule");
      }
    } else {
      expect(job.status).toBe(name === "processing.mov" ? "processing" : "failed");
      if (job.status === "failed") {
        expect(job.error).toMatchObject({ code: "ANALYSIS_FAILED" });
        expect(job.error.message).not.toContain("/private/");
      }
    }
  },
);

test.each(["lost-ack.mov", "malformed-ack.mov"])(
  "reports %s without resubmitting or polling an invented upstream ID",
  async (name) => {
    const response = await upload(name);
    expect(response.status).toBe(503);
    const { error } = (await response.json()) as { error: { code: string } };
    expect(error.code).toBe("SUBMISSION_UNCONFIRMED");
    expect(error).not.toHaveProperty("job_id");
    const record = await fetch(`${inject("inferenceOrigin")}/__test/submissions/${name}`, {
      headers: { "X-API-Key": inject("inferenceKey") },
    });
    expect(await record.json()).toMatchObject({ submissions: 1 });
  },
);

test("rejects invalid media, oversize bodies and overload", async () => {
  expect((await upload("invalid.mov")).status).toBe(422);
  // The local Vite/Worker bridge can leave a rejected upload body unread.
  // Isolate this connection so the next request does not reuse that socket.
  expect(
    (await upload("negative.mov", { ...headers, Connection: "close" }, "/api/v1/analyze", 5000))
      .status,
  ).toBe(413);
  expect((await upload("busy.mov")).status).toBe(429);
  expect(
    (await fetch(`${origin}/api/v1/analyze`, { method: "POST", headers, body: "not multipart" }))
      .status,
  ).toBe(400);
});

test("invalid upstream results and transport outages do not become negative results", async () => {
  for (const [name, status] of [
    ["bad-result.mov", 502],
    ["outage.mov", 503],
  ] as const) {
    const id = await submit(name);
    const response = await fetch(`${origin}/api/v1/jobs/${id}`, { headers });
    expect(response.status).toBe(status);
    expect(await response.json()).not.toHaveProperty("verdict");
  }
});

let accountIp = 200;
async function account() {
  const response = await fetch(`${origin}/api/auth/sign-up/email`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Origin: origin,
      "CF-Connecting-IP": `192.0.2.${++accountIp}`,
    },
    body: JSON.stringify({
      name: "Reviewer",
      email: `${randomUUID()}@example.com`,
      password: randomUUID(),
    }),
  });
  expect(response.status).toBe(200);
  return {
    Cookie: response.headers
      .getSetCookie()
      .map((v) => v.split(";")[0])
      .join("; "),
    Origin: origin,
  };
}

test("session ownership cannot be forged and is separate from evaluator ownership", async () => {
  const alice = await account(),
    bob = await account();
  const created = await upload(
    "positive.mov",
    { ...alice, "X-Job-Owner": "user:forged" },
    "/api/analyses",
  );
  expect(created.status).toBe(202);
  const { job_id: id } = (await created.json()) as { job_id: string };
  expect((await fetch(`${origin}/api/analyses/${id}`, { headers: bob })).status).toBe(404);
  expect((await fetch(`${origin}/api/v1/jobs/${id}`, { headers })).status).toBe(404);
  const result = await fetch(`${origin}/api/analyses/${id}`, { headers: alice });
  expect(await result.json()).toMatchObject({
    job_id: id,
    expires_at: expect.any(String),
    violations: [expect.objectContaining({ type: expect.any(String) }), expect.any(Object)],
  });
});

test("maps Modal IDs to owned UUIDs and serves the ML result contract", async () => {
  const alice = await account();
  const response = await upload("positive.mov", alice, "/api/analyses");
  expect(response.status).toBe(202);
  const { job_id: id } = (await response.json()) as { job_id: string };
  expect(jobIdSchema.safeParse(id).success).toBe(true);
  const result = await fetch(`${origin}/api/analyses/${id}`, { headers: alice });
  expect(result.status).toBe(200);
  const job = reviewJobSchema.parse(await result.json());
  expect(job.job_id).toBe(id);
  expect(job.status).toBe("done");
  if (job.status === "done") {
    expect(job).not.toHaveProperty("duration_sec");
    expect(job.violations[0]).not.toHaveProperty("rule");
    expect(job.violations[0]?.evidence_frames).toHaveLength(1);
  }
  const records = await fetch(`${inject("inferenceOrigin")}/__test/submissions/positive.mov`, {
    headers: { "X-API-Key": inject("inferenceKey") },
  });
  const { ids } = (await records.json()) as { ids: string[] };
  expect(ids.length).toBeGreaterThan(0);
  expect(ids.every((upstreamId) => upstreamId.startsWith("fc-") && upstreamId !== id)).toBe(true);
  expect((await fetch(`${origin}/api/analyses/${ids[0]}`, { headers: alice })).status).toBe(404);
});

function liveBody(start = 0) {
  return JSON.stringify({
    frames: [0, 1, 2, 3].map((i) => ({ timestamp_sec: start + i, image_base64: evidence.jpeg })),
  });
}

test("live inference requires a signed-in user and never accepts an evaluator key as a session", async () => {
  for (const customHeaders of [{}, headers]) {
    const response = await fetch(`${origin}/api/live`, {
      method: "POST",
      headers: { ...customHeaders, "Content-Type": "application/json" },
      body: liveBody(),
    });
    expect(response.status).toBe(401);
  }
});

test("live detections and verified findings stream through the authenticated API", async () => {
  const alice = await account();
  const response = await fetch(`${origin}/api/live`, {
    method: "POST",
    headers: { ...alice, "Content-Type": "application/json" },
    body: liveBody(20),
  });
  expect(response.status).toBe(200);
  expect(response.headers.get("Content-Type")).toContain("text/event-stream");
  expect(response.headers.get("Cache-Control")).toContain("no-store");
  const events = [];
  for await (const event of readLiveEvents(response.body!)) events.push(event);
  expect(events.map((event) => event.type)).toEqual(["stage", "frame", "result"]);
  expect(events[2]).toMatchObject({ from_sec: 20, to_sec: 23, violations: [{ timecode_sec: 23 }] });
});

test("live windows reject invalid and excessive inputs before contacting inference", async () => {
  const alice = await account();
  for (const [body, status] of [
    [JSON.stringify({ frames: [] }), 400],
    [" ".repeat(LIVE_REQUEST_LIMIT_BYTES + 1), 413],
  ] as const) {
    const response = await fetch(`${origin}/api/live`, {
      method: "POST",
      headers: { ...alice, "Content-Type": "application/json", Connection: "close" },
      body,
    });
    expect(response.status).toBe(status);
  }
});

test.each([1000, 1100, 1200, 1300])(
  "interrupted or invalid live inference at %s never becomes a clean result",
  async (start) => {
    const alice = await account();
    const response = await fetch(`${origin}/api/live`, {
      method: "POST",
      headers: { ...alice, "Content-Type": "application/json" },
      body: liveBody(start),
    });
    expect(response.status).toBe(200);
    const events = [];
    for await (const event of readLiveEvents(response.body!)) events.push(event);
    expect(events.at(-1)).toMatchObject({ type: "error", error: { code: "LIVE_FAILED" } });
    expect(events.some((event) => event.type === "result")).toBe(false);
    expect(JSON.stringify(events)).not.toContain("token-secret");
    expect(JSON.stringify(events)).not.toContain("/model/");
  },
);
