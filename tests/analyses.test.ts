import { randomUUID } from "node:crypto";
import { expect, inject, test } from "vite-plus/test";
import { jobIdSchema, jobSchema, reviewJobSchema } from "@app/contracts";

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
  expect((await upload("negative.mov", headers, "/api/v1/analyze", 5000)).status).toBe(413);
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

test("review decisions persist per event and cannot be read or changed by another account", async () => {
  const alice = await account(),
    bob = await account();
  const created = await upload("positive.mov", alice, "/api/analyses");
  const { job_id: id } = (await created.json()) as { job_id: string };
  const endpoint = `${origin}/api/analyses/${id}/reviews/0`;
  const save = (headers: Record<string, string>, json: unknown, path = endpoint) =>
    fetch(path, {
      method: "PUT",
      headers: { ...headers, "Content-Type": "application/json" },
      body: JSON.stringify(json),
    });
  expect((await save(bob, { status: "confirmed", comment: "forged" })).status).toBe(404);
  expect((await save({}, { status: "confirmed" })).status).toBe(401);
  expect((await save(alice, { status: "unknown" })).status).toBe(400);
  expect((await save(alice, { status: "confirmed", comment: "x".repeat(4001) })).status).toBe(400);
  expect((await save(alice, { status: "confirmed" }, endpoint.replace(/0$/, "99"))).status).toBe(
    404,
  );
  const first = await save(alice, { status: "confirmed", comment: "  Проверено по кадрам.  " });
  expect(first.status).toBe(200);
  expect(await first.json()).toMatchObject({
    review: {
      event_index: 0,
      status: "confirmed",
      comment: "Проверено по кадрам.",
      updated_at: expect.any(String),
    },
  });
  const second = await fetch(`${origin}/api/analyses/${id}/reviews/1`, {
    method: "PUT",
    headers: { ...alice, "Content-Type": "application/json" },
    body: JSON.stringify({ status: "needs_review", comment: "Нужно видео." }),
  });
  expect(second.status).toBe(200);
  await save(alice, { status: "rejected", comment: "Ложное срабатывание." });
  const restored = reviewJobSchema.parse(
    await (await fetch(`${origin}/api/analyses/${id}`, { headers: alice })).json(),
  );
  expect(restored.reviews).toHaveLength(2);
  expect(restored.reviews.find((item) => item.event_index === 0)).toMatchObject({
    status: "rejected",
    comment: "Ложное срабатывание.",
  });
  expect(restored.reviews.find((item) => item.event_index === 1)).toMatchObject({
    status: "needs_review",
    comment: "Нужно видео.",
  });
  expect((await fetch(`${origin}/api/analyses/${id}`, { headers: bob })).status).toBe(404);
});
