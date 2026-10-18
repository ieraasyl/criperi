import { Hono } from "hono";
import { beforeEach, expect, test, vi } from "vite-plus/test";
import { createDb } from "@app/db";
import { handleError } from "../errors.ts";
import { AnalysisError, getInferenceJob, submitInferenceJob } from "../inference.ts";
import { createAuth } from "../auth.ts";
import { analysisRoutes, evaluationRoutes } from "./analyses.ts";

vi.mock("@app/db", async (original) => ({
  ...(await original<typeof import("@app/db")>()),
  createDb: vi.fn(),
}));
vi.mock("../inference.ts", async (original) => ({
  ...(await original<typeof import("../inference.ts")>()),
  submitInferenceJob: vi.fn(),
  getInferenceJob: vi.fn(),
}));
vi.mock("../auth.ts", () => ({ createAuth: vi.fn() }));

const env = {
  EVALUATION_API_KEY: "evaluation-key",
  INFERENCE_URL: "https://inference.example.com",
  INFERENCE_API_KEY: "service-key",
};
const headers = { "X-API-Key": env.EVALUATION_API_KEY };
const app = new Hono()
  .onError(handleError)
  .route("/api/v1", evaluationRoutes)
  .route("/api/analyses", analysisRoutes);

beforeEach(() => vi.resetAllMocks());

test("a database outage after acceptance prevents resubmission", async () => {
  const db = {
    insert: () => ({ values: vi.fn().mockRejectedValue(new Error("Database unavailable")) }),
  };
  vi.mocked(createDb).mockReturnValue(db as unknown as ReturnType<typeof createDb>);
  vi.mocked(submitInferenceJob).mockResolvedValue("fc-accepted");
  const body = new FormData();
  body.append("file", new Blob(["video"]), "test.mov");
  const response = await app.request("/api/v1/analyze", { method: "POST", headers, body }, env);
  expect(response.status).toBe(503);
  const { error } = (await response.json()) as { error: { code: string } };
  expect(error.code).toBe("SUBMISSION_UNCONFIRMED");
  expect(error).not.toHaveProperty("job_id");
  expect(submitInferenceJob).toHaveBeenCalledTimes(1);
});

test("an unconfirmed ML submission does not create a placeholder job", async () => {
  vi.mocked(submitInferenceJob).mockRejectedValue(
    new AnalysisError("SUBMISSION_UNCONFIRMED", "Upload acknowledgement lost"),
  );
  const body = new FormData();
  body.append("file", new Blob(["video"]), "test.mov");
  const response = await app.request("/api/v1/analyze", { method: "POST", headers, body }, env);
  expect(response.status).toBe(503);
  expect(createDb).not.toHaveBeenCalled();
  expect(submitInferenceJob).toHaveBeenCalledTimes(1);
});

// Test early body rejection without the local Vite proxy, which resets its pooled connection
// when the Worker returns before consuming the request body.
test.each(["/api/v1/analyze", "/api/analyses"])(
  "%s rejects oversized uploads before contacting inference or writing a job",
  async (path) => {
    vi.mocked(createAuth).mockReturnValue({
      api: { getSession: async () => ({ user: { id: "visitor", isAnonymous: true } }) },
    } as unknown as ReturnType<typeof createAuth>);
    const body = new FormData();
    body.append("file", new Blob([new Uint8Array(5000)]), "test.mov");
    const response = await app.request(
      path,
      {
        method: "POST",
        headers: { ...headers, "Content-Length": "5200" },
        body,
      },
      { ...env, UPLOAD_LIMIT_BYTES: "4096" },
    );
    expect(response.status).toBe(413);
    expect(await response.json()).toMatchObject({ error: { code: "UPLOAD_TOO_LARGE" } });
    expect(submitInferenceJob).not.toHaveBeenCalled();
    expect(createDb).not.toHaveBeenCalled();
  },
);

test.each([
  { path: "/api/analyses/280312ee-e063-42b9-a846-4a18fe361682", method: "GET" },
  { path: "/api/analyses/280312ee-e063-42b9-a846-4a18fe361682/reviews/0", method: "PUT" },
  { path: "/api/v1/jobs/280312ee-e063-42b9-a846-4a18fe361682", method: "GET" },
])(
  "expired jobs reject $method $path before inference or review writes",
  async ({ path, method }) => {
    const db = {
      select: () => ({
        from: () => ({ where: () => ({ limit: async () => [{ expiresAt: Date.now() - 1 }] }) }),
      }),
      insert: vi.fn(),
    };
    vi.mocked(createDb).mockReturnValue(db as unknown as ReturnType<typeof createDb>);
    vi.mocked(createAuth).mockReturnValue({
      api: { getSession: async () => ({ user: { id: "visitor", isAnonymous: true } }) },
    } as unknown as ReturnType<typeof createAuth>);
    const response = await app.request(
      path,
      {
        method,
        headers: { ...headers, "Content-Type": "application/json" },
        ...(method === "PUT" ? { body: JSON.stringify({ status: "confirmed", comment: "" }) } : {}),
      },
      env,
    );
    expect(response.status).toBe(410);
    expect(await response.json()).toMatchObject({ error: { code: "EXPIRED" } });
    expect(getInferenceJob).not.toHaveBeenCalled();
    expect(db.insert).not.toHaveBeenCalled();
  },
);
