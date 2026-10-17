import { Hono } from "hono";
import { beforeEach, expect, test, vi } from "vite-plus/test";
import { createDb } from "@app/db";
import { handleError } from "../errors.ts";
import { AnalysisError, submitInferenceJob } from "../inference.ts";
import { evaluationRoutes } from "./analyses.ts";

vi.mock("@app/db", async (original) => ({
  ...(await original<typeof import("@app/db")>()),
  createDb: vi.fn(),
}));
vi.mock("../inference.ts", async (original) => ({
  ...(await original<typeof import("../inference.ts")>()),
  submitInferenceJob: vi.fn(),
}));

const env = {
  EVALUATION_API_KEY: "evaluation-key",
  INFERENCE_URL: "https://inference.example.com",
  INFERENCE_API_KEY: "service-key",
};
const headers = { "X-API-Key": env.EVALUATION_API_KEY };
const app = new Hono().onError(handleError).route("/api/v1", evaluationRoutes);

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
