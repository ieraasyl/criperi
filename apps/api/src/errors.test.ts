import { Hono } from "hono";
import { HTTPException } from "hono/http-exception";
import { afterEach, expect, test, vi } from "vite-plus/test";
import { handleError } from "./errors.ts";

afterEach(() => vi.restoreAllMocks());

test.each([
  new Error("private database details"),
  new HTTPException(503, { message: "private upstream details" }),
])("unexpected failures return safe JSON and are logged", async (failure) => {
  const log = vi.spyOn(console, "error").mockImplementation(() => {});
  const app = new Hono().onError(handleError).get("/", () => {
    throw failure;
  });
  const response = await app.request("/");

  expect(response.status).toBe(failure instanceof HTTPException ? 503 : 500);
  expect(await response.json()).toEqual({
    error: { code: "INTERNAL_ERROR", message: "Something went wrong. Please try again." },
  });
  expect(log).toHaveBeenCalledWith(failure);
});
