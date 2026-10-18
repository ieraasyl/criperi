import { Hono } from "hono";
import { createAuth } from "./auth.ts";
import { analysisRoutes, evaluationRoutes } from "./routes/analyses.ts";
import { liveRoutes } from "./routes/live.ts";
import { createDb, analysisJobs, lte } from "@app/db";
import { handleError } from "./errors.ts";
import type { Bindings } from "./env.ts";

const app = new Hono<{ Bindings: Bindings }>();

app.onError(handleError);
app.use("/api/*", async (c, next) => {
  await next();
  c.header("Cache-Control", "private, no-store");
});
app.notFound((c) => c.json({ error: { code: "NOT_FOUND", message: "Endpoint not found." } }, 404));

app.use("/api/auth/*", async (c, next) => {
  if (!["GET", "HEAD", "OPTIONS"].includes(c.req.method)) {
    const origin = c.req.header("Origin");
    if (
      (origin && origin !== new URL(c.env.BETTER_AUTH_URL).origin) ||
      c.req.header("Sec-Fetch-Site") === "cross-site"
    ) {
      return c.json(
        { error: { code: "FORBIDDEN", message: "Недопустимый источник запроса." } },
        403,
      );
    }
  }
  await next();
});

app.get("/api/auth/get-session", (c) => {
  return createAuth(c.env).handler(c.req.raw);
});

app.post("/api/auth/sign-in/anonymous", (c) => {
  return createAuth(c.env).handler(c.req.raw);
});

const routes = app
  .get("/api/health", (c) => c.json({ ok: true }))
  .route("/api/analyses", analysisRoutes)
  .route("/api/live", liveRoutes)
  .route("/api/v1", evaluationRoutes);

export type AppType = typeof routes;

export default {
  fetch: app.fetch,
  async scheduled(_event: unknown, env: Bindings) {
    await createDb(env.DB).delete(analysisJobs).where(lte(analysisJobs.expiresAt, Date.now()));
  },
};
