import { Hono } from "hono";
import { createAuth } from "./auth.ts";
import { requireSession, type AuthenticatedEnv } from "./middleware/session.ts";
import { analysisRoutes, evaluationRoutes } from "./routes/analyses.ts";
import { liveRoutes } from "./routes/live.ts";
import { createDb, analysisJobs, lte } from "@app/db";
import { notesRoutes } from "./routes/notes.ts";
import { handleError } from "./errors.ts";
import type { Bindings } from "./env.ts";

const app = new Hono<{ Bindings: Bindings }>();

app.onError(handleError);
app.use("/api/*", async (c, next) => {
  c.header("Cache-Control", "private, no-store");
  await next();
});
app.notFound((c) => c.json({ error: { code: "NOT_FOUND", message: "Endpoint not found." } }, 404));

app.all("/api/auth/*", (c) => {
  return createAuth(c.env).handler(c.req.raw);
});

const meRoutes = new Hono<AuthenticatedEnv>().use("*", requireSession).get("/", (c) => {
  return c.json({ user: c.get("session").user });
});

const routes = app
  .get("/api/health", (c) => c.json({ ok: true }))
  .route("/api/me", meRoutes)
  .route("/api/notes", notesRoutes)
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
