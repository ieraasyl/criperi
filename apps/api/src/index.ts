import { Hono } from "hono";
import { createAuth } from "./auth.ts";
import { requireSession, type AuthenticatedEnv } from "./middleware/session.ts";
import { aiRoutes } from "./routes/ai.ts";
import { notesRoutes } from "./routes/notes.ts";
import { handleError } from "./errors.ts";
import type { Bindings } from "./env.ts";

const app = new Hono<{ Bindings: Bindings }>();

app.onError(handleError);
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
  .route("/api/ai", aiRoutes);

export type AppType = typeof routes;

export default app;
