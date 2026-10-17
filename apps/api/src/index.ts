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

app.get("/api/health", (c) => {
  return c.json({ ok: true });
});

const meRoutes = new Hono<AuthenticatedEnv>();
meRoutes.use("*", requireSession);
meRoutes.get("/", (c) => {
  return c.json({ user: c.get("session").user });
});

app.route("/api/me", meRoutes);
app.route("/api/notes", notesRoutes);
app.route("/api/ai", aiRoutes);

export default app;
