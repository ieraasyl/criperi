import { Hono } from "hono";
import { createAuth } from "./auth.ts";
import { requireSession, type AuthenticatedEnv } from "./middleware/session.ts";
import { aiRoutes } from "./routes/ai.ts";
import { notesRoutes } from "./routes/notes.ts";

const app = new Hono<AuthenticatedEnv>();

app.all("/api/auth/*", (c) => {
  return createAuth(c.env).handler(c.req.raw);
});

app.get("/api/health", (c) => {
  return c.json({ ok: true });
});

app.get("/api/me", requireSession, (c) => {
  return c.json({ user: c.get("session").user });
});

app.route("/api/notes", notesRoutes);
app.route("/api/ai", aiRoutes);

export default app;
