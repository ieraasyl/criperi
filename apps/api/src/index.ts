import { Hono } from "hono";
import { createMiddleware } from "hono/factory";
import { createAuth } from "./auth.ts";
import type { Bindings } from "./env.ts";
import { zValidator } from "@hono/zod-validator";
import { createNoteSchema, aiPromptSchema } from "@app/contracts";
import { createDb, notes, eq } from "@app/db";

type Auth = ReturnType<typeof createAuth>;
type Session = NonNullable<Awaited<ReturnType<Auth["api"]["getSession"]>>>;

type Env = {
  Bindings: Bindings;
  Variables: {
    session: Session;
  };
};

const requireSession = createMiddleware<Env>(async (c, next) => {
  const session = await createAuth(c.env).api.getSession({
    headers: c.req.raw.headers,
  });

  if (!session) {
    return c.json({ error: "Unauthorized" }, 401);
  }

  c.set("session", session);
  await next();
});

const app = new Hono<Env>();

app.all("/api/auth/*", (c) => {
  return createAuth(c.env).handler(c.req.raw);
});

app.get("/api/health", (c) => {
  return c.json({ ok: true });
});

app.get("/api/me", requireSession, (c) => {
  return c.json({ user: c.get("session").user });
});

app.get("/api/notes", requireSession, async (c) => {
  const rows = await createDb(c.env.DB)
    .select()
    .from(notes)
    .where(eq(notes.userId, c.get("session").user.id));

  return c.json({ notes: rows });
});

app.post("/api/notes", requireSession, zValidator("json", createNoteSchema), async (c) => {
  const note = {
    id: crypto.randomUUID(),
    userId: c.get("session").user.id,
    title: c.req.valid("json").title,
    createdAt: new Date(),
  };

  await createDb(c.env.DB).insert(notes).values(note);

  return c.json({ note }, 201);
});

app.post("/api/ai", requireSession, zValidator("json", aiPromptSchema), async (c) => {
  if (!c.env.AI) {
    return c.json({ error: "Workers AI is not configured" }, 503);
  }

  const result = await c.env.AI.run("@cf/google/gemma-4-26b-a4b-it", {
    messages: [
      {
        role: "user",
        content: c.req.valid("json").prompt,
      },
    ],
    max_tokens: 256,
    chat_template_kwargs: { enable_thinking: false },
  });

  return c.json(result);
});

export default app;
