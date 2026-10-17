import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { createNoteSchema } from "@app/contracts";
import { createDb, notes, eq } from "@app/db";
import { requireSession, type AuthenticatedEnv } from "../middleware/session.ts";

export const notesRoutes = new Hono<AuthenticatedEnv>();

notesRoutes.get("/", requireSession, async (c) => {
  const rows = await createDb(c.env.DB)
    .select()
    .from(notes)
    .where(eq(notes.userId, c.get("session").user.id));

  return c.json({ notes: rows });
});

notesRoutes.post("/", requireSession, zValidator("json", createNoteSchema), async (c) => {
  const note = {
    id: crypto.randomUUID(),
    userId: c.get("session").user.id,
    title: c.req.valid("json").title,
    createdAt: new Date(),
  };

  await createDb(c.env.DB).insert(notes).values(note);

  return c.json({ note }, 201);
});
