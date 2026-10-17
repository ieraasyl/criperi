import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { aiPromptSchema } from "@app/contracts";
import { requireSession, type AuthenticatedEnv } from "../middleware/session.ts";

export const aiRoutes = new Hono<AuthenticatedEnv>();

aiRoutes.post("/", requireSession, zValidator("json", aiPromptSchema), async (c) => {
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
