import { createMiddleware } from "hono/factory";
import { createAuth } from "../auth.ts";
import type { Bindings } from "../env.ts";

type Auth = ReturnType<typeof createAuth>;
type Session = NonNullable<Awaited<ReturnType<Auth["api"]["getSession"]>>>;

export type AuthenticatedEnv = {
  Bindings: Bindings;
  Variables: {
    session: Session;
  };
};

export const requireSession = createMiddleware<AuthenticatedEnv>(async (c, next) => {
  const session = await createAuth(c.env).api.getSession({
    headers: c.req.raw.headers,
  });

  if (!session) {
    return c.json({ error: "Unauthorized" }, 401);
  }

  c.set("session", session);
  await next();
});
