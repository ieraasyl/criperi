import { betterAuth } from "better-auth/minimal";
import { drizzleAdapter } from "@better-auth/drizzle-adapter";
import { authSchema, createDb } from "@app/db";
import { authOptions } from "./auth-options.ts";
import type { Bindings } from "./env.ts";

export function createAuth(env: Bindings) {
  return betterAuth({
    ...authOptions,
    baseURL: env.BETTER_AUTH_URL,
    secret: env.BETTER_AUTH_SECRET,
    database: drizzleAdapter(createDb(env.DB), {
      provider: "sqlite",
      // The adapter validates all model definitions; migrations include only schema.ts exports.
      schema: authSchema,
    }),
  });
}
