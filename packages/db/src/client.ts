import { drizzle } from "drizzle-orm/d1";
import type { D1Database } from "@cloudflare/workers-types";
import * as schema from "./schema.ts";

export function createDb(binding: D1Database) {
  return drizzle(binding, { schema });
}

export { notes } from "./schema.ts";
export { and, eq } from "drizzle-orm";
