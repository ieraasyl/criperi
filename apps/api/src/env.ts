import type { D1Database } from "@cloudflare/workers-types";

export type Bindings = {
  DB: D1Database;
  INFERENCE_URL?: string;
  INFERENCE_API_KEY?: string;
  EVALUATION_API_KEY?: string;
  EVALUATION_OWNER_ID?: string;
  UPLOAD_LIMIT_BYTES?: string;
  BETTER_AUTH_URL: string;
  BETTER_AUTH_SECRET: string;
};
