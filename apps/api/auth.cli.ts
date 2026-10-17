import { betterAuth } from "better-auth";
import { authOptions } from "./src/auth-options.ts";

export const auth = betterAuth({
  ...authOptions,
  baseURL: "http://127.0.0.1:5173",
  secret: "schema-generation-only-never-a-runtime-secret",
});
