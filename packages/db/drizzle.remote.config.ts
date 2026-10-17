import "dotenv/config";
import { defineConfig } from "drizzle-kit";

function required(name: string) {
  const value = process.env[name];
  if (!value) throw new Error(`Missing ${name}`);
  return value;
}

export default defineConfig({
  dialect: "sqlite",
  driver: "d1-http",
  schema: "./src/schema.ts",
  dbCredentials: {
    accountId: required("CLOUDFLARE_ACCOUNT_ID"),
    databaseId: required("CLOUDFLARE_DATABASE_ID"),
    token: required("CLOUDFLARE_D1_TOKEN"),
  },
});
