import { readdirSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig } from "drizzle-kit";

const directory = fileURLToPath(
  new URL("../../.wrangler/state/v3/d1/miniflare-D1DatabaseObject/", import.meta.url),
);

const selected = process.env.LOCAL_D1_FILE;

const files = existsSync(directory)
  ? readdirSync(directory).filter((name) => name.endsWith(".sqlite") && name !== "metadata.sqlite")
  : [];

if (!selected && files.length !== 1) {
  throw new Error(
    "Expected one local application database. Set LOCAL_D1_FILE to the desired absolute SQLite path.",
  );
}

export default defineConfig({
  dialect: "sqlite",
  schema: "./src/schema.ts",
  dbCredentials: {
    url: selected ?? `${directory}${files[0]}`,
  },
});
