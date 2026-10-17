import { execFile } from "node:child_process";
import { randomBytes, randomUUID } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { createServer as createNetServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { createServer, type ViteDevServer } from "vite-plus";
import type { TestProject } from "vite-plus/test/node";
import { hc } from "hono/client";
import type { AppType } from "@app/api/types";
import { startMockInference } from "./mock-inference.ts";

const run = promisify(execFile);
const root = fileURLToPath(new URL("../", import.meta.url));

async function availablePort() {
  const listener = createNetServer();
  await new Promise<void>((resolve, reject) => {
    listener.once("error", reject);
    listener.listen(0, "127.0.0.1", resolve);
  });
  const address = listener.address();
  await new Promise<void>((resolve, reject) => {
    listener.close((error) => (error ? reject(error) : resolve()));
  });
  if (!address || typeof address === "string") throw new Error("Could not allocate a test port");
  return address.port;
}

export default async function setup({ provide }: TestProject) {
  const directory = await mkdtemp(join(tmpdir(), "criperi-test-"));
  const configPath = join(directory, "wrangler.json");
  const statePath = join(directory, "state");
  const previousConfig = process.env.CLOUDFLARE_CONFIG_PATH;
  const previousState = process.env.CLOUDFLARE_STATE_PATH;
  let server: ViteDevServer | undefined;
  const inference = await startMockInference();
  const evaluationKey = randomBytes(32).toString("hex");

  async function teardown() {
    try {
      await server?.close();
      await inference.close();
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  }

  try {
    const port = await availablePort();
    const origin = `http://127.0.0.1:${port}`;
    await writeFile(
      configPath,
      JSON.stringify({
        name: "criperi-test",
        main: join(root, "apps/api/src/index.ts"),
        compatibility_date: "2026-10-16",
        compatibility_flags: ["nodejs_compat"],
        assets: {
          not_found_handling: "single-page-application",
          run_worker_first: ["/api", "/api/*"],
        },
        d1_databases: [
          {
            binding: "DB",
            database_name: "criperi-test-db",
            database_id: randomUUID(),
            migrations_dir: join(root, "packages/db/migrations"),
          },
        ],
        vars: {
          BETTER_AUTH_URL: origin,
          BETTER_AUTH_SECRET: randomBytes(32).toString("base64"),
          INFERENCE_URL: inference.origin,
          INFERENCE_API_KEY: inference.key,
          EVALUATION_API_KEY: evaluationKey,
          UPLOAD_LIMIT_BYTES: "4096",
        },
      }),
    );

    await run(
      "vp",
      [
        "-C",
        "apps/api",
        "exec",
        "wrangler",
        "d1",
        "migrations",
        "apply",
        "DB",
        "--local",
        "--config",
        configPath,
        "--persist-to",
        statePath,
      ],
      { cwd: root, timeout: 30_000 },
    );

    process.env.CLOUDFLARE_CONFIG_PATH = configPath;
    process.env.CLOUDFLARE_STATE_PATH = statePath;
    server = await createServer({
      root: join(root, "apps/web"),
      configFile: join(root, "apps/web/vite.config.ts"),
      mode: "test",
      server: { host: "127.0.0.1", port, strictPort: true, open: false },
    });
    await server.listen();

    const deadline = Date.now() + 30_000;
    const client = hc<AppType>(origin).api;
    while (true) {
      try {
        const response = await client.health.$get(
          {},
          { init: { signal: AbortSignal.timeout(1000) } },
        );
        if (response.ok && (await response.json()).ok === true) break;
      } catch {
        // The local Worker may still be starting.
      }
      if (Date.now() >= deadline)
        throw new Error("Test server did not become ready within 30 seconds");
      await setTimeout(100);
    }

    provide("origin", origin);
    provide("evaluationKey", evaluationKey);
    provide("inferenceOrigin", inference.origin);
    provide("inferenceKey", inference.key);
    return teardown;
  } catch (error) {
    await teardown();
    throw error;
  } finally {
    if (previousConfig === undefined) delete process.env.CLOUDFLARE_CONFIG_PATH;
    else process.env.CLOUDFLARE_CONFIG_PATH = previousConfig;
    if (previousState === undefined) delete process.env.CLOUDFLARE_STATE_PATH;
    else process.env.CLOUDFLARE_STATE_PATH = previousState;
  }
}
