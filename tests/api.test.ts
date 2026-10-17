import { randomUUID } from "node:crypto";
import { expect, inject, test } from "vite-plus/test";

declare module "vite-plus/test" {
  interface ProvidedContext {
    origin: string;
  }
}

const origin = inject("origin");
let client = 0;

function nextIp() {
  return `192.0.2.${++client}`;
}

type RequestOptions = {
  cookie?: string;
  body?: unknown;
  method?: string;
  ip?: string;
};

async function call(
  path: string,
  { cookie, body, method = "GET", ip = nextIp() }: RequestOptions = {},
) {
  return fetch(`${origin}${path}`, {
    method,
    headers: {
      Origin: origin,
      "CF-Connecting-IP": ip,
      ...(cookie ? { Cookie: cookie } : {}),
      ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
    },
    ...(body !== undefined && method !== "GET" ? { body: JSON.stringify(body) } : {}),
  });
}

function cookiesFrom(response: Response) {
  return response.headers
    .getSetCookie()
    .map((value) => value.split(";")[0])
    .join("; ");
}

async function register() {
  const ip = nextIp();
  const credentials = {
    name: "Integration test",
    email: `test-${randomUUID()}@example.com`,
    password: `Test-${randomUUID()}`,
  };

  const response = await call("/api/auth/sign-up/email", {
    method: "POST",
    body: credentials,
    ip,
  });

  expect(response.status, await response.clone().text()).toBe(200);

  const cookie = cookiesFrom(response);
  expect(cookie).not.toBe("");

  return { cookie, credentials, ip };
}

test("health endpoint responds", async () => {
  const response = await call("/api/health");

  expect(response.status).toBe(200);
  expect(await response.json()).toEqual({ ok: true });
});

test("limits repeated sign-in attempts per client", async () => {
  const ip = nextIp();
  const body = { email: "missing@example.com", password: "invalid-password" };

  for (let attempt = 0; attempt < 3; attempt++) {
    expect((await call("/api/auth/sign-in/email", { method: "POST", body, ip })).status).toBe(401);
  }

  const limited = await call("/api/auth/sign-in/email", { method: "POST", body, ip });
  expect(limited.status).toBe(429);
  expect(Number(limited.headers.get("X-Retry-After"))).toBeGreaterThan(0);
  expect((await call("/api/auth/sign-in/email", { method: "POST", body })).status).toBe(401);
});

test.each([
  { path: "/api/me", method: "GET" },
  { path: "/api/notes", method: "GET" },
  { path: "/api/notes", method: "POST" },
  { path: "/api/ai", method: "POST" },
])("$method $path rejects unauthenticated requests", async ({ path, method }) => {
  const response = await call(path, {
    method,
    ...(method === "POST" ? { body: {} } : {}),
  });

  expect(response.status).toBe(401);
  expect(await response.json()).toEqual({ error: "Unauthorized" });
});

test("rejects blank note titles", async () => {
  const account = await register();

  const response = await call("/api/notes", {
    ...account,
    method: "POST",
    body: { title: "   " },
  });

  expect(response.status).toBe(400);
});

test("rejects blank AI prompts before checking the AI binding", async () => {
  const account = await register();

  const response = await call("/api/ai", {
    ...account,
    method: "POST",
    body: { prompt: "   " },
  });

  expect(response.status).toBe(400);
});

test("returns 503 when Workers AI is not configured", async () => {
  const account = await register();

  const response = await call("/api/ai", {
    ...account,
    method: "POST",
    body: { prompt: "Hello" },
  });

  expect(response.status).toBe(503);
  expect(await response.json()).toEqual({ error: "Workers AI is not configured" });
});

test("assigns ownership from the session and retrieves the saved note", async () => {
  const account = await register();
  const me = await call("/api/me", account);

  expect(me.status).toBe(200);
  const { user } = await me.json();

  const created = await call("/api/notes", {
    ...account,
    method: "POST",
    body: { title: " First note ", userId: "another-user" },
  });

  expect(created.status, await created.clone().text()).toBe(201);
  const { note } = await created.json();

  expect(note.userId).toBe(user.id);
  expect(note.title).toBe("First note");

  const response = await call("/api/notes", account);

  expect(response.status).toBe(200);
  expect((await response.json()).notes).toEqual([note]);
});

test("keeps one user's notes hidden from another user", async () => {
  const alice = await register();
  const bob = await register();

  const created = await call("/api/notes", {
    ...alice,
    method: "POST",
    body: { title: "Alice's note" },
  });

  expect(created.status, await created.clone().text()).toBe(201);

  const response = await call("/api/notes", bob);

  expect(response.status).toBe(200);
  expect((await response.json()).notes).toEqual([]);
});

test("invalidates signed-out sessions and allows signing back in", async () => {
  const account = await register();

  expect((await call("/api/me", account)).status).toBe(200);

  const signedOut = await call("/api/auth/sign-out", {
    ...account,
    method: "POST",
    body: {},
  });

  expect(signedOut.status).toBe(200);
  expect((await call("/api/me", account)).status).toBe(401);

  const signedIn = await call("/api/auth/sign-in/email", {
    method: "POST",
    body: account.credentials,
  });

  expect(signedIn.status, await signedIn.clone().text()).toBe(200);

  const cookie = cookiesFrom(signedIn);
  expect(cookie).not.toBe("");
  expect((await call("/api/me", { cookie })).status).toBe(200);
});

test("returns 404 for unknown API routes", async () => {
  const response = await call("/api/missing", {
    // Navigation headers are tested separately below.
  });

  expect(response.status).toBe(404);

  const navigation = await fetch(`${origin}/api/missing`, {
    headers: { "Sec-Fetch-Mode": "navigate" },
  });

  expect(navigation.status).toBe(404);
});

test("serves the React app for frontend navigation", async () => {
  const response = await fetch(`${origin}/dashboard`, {
    headers: { "Sec-Fetch-Mode": "navigate" },
  });

  expect(response.status).toBe(200);
  expect(response.headers.get("content-type")).toMatch(/text\/html/);
});
