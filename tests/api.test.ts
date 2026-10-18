import { randomUUID } from "node:crypto";
import { expect, inject, test } from "vite-plus/test";
import { hc } from "hono/client";
import type { AppType } from "@app/api/types";

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

async function visitor() {
  const response = await call("/api/auth/sign-in/anonymous", { method: "POST", body: {} });
  expect(response.status).toBe(200);
  return { response, cookie: cookiesFrom(response) };
}

test("health endpoint responds through the typed client", async () => {
  const response = await hc<AppType>(origin).api.health.$get();
  expect(response.status).toBe(200);
  expect(await response.json()).toEqual({ ok: true });
});

test("anonymous sessions use private cookies, survive reloads and cannot be forged", async () => {
  const { response, cookie } = await visitor();
  const { user } = (await response.json()) as { user: { id: string; isAnonymous: boolean } };
  expect(user.isAnonymous).toBe(true);
  expect(cookie).not.toBe("");
  expect(response.headers.get("cache-control")).toContain("no-store");
  const token = response.headers.getSetCookie().find((value) => value.includes("session_token="));
  expect(token).toMatch(/HttpOnly/i);
  expect(token).toMatch(/SameSite=Lax/i);
  for (let reload = 0; reload < 2; reload++) {
    const restored = await call("/api/auth/get-session", { cookie });
    expect(restored.status).toBe(200);
    expect(await restored.json()).toMatchObject({ user: { id: user.id, isAnonymous: true } });
  }
  const forged = cookie.replace(/session_token=[^;]+/, "session_token=forged");
  expect(await (await call("/api/auth/get-session", { cookie: forged })).json()).toBeNull();
  expect((await call("/api/analyses/future", { cookie: forged })).status).toBe(401);
  const other = await visitor();
  expect(await other.response.json()).not.toMatchObject({ user: { id: user.id } });
});

test("anonymous session creation is rate limited and rejects cross-site requests", async () => {
  const ip = nextIp();
  for (let attempt = 0; attempt < 3; attempt++) {
    expect(
      (await call("/api/auth/sign-in/anonymous", { method: "POST", body: {}, ip })).status,
    ).toBe(200);
  }
  const limited = await call("/api/auth/sign-in/anonymous", { method: "POST", body: {}, ip });
  expect(limited.status).toBe(429);
  expect(Number(limited.headers.get("X-Retry-After"))).toBeGreaterThan(0);
  const crossSite = await fetch(`${origin}/api/auth/sign-in/anonymous`, {
    method: "POST",
    headers: { Origin: "https://other.example", "Content-Type": "application/json" },
    body: "{}",
  });
  expect(crossSite.status).toBe(403);
});

test.each([
  { path: "/api/analyses", method: "POST" },
  { path: "/api/analyses/future", method: "GET" },
  { path: "/api/analyses/future/reviews/0", method: "PUT" },
])("$method $path requires a visitor session", async ({ path, method }) => {
  const response = await call(path, { method, ...(method !== "GET" ? { body: {} } : {}) });
  expect(response.status).toBe(401);
  expect(await response.json()).toEqual({
    error: { code: "UNAUTHORIZED", message: "Сеанс завершён. Обновите страницу." },
  });
});

test("malformed review JSON returns a structured error", async () => {
  const { cookie } = await visitor();
  const response = await fetch(`${origin}/api/analyses/${randomUUID()}/reviews/0`, {
    method: "PUT",
    headers: { Cookie: cookie, "Content-Type": "application/json" },
    body: "{",
  });
  expect(response.status).toBe(400);
  expect(await response.json()).toMatchObject({ error: { code: "BAD_REQUEST" } });
});

test.each([
  { path: "/api/notes", method: "GET" },
  { path: "/api/me", method: "GET" },
  { path: "/api/auth/sign-up/email", method: "POST" },
  { path: "/api/auth/sign-in/email", method: "POST" },
  { path: "/api/auth/list-accounts", method: "GET" },
  { path: "/api/auth/delete-anonymous-user", method: "POST" },
])("unused $method $path is not exposed", async ({ path, method }) => {
  const response = await call(path, { method });
  expect(response.status).toBe(404);
});

test("unknown API routes return JSON even for browser navigation", async () => {
  const response = await call("/api/missing");
  expect(response.status).toBe(404);
  expect(await response.json()).toEqual({
    error: { code: "NOT_FOUND", message: "Endpoint not found." },
  });
  const navigation = await fetch(`${origin}/api/missing`, {
    headers: { "Sec-Fetch-Mode": "navigate" },
  });
  expect(navigation.status).toBe(404);
});

test("frontend navigation serves the React app", async () => {
  const response = await fetch(`${origin}/analyses/${randomUUID()}`, {
    headers: { "Sec-Fetch-Mode": "navigate" },
  });
  expect(response.status).toBe(200);
  expect(response.headers.get("content-type")).toMatch(/text\/html/);
});
