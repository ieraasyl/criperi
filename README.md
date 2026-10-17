# Criperi

A starter with React, Hono, Better Auth, and Cloudflare D1. Includes typed API calls, routing, data fetching, and Workers AI. The UI and API deploy together as one Cloudflare Worker.

[Live app](https://criperi.ieraasyl.workers.dev)

## Project layout

| Directory            | Contents                            |
| -------------------- | ----------------------------------- |
| `apps/web`           | React UI, TanStack Router and Query |
| `apps/api`           | Hono RPC, Better Auth, Workers AI   |
| `packages/contracts` | Shared Zod validation               |
| `packages/db`        | Drizzle schemas and migrations      |
| `tests`              | HTTP tests and server setup         |

See [AGENTS.md](AGENTS.md) for implementation conventions.

## Local setup

Use Node.js 24 and Vite+ (`vp`).

```sh
vp install
cp apps/api/.dev.vars.example apps/api/.dev.vars
openssl rand -hex 32
```

Set `BETTER_AUTH_SECRET` in `apps/api/.dev.vars` to the generated value. Keep the local `BETTER_AUTH_URL` unchanged.

```sh
vp run cf:types
vp run db:migrate:local
vp run dev
```

Open <http://127.0.0.1:5173>. Development uses local bindings by default through `CLOUDFLARE_VITE_FORCE_LOCAL=true` in `apps/web/.env.development.local`, so no Cloudflare login is needed. Local D1 data lives in `.wrangler/state`.

For real Workers AI, set `CLOUDFLARE_VITE_FORCE_LOCAL=false` in `apps/web/.env.development.local`, run `vp -C apps/api exec wrangler login`, and restart the dev server. AI calls use your account's allowance or billing.

## Checks and tools

| Command                   | Purpose                                       |
| ------------------------- | --------------------------------------------- |
| `vp run ready`            | Check formatting, lint, types, and build      |
| `vp check --fix`          | Apply formatting and fixable lint corrections |
| `vp run test`             | Run all tests                                 |
| `vp test watch`           | Run tests in watch mode                       |
| `vp run db:studio:local`  | Inspect local D1                              |
| `vp run db:studio:remote` | Inspect remote D1                             |

Tests manage their own server and temporary database; no separate dev server or Cloudflare login is needed.

Studio opens at <https://local.drizzle.studio>. Remote Studio requires `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_DATABASE_ID`, and `CLOUDFLARE_D1_TOKEN` in `packages/db/.env`. The token needs D1 Edit permission.

## Schema changes

When auth feature changes affect the database schema, run `vp run auth:generate` first. After changing table definitions, generate and apply migrations:

```sh
vp run db:generate
vp run db:migrate:local
```

Commit generated SQL and metadata in `packages/db/migrations`. For Worker binding changes, update `apps/api/src/env.ts` and run `vp run cf:types`.

## Deployment

Log in with `vp -C apps/api exec wrangler login`.

Set the production `BETTER_AUTH_URL` and D1 database ID in `apps/api/wrangler.jsonc`. Store a separate production secret with `vp -C apps/api exec wrangler secret put BETTER_AUTH_SECRET`.

After checks and tests pass:

```sh
vp run deploy:dry
vp run db:migrate:remote
vp run deploy
```
