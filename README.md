# Criperi

A notes app with React, Hono, Better Auth, Cloudflare D1, Drizzle, and Workers AI. React and the API deploy together as one Cloudflare Worker.

[Live app](https://criperi.ieraasyl.workers.dev)

## Project layout

| Directory            | Contents                                                      |
| -------------------- | ------------------------------------------------------------- |
| `apps/web`           | React, Tailwind, and shadcn/ui using Base UI                  |
| `apps/api`           | Hono routes, authentication, and Wrangler configuration       |
| `packages/contracts` | Shared Zod schemas and types                                  |
| `packages/db`        | Drizzle client, schemas, migrations, and Studio configuration |
| `tests`              | Vitest HTTP integration tests and isolated server setup       |

## Local setup

Use Node.js 24 and Vite+ (`vp`). The package manager version is declared in `package.json`.

```sh
vp install
cp apps/api/.dev.vars.example apps/api/.dev.vars
openssl rand -hex 32
```

Set `BETTER_AUTH_SECRET` in `apps/api/.dev.vars` to the generated value. Keep `BETTER_AUTH_URL=http://127.0.0.1:5173` for local development.

```sh
vp run cf:types
vp run db:migrate:local
vp run dev
```

Open <http://127.0.0.1:5173>. Local D1 data lives in `.wrangler/state`. Development defaults to local bindings through `CLOUDFLARE_VITE_FORCE_LOCAL=true` in `apps/web/.env.development.local`, so the app works without Cloudflare login.

For real Workers AI locally, set that flag to `false`, run `vp -C apps/api exec wrangler login` with your own Cloudflare login, and restart the dev server. Remote AI calls use the selected account's AI allowance or billing.

## Checks and tools

| Command                   | Purpose                                       |
| ------------------------- | --------------------------------------------- |
| `vp run ready`            | Check formatting, lint, types, and build      |
| `vp check --fix`          | Apply formatting and fixable lint corrections |
| `vp run test`             | Run Vitest integration tests once             |
| `vp test watch`           | Run tests in watch mode                       |
| `vp run db:studio:local`  | Inspect local D1                              |
| `vp run db:studio:remote` | Inspect remote D1                             |

Tests start and clean up their own server and temporary database. A separate dev server and Cloudflare login are not required for tests.

Studio opens at <https://local.drizzle.studio>. Remote Studio reads `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_DATABASE_ID`, and `CLOUDFLARE_D1_TOKEN` from `packages/db/.env`; the token needs D1 Edit permission.

## Schema changes

For auth feature changes, run `vp run auth:generate` first. After changing table definitions:

```sh
vp run db:generate
vp run db:migrate:local
```

Review and commit the generated SQL and metadata in `packages/db/migrations`.

## Deployment

Run `vp -C apps/api exec wrangler login` with a Cloudflare user that has access to the target account.

Set the production `BETTER_AUTH_URL` and D1 database ID in `apps/api/wrangler.jsonc`. Store a separate production secret with `vp -C apps/api exec wrangler secret put BETTER_AUTH_SECRET`.

After checks and tests pass:

```sh
vp run deploy:dry
vp run db:migrate:remote
vp run deploy
```
