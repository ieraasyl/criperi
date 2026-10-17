# Agent instructions

See [README.md](README.md) for setup and commands.

- Use `vp` for tooling. Check package scripts; run them with `vp run <name>`. Run `vp install` before work and after dependency changes.
- Import tooling from `vite-plus`, tests from `vite-plus/test`. For tooling failures, consult `node_modules/vite-plus/docs` and run `vp env doctor`.
- Keep UI in `apps/web`, shared validation in `packages/contracts`, and database/auth implementation out of browser imports. Use scoped `@cloudflare/workers-types` imports; keep Worker ambient globals confined to the API project.
- Chain Hono routes for RPC inference. Use `apps/web/src/lib/api.ts`; import API types with `import type` from `@app/api/types`.
- Isolate private Query caches by user, include user IDs in keys, and forward cancellation signals. Retain saved data when refresh fails; never retry a confirmed write because its response is malformed.
- Add routes in `apps/web/src/routes`, with protected pages under `_authenticated`. Preserve safe login redirects. Regenerate and commit `routeTree.gen.ts` without hand-editing it.
- Derive ownership from the authenticated session and filter private queries by owner.
- After binding changes, regenerate Cloudflare types and update `apps/api/src/env.ts`. After auth feature changes, regenerate auth schema before SQL migrations. Commit migration SQL and metadata; never hand-edit generated files.
- Keep generation, migrations, and deployment uncached. Deploy through the root script.
- Validate code changes with `vp run ready` and `vp run test`; tests manage their own server/database.
- Edit documentation only when explicitly requested.

## Git conventions

- Branches: `feat/<short-kebab-case-description>` unless the user specifies otherwise.
- Conventional Commits with a short imperative subject, e.g. `fix(auth): reject expired sessions`. One line only; no body or trailers.
- Keep commits focused. Stage related files only and review the staged diff.
