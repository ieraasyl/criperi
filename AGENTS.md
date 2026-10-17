# Agent instructions

Rules for coding agents working in this repository. See [README.md](README.md) for setup and operating commands.

- Use Vite+ (`vp`). `vp <command>` runs a built-in; `vp run <name>` runs a project script. Check package scripts before choosing a command.
- Run `vp install` before starting work and after dependency changes. For tooling issues, consult `node_modules/vite-plus/docs` and run `vp env doctor`.
- Import tooling APIs from `vite-plus` and test APIs from `vite-plus/test`.
- Keep database and server auth code out of browser imports. Put shared validation in `packages/contracts`; keep UI in `apps/web`.
- Derive note ownership from the authenticated session and filter queries by that owner.
- Regenerate Cloudflare types after binding changes. Regenerate auth schema after auth feature changes, then generate SQL migrations. Commit migration SQL and metadata; keep generated files generated.
- Keep migrations, generation, and deployment tasks uncached. Deploy through the root script so the Worker and frontend ship together.
- Validate code changes with `vp run ready` and `vp run test`. Vitest manages its own server and temporary D1 database.
- Create or update documentation ONLY when EXPLICITLY requested by the user.

## Git conventions

- Name new branches `feat/<short-kebab-case-description>`, unless the user specifies another name.
- Use Conventional Commits: `<type>: <short imperative summary>`. Types include `feat`, `fix`, `docs`, `refactor`, `test`, and `chore`; add a scope when useful, such as `fix(auth): reject expired sessions`.
- Commit messages must contain only a single subject line. NEVER include a body or trailers, including `Co-authored-by` or `Signed-off-by`.
- Keep each commit focused on one change. Stage only related files and review the staged diff before committing.
