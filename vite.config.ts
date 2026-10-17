import { defineConfig } from "vite-plus";

export default defineConfig({
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    globalSetup: ["./tests/global-setup.ts"],
    testTimeout: 15_000,
  },
  staged: {
    "*": "vp check --fix",
  },
  fmt: {
    ignorePatterns: [
      "**/worker-configuration.d.ts",
      "packages/db/src/auth-schema.ts",
      "packages/db/migrations/meta/**",
    ],
  },
  lint: {
    ignorePatterns: ["**/worker-configuration.d.ts"],
    jsPlugins: [{ name: "vite-plus", specifier: "vite-plus/oxlint-plugin" }],
    rules: {
      "vite-plus/prefer-vite-plus-imports": "error",
    },
    options: {
      typeAware: true,
      typeCheck: true,
    },
    overrides: [
      {
        files: ["apps/web/src/**/*.{ts,tsx}"],
        plugins: ["react"],
        rules: {
          "react/rules-of-hooks": "error",
        },
      },
    ],
  },
});
