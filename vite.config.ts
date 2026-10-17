import { defineConfig } from "vite-plus";
import { fileURLToPath, URL } from "node:url";

export default defineConfig({
  resolve: {
    alias: { "@": fileURLToPath(new URL("./apps/web/src", import.meta.url)) },
  },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts", "apps/*/src/**/*.test.{ts,tsx}"],
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
    jsPlugins: [{ name: "vite-plus", specifier: "vite-plus/oxlint-plugin" }, "@shadcn/lint"],
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
        plugins: ["react", "typescript", "oxc"],
        rules: {
          "react/rules-of-hooks": "error",
          "react/only-export-components": ["warn", { allowConstantExport: true }],
          "shadcn/no-restyle": ["error", { allow: ["layout"] }],
          "shadcn/no-raw-colors": "error",
          "shadcn/no-arbitrary-values": "error",
          "shadcn/no-inline-styles": "error",
          "shadcn/no-unknown-classes": "error",
          "shadcn/require-static-classes": "error",
        },
      },
      {
        files: ["apps/web/src/components/ui/**/*.{ts,tsx}"],
        rules: {
          "shadcn/no-restyle": "off",
          "shadcn/no-arbitrary-values": "off",
          "shadcn/require-static-classes": "off",
        },
      },
    ],
  },
});
