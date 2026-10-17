import { fileURLToPath, URL } from "node:url";
import { cloudflare } from "@cloudflare/vite-plugin";
import react, { reactCompilerPreset } from "@vitejs/plugin-react";
import babel from "@rolldown/plugin-babel";
import { defineConfig, lazyPlugins } from "vite-plus";
import tailwindcss from "@tailwindcss/vite";

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: lazyPlugins(() => [
    react(),
    babel({ presets: [reactCompilerPreset()] }),
    tailwindcss(),
    cloudflare({
      inspectorPort: mode === "test" ? false : undefined,
      configPath: process.env.CLOUDFLARE_CONFIG_PATH ?? "../api/wrangler.jsonc",
      persistState: {
        path:
          process.env.CLOUDFLARE_STATE_PATH ??
          fileURLToPath(new URL("../../.wrangler/state", import.meta.url)),
      },
    }),
  ]),
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
}));
