import { anonymous } from "better-auth/plugins/anonymous";

export const authOptions = {
  appName: "criperi",
  basePath: "/api/auth",
  plugins: [anonymous()],
  rateLimit: { enabled: true, storage: "database" as const },
  advanced: {
    ipAddress: { ipAddressHeaders: ["cf-connecting-ip"] },
  },
};
