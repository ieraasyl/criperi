export const authOptions = {
  appName: "criperi",
  basePath: "/api/auth",
  emailAndPassword: { enabled: true },
  rateLimit: { enabled: true, storage: "database" as const },
  advanced: {
    ipAddress: { ipAddressHeaders: ["cf-connecting-ip"] },
  },
};
