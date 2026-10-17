import { hc } from "hono/client";
import type { AppType } from "@app/api/types";

export const api = hc<AppType>("/").api;
