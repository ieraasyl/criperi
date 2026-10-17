import { createContext } from "react";
import type { authClient } from "./auth-client";

export type Session = typeof authClient.$Infer.Session;
export const SessionContext = createContext<Session | null>(null);
