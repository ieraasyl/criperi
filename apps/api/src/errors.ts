import { AnalysisError } from "./inference.ts";
import type { Context, ErrorHandler } from "hono";
import { HTTPException } from "hono/http-exception";
import type { ApiError } from "@app/contracts";

export const handleError: ErrorHandler = (error, c) => {
  if (error instanceof AnalysisError)
    return c.json(
      {
        error: {
          code: error.code,
          message: error.message,
          ...(error.jobId ? { job_id: error.jobId } : {}),
        },
      },
      error.status,
    );
  const status = error instanceof HTTPException ? error.status : 500;

  if (status >= 500) console.error(error);

  return c.json(
    {
      error: {
        code: status >= 500 ? "INTERNAL_ERROR" : status === 400 ? "BAD_REQUEST" : "HTTP_ERROR",
        message: status >= 500 ? "Something went wrong. Please try again." : error.message,
      },
    } satisfies ApiError,
    status,
  );
};

export function validationError(
  result: {
    success: boolean;
    error?: { issues: readonly { path: readonly PropertyKey[]; message: string }[] };
  },
  c: Context,
) {
  if (!result.success) {
    return c.json(
      {
        error: {
          code: "VALIDATION_ERROR",
          message: "Check the submitted fields.",
          issues: result.error?.issues.map((issue) => ({
            path: issue.path.map(String),
            message: issue.message,
          })),
        },
      } satisfies ApiError,
      400,
    );
  }
}
