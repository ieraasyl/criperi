import { Hono } from "hono";
import { bodyLimit } from "hono/body-limit";
import { streamSSE } from "hono/streaming";
import { zValidator } from "@hono/zod-validator";
import {
  LIVE_REQUEST_LIMIT_BYTES,
  liveWindowSchema,
  readLiveEvents,
  type LiveEvent,
} from "@app/contracts";
import { validationError } from "../errors.ts";
import { requireSession, type AuthenticatedEnv } from "../middleware/session.ts";
import { openLiveInference, validateLiveEvent } from "../live-inference.ts";

const failed: LiveEvent = {
  type: "error",
  error: {
    code: "LIVE_FAILED",
    message:
      "Анализ прерван. Найденные события остаются на странице. Остановите воспроизведение и повторите анализ.",
  },
};

export const liveRoutes = new Hono<AuthenticatedEnv>().use("*", requireSession).post(
  "/",
  bodyLimit({
    maxSize: LIVE_REQUEST_LIMIT_BYTES,
    onError: (c) =>
      c.json(
        { error: { code: "UPLOAD_TOO_LARGE", message: "Кадры превышают допустимый размер." } },
        413,
      ),
  }),
  zValidator("json", liveWindowSchema, validationError),
  async (c) => {
    const controller = new AbortController();
    const window = c.req.valid("json");
    const body = await openLiveInference(
      c.env,
      window,
      AbortSignal.any([c.req.raw.signal, controller.signal]),
    );
    c.header("X-Accel-Buffering", "no");
    const response = streamSSE(c, async (stream) => {
      stream.onAbort(() => controller.abort());
      let previousTime = -1;
      try {
        for await (const event of readLiveEvents(body)) {
          validateLiveEvent(event, window, previousTime);
          if (event.type === "frame") previousTime = event.timestamp_sec;
          // The model service's exceptions and internal paths never reach the browser.
          await stream.writeSSE({ data: JSON.stringify(event.type === "error" ? failed : event) });
        }
      } catch {
        if (!controller.signal.aborted) await stream.writeSSE({ data: JSON.stringify(failed) });
      } finally {
        controller.abort();
      }
    });
    response.headers.set("Cache-Control", "private, no-store");
    return response;
  },
);
