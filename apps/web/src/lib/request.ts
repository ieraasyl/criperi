import { apiErrorSchema, type ApiError } from "@app/contracts";
import type { ClientResponse, InferResponseType } from "hono/client";
import type { SuccessStatusCode } from "hono/utils/http-status";

type SuccessBody<T extends ClientResponse<unknown>> = InferResponseType<
  () => Promise<T>,
  SuccessStatusCode
>;

export class ApiRequestError extends Error {
  readonly status: number;
  readonly code: string;
  readonly issues?: ApiError["error"]["issues"];

  constructor(message: string, status: number, code: string, issues?: ApiError["error"]["issues"]) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
    this.code = code;
    this.issues = issues;
  }
}

export async function request<T extends ClientResponse<unknown>>(
  send: (init: RequestInit) => Promise<T>,
  options: RequestInit = {},
): Promise<SuccessBody<T>> {
  let response: T;
  try {
    response = await send(options);
  } catch (cause) {
    if (options?.signal?.aborted) throw cause;
    throw new ApiRequestError(
      "Не удалось подключиться. Проверьте соединение и повторите попытку.",
      0,
      "NETWORK_ERROR",
    );
  }

  let body: unknown;
  try {
    body = await response.json();
  } catch (cause) {
    if (options.signal?.aborted) throw cause;
    if (response.ok) {
      throw new ApiRequestError(
        "Сервер вернул некорректный ответ.",
        response.status,
        "INVALID_RESPONSE",
      );
    }
  }

  if (!response.ok) {
    const parsed = apiErrorSchema.safeParse(body);
    const error = parsed.success ? parsed.data.error : undefined;
    throw new ApiRequestError(
      response.status === 401
        ? "Сеанс завершён. Войдите снова."
        : (error?.message ??
            (response.status >= 500
              ? "Произошла ошибка. Повторите попытку."
              : "Запрос не выполнен. Повторите попытку.")),
      response.status,
      error?.code ?? (response.status === 401 ? "UNAUTHORIZED" : "HTTP_ERROR"),
      error?.issues,
    );
  }

  return body as SuccessBody<T>;
}
