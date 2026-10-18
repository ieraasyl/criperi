import { authClient } from "./auth-client";

let pending: Promise<void> | undefined;

async function prepareSession() {
  const current = await authClient.getSession();
  if (current.error) throw new Error("Не удалось подготовить сеанс.");
  if (current.data) return;

  const result = await authClient.signIn.anonymous();
  // Check the cookie even if the acknowledgement failed or another tab created the session.
  const restored = await authClient.getSession();
  if (restored.data && !restored.error) return;
  if (result.error)
    throw new Error(
      result.error.status === 429
        ? "Слишком много попыток. Подождите немного и повторите."
        : "Не удалось подготовить сеанс. Проверьте соединение и повторите попытку.",
    );
  if (restored.error) throw new Error("Не удалось проверить сеанс. Повторите попытку.");
  throw new Error("Разрешите cookies для этого сайта и повторите попытку.");
}

export function ensureVisitorSession() {
  if (!pending) {
    pending = navigator.locks.request("criperi-visitor-session", prepareSession).finally(() => {
      pending = undefined;
    });
  }
  return pending;
}
