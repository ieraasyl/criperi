export function loginRedirect(value: unknown): string {
  if (typeof value !== "string" || !value.startsWith("/") || value.startsWith("//")) return "/";
  if (
    value.includes("\\") ||
    Array.from(value).some((char) => char.charCodeAt(0) <= 32 || char.charCodeAt(0) === 127)
  )
    return "/";

  try {
    const origin = "https://app.invalid";
    const url = new URL(value, origin);
    const path = decodeURIComponent(url.pathname).replace(/\/+$/, "");
    if (url.origin !== origin || path.toLowerCase() === "/login" || path.includes("\\")) return "/";
    return `${url.pathname}${url.search}${url.hash}`;
  } catch {
    return "/";
  }
}
