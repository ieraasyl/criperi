import type { Job, EventReview } from "@app/contracts";
import { findingDisplay } from "./finding-display";

export type Finding = Extract<Job, { status: "done" }>["violations"][number];
export const reviewLabels = {
  confirmed: "Подтверждено",
  rejected: "Отклонено",
  needs_review: "Нужна проверка",
};
export function counted(count: number, forms: [string, string, string]) {
  const last = count % 10;
  const special = count % 100 >= 11 && count % 100 <= 14;
  return `${count} ${forms[special ? 2 : last === 1 ? 0 : last >= 2 && last <= 4 ? 1 : 2]}`;
}
export function isReviewed(review: EventReview | undefined) {
  return review?.status === "confirmed" || review?.status === "rejected";
}
export function timecode(seconds: number) {
  const tenths = Math.round(seconds * 10);
  return `${String(Math.floor(tenths / 600)).padStart(2, "0")}:${((tenths % 600) / 10).toFixed(1).padStart(4, "0")}`;
}
export function closestFrame(event: Finding) {
  return event.evidence_frames.reduce(
    (best, frame, index, frames) =>
      Math.abs(frame.timestamp_sec - event.timecode_sec) <
      Math.abs(frames[best]!.timestamp_sec - event.timecode_sec)
        ? index
        : best,
    0,
  );
}
export function observationDisplay(text: string) {
  return text.replace(
    /(^|\s)(С|В|Около) (\d+(?:[.,]\d+)?) с(?=\s|[.,;:]|$)/gu,
    (_, before: string, prefix: string, seconds: string) =>
      `${before}${prefix} ${timecode(Number(seconds.replace(",", "."))).replace(/\.0$/, "")}`,
  );
}
export function documentUrl(reference: string | undefined, source: string | undefined) {
  const text = reference ?? source ?? "";
  if (/НБРК[\s\S]*?№\s*231\b/.test(text)) return "https://adilet.zan.kz/rus/docs/V1900019680";
  if (/НБРК[\s\S]*?№\s*14\b/.test(text)) return "https://adilet.zan.kz/rus/docs/V2000020075";
  if (/АРРФР[\s\S]*?№\s*53\b/.test(text)) return "https://adilet.zan.kz/rus/docs/V2600038336";
  return undefined;
}
export function eventGroups(events: Finding[]) {
  const groups = new Map<string, { title: string; code: string | undefined; indices: number[] }>();
  events.forEach((event, index) => {
    const finding = findingDisplay(event.type);
    const key = `${finding.code ?? ""}:${finding.title}`;
    const group = groups.get(key) ?? { title: finding.title, code: finding.code, indices: [] };
    group.indices.push(index);
    groups.set(key, group);
  });
  return [...groups.values()];
}
export function expiryDisplay(expiresAt: string, now: number) {
  const minutes = Math.max(0, Math.ceil((Date.parse(expiresAt) - now) / 60_000));
  return {
    urgent: minutes <= 60,
    remaining: minutes >= 60 ? `осталось ${Math.floor(minutes / 60)} ч` : `осталось ${minutes} мин`,
  };
}
