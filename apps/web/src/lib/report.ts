import type { ReviewJob } from "@app/contracts";
import { findingDisplay, descriptionDisplay } from "./finding-display";
import { isReviewed, observationDisplay, reviewLabels, timecode, documentUrl } from "./review";

function escape(value: string) {
  return value.replace(
    /[&<>"']/g,
    (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]!,
  );
}
export function reportHtml(job: ReviewJob, now = new Date()) {
  if (job.status !== "done") throw new Error("Анализ ещё не завершён.");
  const reviews = job.reviews ?? [];
  const reviewed = job.violations.filter((_, index) =>
    isReviewed(reviews.find((review) => review.event_index === index)),
  ).length;
  return `<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Отчёт по проверке видео</title><style>
body{font:15px/1.5 system-ui,sans-serif;color:#111;max-width:1000px;margin:32px auto;padding:0 24px}h1{font-size:26px}h2{font-size:21px}h3{font-size:16px}section{border-top:1px solid #bbb;margin-top:24px;padding-top:12px}p,blockquote{white-space:pre-wrap}blockquote{border-left:2px solid #aaa;margin-left:0;padding-left:16px}.frames{display:flex;flex-wrap:wrap;gap:16px}figure{margin:0}img{max-width:280px;max-height:360px;object-fit:contain}button{padding:8px 14px}@media print{button{display:none}body{margin:0;padding:0}h2,h3{break-after:avoid}figure,blockquote{break-inside:avoid}img{max-height:260px}}
</style></head><body><button onclick="window.print()">Печать / сохранить PDF</button><h1>Отчёт по проверке видео</h1>
<p>Экспортировано: ${escape(now.toLocaleString("ru-RU"))}\nПроверено ${reviewed} из ${job.violations.length}\n${reviewed < job.violations.length ? "Предварительный отчёт: есть события без окончательного решения." : "Все события рассмотрены."}</p>
<p>Результат автоматического анализа предназначен для проверки человеком и не является окончательным заключением.</p>
${job.violations
  .map((event, index) => {
    const finding = findingDisplay(event.type);
    const description = descriptionDisplay(event.description);
    const review = reviews.find((item) => item.event_index === index);
    const url = documentUrl(finding.reference, description.source);
    return `<section><h2>${index + 1}. ${escape(finding.title)}</h2><p>Время на видео: ${timecode(event.timecode_sec)}${finding.code ? `\nКод: ${escape(finding.code)}` : ""}${finding.reference ? `\nНормативная ссылка: ${escape(finding.reference)}` : ""}\nРешение: ${review ? reviewLabels[review.status] : "Не проверено"}</p>
${review?.comment ? `<h3>Комментарий проверяющего</h3><p>${escape(review.comment)}</p>` : ""}
<h3>Наблюдение</h3><p>${escape(observationDisplay(description.observation))}</p>
${description.qualification ? `<h3>Ограничения вывода</h3><p>${escape(description.qualification)}</p>` : ""}
${description.source ? `<h3>Нормативный источник</h3><p>${escape(description.source)}</p>` : ""}
${description.quote ? `<blockquote>«${escape(description.quote)}»</blockquote>` : ""}${url ? `<p><a href="${url}">Источник на Adilet</a></p>` : ""}
<div class="frames">${event.evidence_frames.map((frame) => `<figure><img src="data:image/jpeg;base64,${frame.image_base64}" alt="Кадр события"><figcaption>${timecode(frame.timestamp_sec)}</figcaption></figure>`).join("")}</div></section>`;
  })
  .join("")}</body></html>`;
}
export function exportReport(job: ReviewJob) {
  const url = URL.createObjectURL(new Blob([reportHtml(job)], { type: "text/html;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `report-${job.job_id}.html`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
