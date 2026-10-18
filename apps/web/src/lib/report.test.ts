import { expect, test } from "vite-plus/test";
import type { ReviewJob } from "@app/contracts";
import { reportHtml } from "./report";
import evidence from "../../../../tests/fixtures/evidence.json";

test("exports decisions, comments, citations and evidence while escaping report content", () => {
  const job: ReviewJob = {
    job_id: "a97aa7d9-d1d9-467b-980b-7a8b9fd4b251",
    status: "done",
    verdict: "violation",
    expires_at: "2026-10-19T00:00:00Z",
    reviews: [
      {
        event_index: 0,
        status: "confirmed",
        comment: '<script>alert("x")</script>',
        updated_at: "2026-10-18T00:00:00Z",
      },
    ],
    violations: [
      {
        type: "V18. Дверь открыта (НБРК № 14, п. 8)",
        timecode_sec: 2,
        description:
          "С 2 с дверь открыта. Источник: Документ: «Цитата <не HTML>.» Связь с нормой косвенная; результат является подсказкой для проверки человеком.",
        evidence_frames: [{ timestamp_sec: 2.5, image_base64: evidence.jpeg }],
      },
    ],
  };
  const html = reportHtml(job);
  expect(html).toContain("Проверено 1 из 1");
  expect(html).toContain("Решение: Подтверждено");
  expect(html).toContain("&lt;script&gt;");
  expect(html).not.toContain("<script>");
  expect(html).toContain("&lt;не HTML&gt;");
  expect(html).toContain("Связь с нормой косвенная");
  expect(html).toContain(`data:image/jpeg;base64,${evidence.jpeg}`);
  expect(html).toContain("https://adilet.zan.kz/rus/docs/V2000020075");
  expect(reportHtml({ ...job, reviews: [] })).toContain("Предварительный отчёт");
});
