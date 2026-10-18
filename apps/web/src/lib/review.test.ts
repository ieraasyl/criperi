import { expect, test } from "vite-plus/test";
import {
  closestFrame,
  eventGroups,
  observationDisplay,
  expiryDisplay,
  isReviewed,
  counted,
  type Finding,
} from "./review";

const event = {
  type: "V18. Дверь открыта",
  timecode_sec: 2,
  description: "С 2 с дверь открыта.",
  evidence_frames: [0, 2.5, 5].map((timestamp_sec) => ({ timestamp_sec, image_base64: "" })),
} satisfies Finding;

test("selects the closest event frame and keeps the earlier frame on a tie", () => {
  expect(closestFrame(event)).toBe(1);
  expect(closestFrame({ ...event, timecode_sec: 1.25 })).toBe(0);
  expect(closestFrame({ ...event, timecode_sec: 99 })).toBe(2);
});
test("groups duplicate categories without losing episode order", () => {
  expect(
    eventGroups([
      event,
      { ...event, type: "V17. Еда" },
      { ...event, type: "V17. Еда", timecode_sec: 20 },
    ]),
  ).toEqual([
    { title: "Дверь открыта", code: "V18", indices: [0] },
    { title: "Еда", code: "V17", indices: [1, 2] },
  ]);
});
test("formats observation timestamps without changing uncertainty or unrelated text", () => {
  expect(observationDisplay("С 2 с дверь остаётся открытой. Около 65 с виден предмет.")).toBe(
    "С 00:02 дверь остаётся открытой. Около 01:05 виден предмет.",
  );
  expect(observationDisplay("С 2,5 с виден предмет.")).toBe("С 00:02.5 виден предмет.");
  expect(observationDisplay("Нет временной отметки; нужна проверка.")).toBe(
    "Нет временной отметки; нужна проверка.",
  );
});
test("warnings use remaining lifetime and unresolved decisions do not count as reviewed", () => {
  expect(expiryDisplay("2026-10-18T12:30:00Z", Date.parse("2026-10-18T12:00:00Z"))).toEqual({
    urgent: true,
    remaining: "осталось 30 мин",
  });
  expect(expiryDisplay("2026-10-19T05:00:00Z", Date.parse("2026-10-18T12:00:00Z"))).toEqual({
    urgent: false,
    remaining: "осталось 17 ч",
  });
  expect(isReviewed(undefined)).toBe(false);
  expect(isReviewed({ event_index: 0, status: "needs_review", comment: "", updated_at: "" })).toBe(
    false,
  );
  expect(isReviewed({ event_index: 0, status: "rejected", comment: "", updated_at: "" })).toBe(
    true,
  );
  expect(counted(12, ["эпизод", "эпизода", "эпизодов"])).toBe("12 эпизодов");
});
