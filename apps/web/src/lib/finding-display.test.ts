import { expect, test } from "vite-plus/test";
import { descriptionDisplay, findingDisplay } from "./finding-display";

test("separates the supplied code and citation without changing the finding", () => {
  expect(findingDisplay("V18. Дверь кабинета оставлена открытой (НБРК № 14, п. 8)")).toEqual({
    title: "Дверь кабинета оставлена открытой",
    code: "V18",
    reference: "НБРК № 14, п. 8",
  });
});

test.each([
  "Личные вещи",
  "Дверь открыта (вид со стороны коридора)",
  "V18",
  "V18. (НБРК № 14, п. 8)",
])("preserves an unrecognized or incomplete title: %s", (type) => {
  expect(findingDisplay(type)).toEqual({ title: type, code: undefined, reference: undefined });
});

test("handles a code and a reference independently", () => {
  expect(findingDisplay("V18. Дверь кабинета оставлена открытой")).toEqual({
    title: "Дверь кабинета оставлена открытой",
    code: "V18",
    reference: undefined,
  });
  expect(findingDisplay("Дверь кабинета оставлена открытой (НБРК № 14, п. 8)")).toEqual({
    title: "Дверь кабинета оставлена открытой",
    code: undefined,
    reference: "НБРК № 14, п. 8",
  });
});

test("recognizes the other regulator used by the report generator", () => {
  expect(findingDisplay("V14. Открытый рабочий экран (АРРФР № 53, п. 71)")).toEqual({
    title: "Открытый рабочий экран",
    code: "V14",
    reference: "АРРФР № 53, п. 71",
  });
});

test("preserves parentheses within a regulatory reference", () => {
  expect(findingDisplay("V01. Деньги на столе (НБРК № 231, п. 20 (часть 2))")).toEqual({
    title: "Деньги на столе",
    code: "V01",
    reference: "НБРК № 231, п. 20 (часть 2)",
  });
});

test("separates observation, source, quote and the supplied qualification", () => {
  const observation = "С 2 с дверь кабинета остаётся открытой после прохода людей.";
  const source = "Название документа из ответа сервиса, п. 8";
  const quote = "Цитата с вложенными «кавычками» и переносом\nстроки.";
  const qualification =
    "Связь с нормой косвенная; результат является подсказкой для проверки человеком.";

  expect(
    descriptionDisplay(`${observation} Источник: ${source}: «${quote}» ${qualification}`),
  ).toEqual({ observation, source, quote, qualification });
});

test("keeps the absence of a normative source visible", () => {
  const observation = "С 3 с в кабинете одновременно находятся не менее трёх человек.";
  const qualification =
    "Открытой нормы в актах adilet.zan.kz не найдено; класс задан правилами кейса.";

  expect(descriptionDisplay(`${observation} ${qualification}`)).toEqual({
    observation,
    source: undefined,
    quote: undefined,
    qualification,
  });
});

test.each([
  "На столе видна личная вещь.\nНужна проверка.",
  "На столе видна личная вещь. Источник:",
  "Оценка предварительная; точная связь с нормой не установлена.",
])("preserves unfamiliar descriptions without dropping text: %s", (observation) => {
  expect(descriptionDisplay(observation)).toEqual({
    observation,
    source: undefined,
    quote: undefined,
    qualification: undefined,
  });
});

test("retains source text and commentary when a quotation is not provided", () => {
  expect(
    descriptionDisplay(
      "На столе видна личная вещь. Источник: НБРК №231, п. 12. Требуется проверка.",
    ),
  ).toEqual({
    observation: "На столе видна личная вещь.",
    source: "НБРК №231, п. 12. Требуется проверка.",
    quote: undefined,
    qualification: undefined,
  });
});
