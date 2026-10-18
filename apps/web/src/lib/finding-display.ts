export function findingDisplay(type: string) {
  const coded = /^(V\d+)\.\s+(.+)$/i.exec(type);
  const text = coded?.[2] ?? type;
  const cited = /^(.*?)\s*\(((?:НБРК|АРРФР)\s+№.+)\)$/.exec(text);
  const title = cited?.[1]?.trim() ?? text;

  // Keep unrecognized or incomplete titles intact.
  if (!title) return { title: type, code: undefined, reference: undefined };

  return { title, code: coded?.[1], reference: cited?.[2] };
}

export function descriptionDisplay(description: string) {
  let text = description.trim();
  let qualification: string | undefined;
  const qualifications = [
    "Связь с нормой косвенная; результат является подсказкой для проверки человеком.",
    "Открытой нормы в актах adilet.zan.kz не найдено; класс задан правилами кейса.",
  ];

  for (const suffix of qualifications) {
    if (text === suffix || text.endsWith(` ${suffix}`)) {
      qualification = suffix;
      text = text.slice(0, -suffix.length).trim();
      break;
    }
  }

  const sourceMarker = /(?:^|\s)Источник:\s*/.exec(text);
  if (!sourceMarker) {
    return { observation: text, source: undefined, quote: undefined, qualification };
  }

  const observation = text.slice(0, sourceMarker.index).trim();
  const source = text.slice(sourceMarker.index + sourceMarker[0].length);
  if (!source.trim()) {
    return { observation: text, source: undefined, quote: undefined, qualification };
  }
  const quoted = /^([\s\S]+?):\s*«([\s\S]+)»$/.exec(source);

  // Leave unfamiliar source formats intact, including any additional commentary.
  return {
    observation,
    source: quoted?.[1] ?? source,
    quote: quoted?.[2],
    qualification,
  };
}
