import { useEffect, useState } from "react";
import { useNavigate, useSearch } from "@tanstack/react-router";
import type { EventReview, SaveReview } from "@app/contracts";
import { descriptionDisplay, findingDisplay } from "@/lib/finding-display";
import {
  closestFrame,
  documentUrl,
  eventGroups,
  isReviewed,
  observationDisplay,
  timecode,
  type Finding,
} from "@/lib/review";
import { Button } from "@/components/ui/button";
import { EvidenceViewer } from "./evidence-viewer";
import { Sheet, SheetContent, SheetTitle, SheetClose, SheetTrigger } from "@/components/ui/sheet";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Field, FieldLabel } from "@/components/ui/field";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  ArrowRight01Icon,
  ArrowUpRight01Icon,
  Copy01Icon,
  Link01Icon,
} from "@hugeicons/core-free-icons";
import { SourceVideo } from "./source-video";
import { ReviewIcon, ReviewStatus } from "./review-status";
import { FindingCode } from "./finding-code";

type Props = {
  jobId: string;
  events: Finding[];
  reviews: EventReview[];
  saving: boolean;
  onReview: (index: number, input: SaveReview) => Promise<boolean>;
  onDirtyChange: (dirty: boolean) => void;
  videoFile: File | null;
  onVideoFile: (file: File) => void;
};

export function Findings({
  jobId,
  events,
  reviews,
  saving,
  onReview,
  onDirtyChange,
  videoFile,
  onVideoFile,
}: Props) {
  const search = useSearch({ from: "/_authenticated/analyses/$jobId" });
  const navigate = useNavigate();
  const [normOpen, setNormOpen] = useState(false);
  const [copyMessage, setCopyMessage] = useState("");
  const [drafts, setDrafts] = useState<Record<number, string>>({});
  const [decisions, setDecisions] = useState<Record<number, SaveReview["status"]>>({});
  const current = search.event && search.event <= events.length ? search.event - 1 : 0;
  const event = events[current];
  const frame =
    event && search.frame && search.frame <= event.evidence_frames.length
      ? search.frame - 1
      : event
        ? closestFrame(event)
        : 0;
  const dirty =
    Object.entries(decisions).some(
      ([index, status]) =>
        status !== reviews.find((item) => item.event_index === Number(index))?.status,
    ) ||
    Object.entries(drafts).some(
      ([index, text]) =>
        text.trim() !== (reviews.find((item) => item.event_index === Number(index))?.comment ?? ""),
    );
  useEffect(() => {
    onDirtyChange(dirty);
  }, [dirty, onDirtyChange]);
  useEffect(() => () => onDirtyChange(false), [onDirtyChange]);
  function select(index: number, nextFrame = closestFrame(events[index]!)) {
    setNormOpen(false);
    setCopyMessage("");
    void navigate({
      to: "/analyses/$jobId",
      params: { jobId },
      search: { event: index + 1, frame: nextFrame + 1 },
      replace: true,
    });
  }
  useEffect(() => {
    const keydown = (key: KeyboardEvent) => {
      if (key.altKey || key.ctrlKey || key.metaKey) return;
      const target = key.target;
      if (
        !(target instanceof HTMLElement) ||
        target.closest(
          "input,textarea,select,[contenteditable=true],[role=combobox],[role=listbox],[role=option],[role=dialog],[role=alertdialog]",
        ) ||
        document.querySelector("[role=dialog],[role=alertdialog]")
      )
        return;
      if (key.key === "ArrowUp" || key.key === "ArrowDown") {
        key.preventDefault();
        const next = Math.max(
          0,
          Math.min(events.length - 1, current + (key.key === "ArrowDown" ? 1 : -1)),
        );
        select(next);
        document.querySelector<HTMLButtonElement>(`[data-event-index="${next}"]`)?.focus();
      }
      if (
        key.key === "Enter" &&
        (target === document.body || target.closest("nav[aria-label='Выбор события']"))
      ) {
        key.preventDefault();
        const row = target.closest<HTMLButtonElement>("[data-event-index]");
        if (row && Number(row.dataset.eventIndex) !== current)
          select(Number(row.dataset.eventIndex));
        else setNormOpen(true);
      }
    };
    window.addEventListener("keydown", keydown);
    return () => window.removeEventListener("keydown", keydown);
  });
  if (!event) return null;
  const finding = findingDisplay(event.type);
  const description = descriptionDisplay(event.description);
  const review = reviews.find((item) => item.event_index === current);
  const comment = drafts[current] ?? review?.comment ?? "";
  const decision = decisions[current] ?? review?.status;
  const changed = decision !== review?.status || comment.trim() !== (review?.comment ?? "");
  const reviewed = events.filter((_, index) =>
    isReviewed(reviews.find((item) => item.event_index === index)),
  ).length;
  const next = Array.from(
    { length: events.length - 1 },
    (_, offset) => (current + offset + 1) % events.length,
  ).find((index) => !isReviewed(reviews.find((item) => item.event_index === index)));
  async function saveAndNext() {
    if (!decision || saving) return;
    if (!changed) {
      if (next !== undefined) select(next);
      return;
    }
    const saved = await onReview(current, { status: decision, comment: comment.trim() });
    if (saved && next !== undefined) select(next);
  }
  const source = documentUrl(finding.reference, description.source);
  const link = new URL(`/analyses/${jobId}`, window.location.origin);
  link.searchParams.set("event", String(current + 1));
  link.searchParams.set("frame", String(frame + 1));
  const status = (index: number) => reviews.find((item) => item.event_index === index);
  async function copy(text: string, success: string) {
    try {
      await navigator.clipboard.writeText(text);
      setCopyMessage(success);
    } catch {
      setCopyMessage("Не удалось скопировать. Выделите текст и скопируйте вручную.");
    }
  }
  return (
    <Sheet open={normOpen} onOpenChange={setNormOpen}>
      <section aria-label="Проверка событий" className="flex flex-col gap-5 lg:flex-row lg:gap-6">
        <div className="min-w-0 space-y-2 lg:hidden">
          <Field>
            <FieldLabel htmlFor="finding-select">
              Событие для проверки: {current + 1} из {events.length}
            </FieldLabel>
            <Select
              value={current}
              items={events.map((item, index) => ({
                value: index,
                label: `${index + 1}. ${[findingDisplay(item.type).code, findingDisplay(item.type).title].filter(Boolean).join(" ")}`,
              }))}
              onValueChange={(index) => {
                if (index !== null) select(index);
              }}
            >
              <SelectTrigger
                id="finding-select"
                aria-controls="selected-finding"
                className="w-full min-w-0"
              >
                <SelectValue className="min-w-0" />
              </SelectTrigger>
              <SelectContent alignItemWithTrigger={false}>
                {events.map((item, index) => (
                  <SelectItem key={index} value={index} multiline>
                    <span className="flex items-start gap-2">
                      <ReviewStatus status={status(index)?.status} />
                      <span>
                        {index + 1}. {findingDisplay(item.type).code}{" "}
                        {findingDisplay(item.type).title} (видео: {timecode(item.timecode_sec)})
                      </span>
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
        </div>
        <nav
          aria-label="Выбор события"
          className="hidden min-w-0 shrink-0 lg:block lg:w-64 lg:border-r lg:pr-5 2xl:w-72"
        >
          <h2 className="mb-2 text-sm font-semibold">
            Проверено {reviewed} из {events.length}
          </h2>
          <ol
            aria-label="События"
            className="max-h-128 space-y-3 overflow-y-auto overscroll-contain"
          >
            {eventGroups(events).map((group) => (
              <li key={group.title} className="space-y-1">
                <div className="text-sm font-medium">
                  <FindingCode code={group.code} title={group.title} /> {group.title}
                </div>
                <ol className="divide-y">
                  {group.indices.map((index, episode) => {
                    const item = events[index]!;
                    const saved = status(index);
                    const thumbnail = item.evidence_frames[closestFrame(item)]!;
                    return (
                      <li key={index}>
                        <Button
                          variant="navigation"
                          size="content"
                          data-event-index={index}
                          aria-pressed={index === current}
                          aria-controls="selected-finding"
                          onClick={() => select(index)}
                          className="w-full"
                        >
                          <img
                            className="h-16 w-10 shrink-0 object-contain"
                            src={`data:image/jpeg;base64,${thumbnail.image_base64}`}
                            alt=""
                          />
                          <span className="min-w-0 space-y-1 text-xs">
                            <span className="block font-medium">
                              {group.indices.length > 1
                                ? `Эпизод ${episode + 1}/${group.indices.length}`
                                : "Событие"}
                            </span>
                            <span className="block tabular-nums">
                              Время на видео: {timecode(item.timecode_sec)}
                            </span>
                            <ReviewStatus status={saved?.status} />
                          </span>
                        </Button>
                      </li>
                    );
                  })}
                </ol>
              </li>
            ))}
          </ol>
        </nav>
        <article
          id="selected-finding"
          aria-labelledby="finding-title"
          className="grid min-w-0 flex-1 content-start gap-5 xl:grid-cols-2 xl:gap-6"
        >
          <div className="min-w-0 space-y-4">
            <div className="space-y-2">
              <div className="min-w-0">
                <FindingCode code={finding.code} title={finding.title} />{" "}
                <h3 id="finding-title" className="inline text-lg font-semibold">
                  {finding.title}
                </h3>{" "}
                <span className="inline-flex align-middle">
                  <Tooltip>
                    <TooltipTrigger
                      render={
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          aria-label="Копировать ссылку на событие"
                          onClick={() => void copy(link.href, "Ссылка скопирована")}
                        />
                      }
                    >
                      <HugeiconsIcon icon={Link01Icon} strokeWidth={2} />
                    </TooltipTrigger>
                    <TooltipContent>Копировать ссылку на событие</TooltipContent>
                  </Tooltip>
                </span>
              </div>
              <div className="flex flex-wrap items-center gap-2 text-sm">
                <span className="tabular-nums">
                  Время на видео: <strong>{timecode(event.timecode_sec)}</strong>
                </span>
                {(finding.reference || description.source) && (
                  <SheetTrigger
                    render={<Button variant="outline" size="sm" />}
                    onClick={() => setCopyMessage("")}
                  >
                    {finding.reference ?? "Нормативное основание"}
                    <HugeiconsIcon icon={ArrowRight01Icon} strokeWidth={2} />
                  </SheetTrigger>
                )}
              </div>
              {copyMessage && !normOpen && (
                <p role="status" className="text-xs">
                  {copyMessage}
                </p>
              )}
            </div>
            {description.observation && (
              <div className="space-y-1">
                <h4 className="text-sm font-medium">Наблюдение</h4>
                <p className="text-sm whitespace-pre-wrap">
                  {observationDisplay(description.observation)}
                </p>
              </div>
            )}
            {description.qualification && (
              <div className="space-y-1 border-l-2 pl-3">
                <h4 className="text-sm font-medium">Ограничения вывода</h4>
                <p className="text-sm whitespace-pre-wrap">{description.qualification}</p>
              </div>
            )}
            <div className="space-y-2 border-t pt-3">
              <h4 className="text-sm font-medium">Решение проверяющего</h4>
              <Field>
                <FieldLabel htmlFor="review-comment">Комментарий (необязательно)</FieldLabel>
                <Textarea
                  id="review-comment"
                  value={comment}
                  maxLength={4000}
                  disabled={saving}
                  onChange={(key) =>
                    setDrafts((values) => ({ ...values, [current]: key.target.value }))
                  }
                  className="min-h-20"
                />
              </Field>
              <div
                role="group"
                aria-label="Решение проверяющего"
                className="flex w-full flex-wrap gap-2"
              >
                {(
                  [
                    ["confirmed", "Подтвердить"],
                    ["rejected", "Отклонить"],
                    ["needs_review", "Нужна проверка"],
                  ] as const
                ).map(([decision, label]) => (
                  <Button
                    key={decision}
                    variant={decision}
                    className="h-auto min-h-8 min-w-min flex-1 shrink whitespace-normal"
                    aria-pressed={(decisions[current] ?? review?.status) === decision}
                    disabled={saving}
                    onClick={() => setDecisions((values) => ({ ...values, [current]: decision }))}
                  >
                    <ReviewIcon status={decision} />
                    <span>{label}</span>
                  </Button>
                ))}
              </div>
              <Button
                variant="outline"
                className="w-full"
                disabled={saving || !decision || (!changed && next === undefined)}
                onClick={() => void saveAndNext()}
              >
                {next !== undefined ? "Сохранить и перейти к следующему" : "Сохранить решение"}
              </Button>
              {(saving || changed) && (
                <p role="status" className="text-xs">
                  {saving ? "Сохраняем решение…" : "Изменения не сохранены"}
                </p>
              )}
            </div>
          </div>
          <div className="min-w-0 space-y-4">
            <EvidenceViewer
              key={`${current}-${event.type}`}
              event={event}
              title={finding.title}
              selected={frame}
              onSelect={(index) => select(current, index)}
            />
            <SourceVideo
              key={current}
              file={videoFile}
              onFile={onVideoFile}
              seconds={event.timecode_sec}
            />
          </div>
        </article>
        <SheetContent
          aria-label="Нормативное основание"
          showCloseButton={false}
          className="overflow-y-auto"
        >
          <div className="p-6">
            <div className="space-y-4">
              <div className="flex items-start justify-between gap-3">
                <SheetTitle>Нормативное основание</SheetTitle>
                <SheetClose render={<Button variant="outline" />}>Закрыть</SheetClose>
              </div>
              <h3 className="font-semibold">{finding.title}</h3>
              {description.source && (
                <div className="space-y-1">
                  <h4 className="text-sm font-medium">Нормативный источник</h4>
                  <p className="text-sm whitespace-pre-wrap">{description.source}</p>
                </div>
              )}
              {description.quote && (
                <div className="space-y-2">
                  <h4 className="text-sm font-medium">Текст нормы</h4>
                  <blockquote className="border-l-2 pl-3 whitespace-pre-wrap">
                    «{description.quote}»
                  </blockquote>
                  <Button
                    variant="outline"
                    onClick={() => void copy(description.quote!, "Цитата скопирована")}
                  >
                    <HugeiconsIcon icon={Copy01Icon} strokeWidth={2} /> Копировать цитату
                  </Button>
                </div>
              )}
              {description.qualification && <p className="text-sm">{description.qualification}</p>}
              {source && (
                <a
                  href={source}
                  target="_blank"
                  rel="noreferrer"
                  className="block text-sm underline underline-offset-4"
                >
                  Открыть источник на Adilet{" "}
                  <HugeiconsIcon
                    icon={ArrowUpRight01Icon}
                    strokeWidth={2}
                    className="inline size-4"
                  />
                </a>
              )}
              {copyMessage && (
                <p role="status" className="text-sm">
                  {copyMessage}
                </p>
              )}
            </div>
          </div>
        </SheetContent>
      </section>
    </Sheet>
  );
}
