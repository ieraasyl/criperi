import { useEffect, useRef, useState, type SubmitEvent } from "react";
import { useNavigate } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  reviewJobSchema,
  submissionSchema,
  eventReviewSchema,
  UPLOAD_LIMIT_BYTES,
  type ReviewJob,
  type SaveReview,
} from "@app/contracts";
import { isReviewed, expiryDisplay, counted } from "@/lib/review";
import { exportReport } from "@/lib/report";
import { analysisPollingInterval } from "@/lib/analysis-query";
import { Findings } from "@/components/findings";
import { api } from "@/lib/api";
import { ApiRequestError, request } from "@/lib/request";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogTitle,
  AlertDialogDescription,
  AlertDialogHeader,
  AlertDialogFooter,
  AlertDialogAction,
  AlertDialogCancel,
} from "@/components/ui/alert-dialog";
import { Input } from "@/components/ui/input";
import { Field, FieldDescription, FieldError, FieldGroup, FieldLabel } from "@/components/ui/field";
import { LiveAnalysis } from "@/components/live-analysis";

export function UploadAnalysis({ userId }: { userId: string }) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const active = useRef(true);
  useEffect(() => {
    active.current = true;
    return () => {
      active.current = false;
    };
  }, []);
  const [error, setError] = useState("");
  const [accepted, setAccepted] = useState(false);
  const [mode, setMode] = useState<"full" | "live">("full");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const upload = useMutation({
    mutationFn: async (file: File) => {
      const body = new FormData();
      body.append("file", file);
      const result = await request((init) => api.analyses.$post({}, { init }), { body });
      const parsed = submissionSchema.safeParse(result);
      if (!parsed.success)
        throw new ApiRequestError("Сервер вернул некорректный ответ.", 202, "INVALID_RESPONSE");
      return parsed.data;
    },
    onSuccess: (result, file) => {
      if (!active.current) return;
      queryClient.setQueryDefaults(["analysis-video", userId], { gcTime: 24 * 60 * 60 * 1000 });
      queryClient.setQueryData(["analysis-video", userId, result.job_id], file);
      setAccepted(true);
      void navigate({ to: "/analyses/$jobId", params: { jobId: result.job_id } });
    },
    onError: (cause) => {
      if (!active.current) return;
      if (
        cause instanceof ApiRequestError &&
        (cause.code === "SUBMISSION_UNCONFIRMED" ||
          (cause.code === "INVALID_RESPONSE" && cause.status === 202))
      ) {
        setAccepted(true);
        setError(
          cause.code === "SUBMISSION_UNCONFIRMED"
            ? cause.message
            : "Видео принято, но ответ не удалось прочитать. Повторная отправка отключена.",
        );
      } else setError(cause instanceof Error ? cause.message : "Не удалось отправить видео.");
    },
    retry: false,
  });
  function submit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    if (upload.isPending || accepted) return;
    setError("");
    const file = new FormData(event.currentTarget).get("file");
    if (!(file instanceof File) || !file.size || !file.name.toLowerCase().endsWith(".mov")) {
      setError("Выберите видео в формате MOV.");
      return;
    }
    if (file.size > UPLOAD_LIMIT_BYTES - 4096) {
      setError("Файл слишком большой. Выберите MOV-файл размером меньше 64 МиБ.");
      return;
    }
    upload.mutate(file);
  }
  return (
    <>
      <h1 className="text-2xl font-semibold">Анализ кассового видео</h1>
      <div role="group" aria-label="Режим анализа" className="flex flex-wrap gap-2">
        <Button
          variant={mode === "full" ? "default" : "outline"}
          aria-pressed={mode === "full"}
          disabled={upload.isPending || accepted}
          onClick={() => setMode("full")}
        >
          Полный анализ
        </Button>
        <Button
          variant={mode === "live" ? "default" : "outline"}
          aria-pressed={mode === "live"}
          disabled={upload.isPending || accepted}
          onClick={() => setMode("live")}
        >
          При воспроизведении
        </Button>
      </div>
      {mode === "live" ? (
        <LiveAnalysis />
      ) : (
        <>
          <form noValidate onSubmit={submit} aria-busy={upload.isPending} className="max-w-xl">
            <FieldGroup>
              <Field data-invalid={!!error}>
                <FieldLabel htmlFor="file">Видеозапись</FieldLabel>
                <FieldDescription id="file-help">MOV, меньше 64 МиБ.</FieldDescription>
                <Input
                  id="file"
                  name="file"
                  type="file"
                  accept=".mov,video/quicktime"
                  disabled={upload.isPending || accepted}
                  aria-invalid={!!error}
                  aria-describedby={error ? "file-help upload-error" : "file-help"}
                  onChange={(event) => {
                    setSelectedFile(event.target.files?.[0] ?? null);
                    setError("");
                  }}
                />
                {selectedFile && (
                  <p role="status" className="text-sm break-all text-muted-foreground">
                    Выбран файл: {selectedFile.name} (
                    {(selectedFile.size / 1024 / 1024).toLocaleString("ru-RU", {
                      maximumFractionDigits: 1,
                    })}{" "}
                    МиБ)
                  </p>
                )}
                <FieldError id="upload-error">{error}</FieldError>
              </Field>
              <Field>
                <Button
                  type="submit"
                  disabled={upload.isPending || accepted}
                  size="lg"
                  className="sm:w-fit"
                >
                  {upload.isPending ? "Отправка видео…" : "Начать анализ"}
                </Button>
                <FieldDescription>Результат доступен 24 часа в этом браузере. Очистка cookies удалит доступ.</FieldDescription>
              </Field>
            </FieldGroup>
          </form>
          {upload.isPending && (
            <p role="status">Видео отправляется. Не закрывайте страницу до завершения отправки.</p>
          )}
        </>
      )}
    </>
  );
}

export function AnalysisResult({ userId, jobId }: { userId: string; jobId: string }) {
  const navigate = useNavigate();
  const [confirmNew, setConfirmNew] = useState(false);
  const queryClient = useQueryClient();
  const queryKey = ["analysis", userId, jobId];
  const [draftDirty, setDraftDirty] = useState(false);
  const video = useQuery<File | null>({
    queryKey: ["analysis-video", userId, jobId],
    queryFn: () => null,
    enabled: false,
    gcTime: 24 * 60 * 60 * 1000,
  });
  const job = useQuery({
    queryKey,
    queryFn: async ({ signal }) => {
      const result = await request(
        (init) => api.analyses[":job_id"].$get({ param: { job_id: jobId } }, { init }),
        { signal },
      );
      const parsed = reviewJobSchema.safeParse(result);
      if (!parsed.success)
        throw new ApiRequestError("Сервис вернул некорректный результат.", 200, "INVALID_RESPONSE");
      return parsed.data;
    },
    refetchInterval: (query) => {
      const error = query.state.error;
      if (error instanceof ApiRequestError && [401, 404, 410].includes(error.status)) return false;
      return analysisPollingInterval(query.state.data);
    },
    refetchOnWindowFocus: false,
  });
  const save = useMutation({
    mutationFn: async ({ index, input }: { index: number; input: SaveReview }) => {
      try {
        const result = await request((init) =>
          api.analyses[":job_id"].reviews[":event_index"].$put(
            { param: { job_id: jobId, event_index: String(index) }, json: input },
            { init },
          ),
        );
        const parsed = eventReviewSchema.safeParse(result.review);
        return parsed.success &&
          parsed.data.event_index === index &&
          parsed.data.status === input.status &&
          parsed.data.comment === input.comment
          ? { review: parsed.data, unreadable: false }
          : { review: undefined, unreadable: true };
      } catch (error) {
        if (
          error instanceof ApiRequestError &&
          error.code === "INVALID_RESPONSE" &&
          error.status >= 200 &&
          error.status < 300
        )
          return { review: undefined, unreadable: true };
        throw error;
      }
    },
    retry: false,
    onSuccess: async ({ review }) => {
      await queryClient.cancelQueries({ queryKey });
      if (review)
        queryClient.setQueryData<ReviewJob>(
          queryKey,
          (current) =>
            current && {
              ...current,
              reviews: [
                ...(current.reviews ?? []).filter(
                  (item) => item.event_index !== review.event_index,
                ),
                review,
              ],
            },
        );
      else void queryClient.invalidateQueries({ queryKey });
    },
  });
  const data = job.data;
  const awaitingReview = Boolean(
    save.data?.unreadable &&
    !data?.reviews?.some(
      (item) =>
        item.event_index === save.variables?.index &&
        item.status === save.variables.input.status &&
        item.comment === save.variables.input.comment,
    ),
  );
  const reviewed =
    data?.status === "done"
      ? data.violations.filter((_, index) =>
          isReviewed(data.reviews?.find((review) => review.event_index === index)),
        ).length
      : 0;
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!data) return;
    const timer = setTimeout(
      () => setNow(Date.now()),
      Math.max(0, Date.parse(data.expires_at) - Date.now()),
    );
    return () => clearTimeout(timer);
  }, [data]);
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 60_000);
    return () => clearInterval(timer);
  }, []);
  useEffect(() => {
    if (!draftDirty && !save.isPending) return;
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [draftDirty, save.isPending]);
  const expired =
    (job.error instanceof ApiRequestError && job.error.status === 410) ||
    (data && Date.parse(data.expires_at) <= now);
  return (
    <>
      <div className="sticky top-0 z-10 flex flex-wrap items-center justify-between gap-3 border-b bg-background py-3 sm:flex-nowrap">
        <h1 className="min-w-0 text-xl font-semibold sm:text-2xl">Результат анализа</h1>
        <div className="flex w-full flex-wrap justify-between gap-2 sm:w-auto sm:justify-end">
          <Button
            variant="outline"
            disabled={save.isPending}
            onClick={() => {
              if (draftDirty || (data?.status === "done" && reviewed < data.violations.length))
                setConfirmNew(true);
              else void navigate({ to: "/" });
            }}
          >
            Новый анализ
          </Button>
          {data?.status === "done" && !expired && (
            <Tooltip
              disabled={
                !(
                  save.isPending ||
                  awaitingReview ||
                  draftDirty ||
                  reviewed < data.violations.length
                )
              }
            >
              <TooltipTrigger render={<span tabIndex={0} className="inline-flex" />}>
                <Button
                  disabled={
                    save.isPending ||
                    awaitingReview ||
                    draftDirty ||
                    reviewed < data.violations.length
                  }
                  onClick={() => exportReport(data)}
                >
                  Экспорт отчёта
                </Button>
              </TooltipTrigger>
              <TooltipContent side="bottom" align="end">
                {save.isPending || awaitingReview
                  ? "Дождитесь подтверждения сохранения решения."
                  : draftDirty
                    ? "Сохраните изменения перед экспортом."
                    : "Подтвердите или отклоните каждый эпизод перед экспортом. Статус «Нужна проверка» не завершает проверку."}
              </TooltipContent>
            </Tooltip>
          )}
        </div>
      </div>
      <AlertDialog open={confirmNew} onOpenChange={setConfirmNew}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Начать новый анализ?</AlertDialogTitle>
            <AlertDialogDescription>
              {draftDirty
                ? "Несохранённые изменения будут потеряны. Сохранённые решения останутся в этом результате."
                : "Есть события без окончательного решения. Сохранённые решения останутся в этом результате."}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Продолжить проверку</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                setConfirmNew(false);
                void navigate({ to: "/" });
              }}
            >
              Начать новый анализ
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      {expired ? (
        <div role="status" className="space-y-1 py-2">
          <h2 className="font-semibold">Срок хранения анализа истёк</h2>
          <p className="text-sm text-muted-foreground">Результат больше недоступен в приложении.</p>
        </div>
      ) : (
        <>
          {job.isPending && !job.error && <p role="status">Загрузка результата…</p>}
          {job.error && (
            <div className="space-y-2 py-2">
              {data && (
                <p className="text-sm text-muted-foreground">
                  Не удалось обновить результат. Ниже показаны последние полученные данные.
                </p>
              )}
              <p role="alert">{job.error.message}</p>
              <Button variant="outline" onClick={() => void job.refetch()}>
                Обновить результат
              </Button>
            </div>
          )}
          {data && (
            <>
              <section
                aria-label="Сводка анализа"
                className="flex flex-wrap items-start justify-between gap-x-6 gap-y-2 border-b pb-3"
              >
                <div className="min-w-0 basis-full sm:flex-1 sm:basis-auto">
                  {data.status === "processing" && (
                    <div role="status" className="space-y-1">
                      <h2 className="text-lg font-semibold">Видео обрабатывается</h2>
                      <p className="text-sm text-muted-foreground">
                        Результат появится автоматически.
                      </p>
                    </div>
                  )}
                  {data.status === "failed" && (
                    <div role="alert" className="space-y-1">
                      <h2 className="font-semibold">Анализ не завершён</h2>
                      <p>{data.error.message}</p>
                    </div>
                  )}
                  {data.status === "done" && (
                    <div role="status" className="space-y-1">
                      <h2 className="text-base font-semibold sm:text-lg">
                        {data.verdict === "violation"
                          ? `Обнаружено ${counted(data.violations.length, ["событие", "события", "событий"])} для проверки`
                          : "Нарушений не обнаружено"}
                      </h2>
                      <p className="text-sm">
                        Результат предназначен для проверки человеком и не является окончательным
                        заключением.
                      </p>
                    </div>
                  )}
                </div>
                <p
                  className={`text-xs sm:text-right ${expiryDisplay(data.expires_at, now).urgent ? "font-medium text-destructive" : "text-muted-foreground"}`}
                >
                  <span className="mr-1 sm:mr-0 sm:block">Результат хранится до</span>
                  <time dateTime={data.expires_at}>
                    {new Date(data.expires_at).toLocaleString("ru-RU", {
                      dateStyle: "short",
                      timeStyle: "short",
                    })}
                  </time>
                  <span className="ml-1 sm:ml-0 sm:block">
                    ({expiryDisplay(data.expires_at, now).remaining})
                  </span>
                  {expiryDisplay(data.expires_at, now).urgent && (
                    <span role="status" className="block">
                      Скоро истечёт. Экспортируйте отчёт.
                    </span>
                  )}
                </p>
              </section>
              {save.error && (
                <p role="alert" className="text-sm text-destructive">
                  {save.error.message}
                </p>
              )}
              {awaitingReview && (
                <div className="flex flex-wrap items-center gap-2 text-sm">
                  <p role="status">
                    Решение принято, но ответ не удалось прочитать. Обновите результат, чтобы
                    проверить сохранение.
                  </p>
                  <Button variant="outline" onClick={() => void job.refetch()}>
                    Обновить результат
                  </Button>
                </div>
              )}
              {data.status === "done" && (
                <Findings
                  key={`${userId}-${jobId}`}
                  jobId={jobId}
                  events={data.violations}
                  reviews={data.reviews ?? []}
                  saving={save.isPending || awaitingReview}
                  onReview={async (index, input) => {
                    try {
                      const result = await save.mutateAsync({ index, input });
                      return !result.unreadable;
                    } catch {
                      return false;
                    }
                  }}
                  onDirtyChange={setDraftDirty}
                  videoFile={video.data ?? null}
                  onVideoFile={(file) =>
                    queryClient.setQueryData(["analysis-video", userId, jobId], file)
                  }
                />
              )}
            </>
          )}
        </>
      )}
    </>
  );
}
