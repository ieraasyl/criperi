import { useEffect, useRef, useState, type SubmitEvent } from "react";
import { Link, useNavigate } from "@tanstack/react-router";
import { useMutation, useQuery } from "@tanstack/react-query";
import { reviewJobSchema, submissionSchema, UPLOAD_LIMIT_BYTES } from "@app/contracts";
import { analysisPollingInterval } from "@/lib/analysis-query";
import { api } from "@/lib/api";
import { authClient } from "@/lib/auth-client";
import { ApiRequestError, request } from "@/lib/request";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Field, FieldError, FieldGroup, FieldLabel } from "@/components/ui/field";

export function UploadAnalysis({ email }: { email: string }) {
  const navigate = useNavigate();
  const active = useRef(true);
  useEffect(() => {
    active.current = true;
    return () => {
      active.current = false;
    };
  }, []);
  const [error, setError] = useState("");
  const [accepted, setAccepted] = useState(false);
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
    onSuccess: (result) => {
      if (!active.current) return;
      setAccepted(true);
      void navigate({ to: "/analyses/$jobId", params: { jobId: result.job_id } });
    },
    onError: (cause) => {
      if (!active.current) return;
      if (cause instanceof ApiRequestError && cause.jobId) {
        setAccepted(true);
        void navigate({ to: "/analyses/$jobId", params: { jobId: cause.jobId } });
        return;
      }
      if (
        cause instanceof ApiRequestError &&
        cause.code === "INVALID_RESPONSE" &&
        cause.status === 202
      ) {
        setAccepted(true);
        setError("Видео принято, но ответ не удалось прочитать. Повторная отправка отключена.");
      } else setError(cause instanceof Error ? cause.message : "Не удалось отправить видео.");
    },
    retry: false,
  });
  const signOut = useMutation({
    mutationFn: async () => {
      const result = await authClient.signOut();
      if (result.error) throw new Error("Не удалось выйти из аккаунта.");
    },
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
      setError("Размер файла должен быть меньше 64 МиБ.");
      return;
    }
    upload.mutate(file);
  }
  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p>{email}</p>
        <Button
          variant="outline"
          disabled={signOut.isPending || upload.isPending}
          onClick={() => signOut.mutate()}
        >
          Выйти
        </Button>
      </div>
      {signOut.error && <p role="alert">{signOut.error.message}</p>}
      <h1 className="text-2xl font-semibold">Анализ кассового видео</h1>
      <p>Загрузите MOV-файл. Анализ занимает до пяти минут. Видео и результаты хранятся 24 часа.</p>
      <form noValidate onSubmit={submit}>
        <FieldGroup>
          <Field>
            <FieldLabel htmlFor="file">Видео</FieldLabel>
            <Input
              id="file"
              name="file"
              type="file"
              accept=".mov,video/quicktime"
              disabled={upload.isPending || accepted}
              aria-describedby="upload-error"
            />
            <FieldError id="upload-error">{error}</FieldError>
          </Field>
          <Field>
            <Button type="submit" disabled={upload.isPending || accepted}>
              {upload.isPending ? "Отправка видео…" : "Начать анализ"}
            </Button>
          </Field>
        </FieldGroup>
      </form>
      {upload.isPending && (
        <p role="status">Видео отправляется. Не закрывайте страницу до завершения отправки.</p>
      )}
    </>
  );
}

function timecode(seconds: number) {
  const tenths = Math.round(seconds * 10);
  return `${Math.floor(tenths / 600)}:${((tenths % 600) / 10).toFixed(1).padStart(4, "0")}`;
}

export function AnalysisResult({ userId, jobId }: { userId: string; jobId: string }) {
  const job = useQuery({
    queryKey: ["analysis", userId, jobId],
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
  const data = job.data;
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!data) return;
    const timer = setTimeout(
      () => setNow(Date.now()),
      Math.max(0, Date.parse(data.expires_at) - Date.now()),
    );
    return () => clearTimeout(timer);
  }, [data]);
  const expired =
    (job.error instanceof ApiRequestError && job.error.status === 410) ||
    (data && Date.parse(data.expires_at) <= now);
  return (
    <>
      <Link to="/">Загрузить другое видео</Link>
      <h1 className="text-2xl font-semibold">Результат анализа</h1>
      <p className="break-all text-sm text-muted-foreground">{jobId}</p>
      {expired ? (
        <p role="status">Срок хранения анализа истёк. Видео и результаты удаляются.</p>
      ) : (
        <>
          {job.isPending && !job.error && <p role="status">Загрузка результата…</p>}
          {job.error && (
            <div className="space-y-2">
              <p role="alert">{job.error.message}</p>
              <Button variant="outline" onClick={() => void job.refetch()}>
                Обновить результат
              </Button>
            </div>
          )}
          {data && (
            <>
              <p>Доступно до {new Date(data.expires_at).toLocaleString("ru-RU")}</p>
              {data.status === "processing" && (
                <p role="status">
                  Видео обрабатывается. Результат появится автоматически. Эту страницу можно открыть
                  повторно.
                </p>
              )}
              {data.status === "failed" && (
                <p role="alert">Анализ не завершён. {data.error.message}</p>
              )}
              {data.status === "done" && (
                <>
                  <h2 className="text-xl font-semibold">
                    {data.verdict === "violation"
                      ? "Обнаружены признаки возможных нарушений"
                      : "Нарушений не обнаружено"}
                  </h2>
                  <p>
                    Результат предназначен для проверки человеком и не является окончательным
                    заключением.
                  </p>
                  <ol aria-label="События" className="space-y-8">
                    {data.violations.map((event, index) => (
                      <li key={`${event.timecode_sec}-${index}`} className="space-y-3">
                        <h3 className="text-lg font-semibold">
                          {timecode(event.timecode_sec)} · {event.type}
                        </h3>
                        <p className="whitespace-pre-wrap">{event.description}</p>
                        <a
                          className="underline"
                          href={event.rule.url}
                          target="_blank"
                          rel="noreferrer"
                        >
                          {event.rule.document}
                          {event.rule.point ? `, ${event.rule.point}` : ""}
                        </a>
                        <div className="grid gap-4 sm:grid-cols-3">
                          {event.evidence_frames.map((frame, frameIndex) => (
                            <figure key={frameIndex}>
                              <EvidenceFrame frame={frame} />
                              <figcaption>{timecode(frame.timestamp_sec)}</figcaption>
                            </figure>
                          ))}
                        </div>
                      </li>
                    ))}
                  </ol>
                </>
              )}
            </>
          )}
        </>
      )}
    </>
  );
}

function EvidenceFrame({ frame }: { frame: { timestamp_sec: number; image_base64: string } }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const label = `Кадр события: ${timecode(frame.timestamp_sec)}`;
  return (
    <>
      <button
        type="button"
        className="block w-full"
        aria-label={`Открыть кадр ${timecode(frame.timestamp_sec)}`}
        onClick={() => dialog.current?.showModal()}
      >
        <img
          className="h-auto w-full rounded-md"
          src={`data:image/jpeg;base64,${frame.image_base64}`}
          alt={label}
        />
      </button>
      <dialog
        ref={dialog}
        aria-label={label}
        className="fixed inset-0 m-auto max-h-screen max-w-4xl overflow-auto rounded-lg border bg-background p-4 text-foreground"
      >
        <div className="space-y-3">
          <Button variant="outline" onClick={() => dialog.current?.close()}>
            Закрыть кадр
          </Button>
          <p>{label}</p>
          <img
            className="h-auto w-full"
            src={`data:image/jpeg;base64,${frame.image_base64}`}
            alt={label}
          />
        </div>
      </dialog>
    </>
  );
}
