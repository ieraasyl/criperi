import { useEffect, useRef, useState } from "react";
import { LIVE_IMAGE_LIMIT, type LiveFrame } from "@app/contracts";
import {
  LiveAnalysisSession,
  liveTimecode,
  mergeLiveFindings,
  type LiveFinding,
} from "@/lib/live-analysis";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Field, FieldLabel } from "@/components/ui/field";

const stages = {
  layout: "Определение рабочих зон",
  detecting: "Распознавание людей и предметов",
  verifying: "Проверка возможных нарушений",
};

export function LiveAnalysis() {
  const video = useRef<HTMLVideoElement>(null);
  const session = useRef<LiveAnalysisSession | null>(null);
  const autoPaused = useRef(false);
  const restarting = useRef(false);
  const lastSample = useRef(-Infinity);
  const finalSample = useRef<(() => void) | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [running, setRunning] = useState(false);
  const [busy, setBusy] = useState(false);
  const [buffering, setBuffering] = useState(false);
  const [status, setStatus] = useState("Выберите видео и начните воспроизведение с анализом.");
  const [error, setError] = useState("");
  const [snapshot, setSnapshot] = useState<LiveFrame | null>(null);
  const [findings, setFindings] = useState<LiveFinding[]>([]);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [analyzedThrough, setAnalyzedThrough] = useState<number | null>(null);

  function stop(message = "Анализ остановлен. Найденные события остаются на странице.") {
    session.current?.stop();
    session.current = null;
    autoPaused.current = false;
    restarting.current = false;
    video.current?.pause();
    setRunning(false);
    setBusy(false);
    setBuffering(false);
    setStatus(message);
  }

  useEffect(() => {
    const player = video.current;
    if (!file || !player) return;
    const url = URL.createObjectURL(file);
    player.src = url;
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
    let callback = 0;
    let disposed = false;
    const nativeCallbacks = typeof player.requestVideoFrameCallback === "function";
    function sampleFrame(final = false) {
      if (disposed) return;
      const current = session.current;
      if (
        current &&
        (!player!.paused || final) &&
        !player!.seeking &&
        player!.readyState >= 2 &&
        player!.currentTime - lastSample.current >= (final ? 0.5 : 1)
      ) {
        if (!context || !player!.videoWidth || !player!.videoHeight) {
          stop("Браузер не смог прочитать кадры видео.");
        } else {
          const scale = Math.min(1, 960 / Math.max(player!.videoWidth, player!.videoHeight));
          canvas.width = Math.round(player!.videoWidth * scale);
          canvas.height = Math.round(player!.videoHeight * scale);
          try {
            context.drawImage(player!, 0, 0, canvas.width, canvas.height);
            const image = canvas.toDataURL("image/jpeg", 0.7).split(",")[1]!;
            if (image.length > LIVE_IMAGE_LIMIT) throw new Error("Frame too large");
            lastSample.current = player!.currentTime;
            current.add({
              timestamp_sec: Math.round(player!.currentTime * 1000) / 1000,
              image_base64: image,
            });
          } catch {
            stop("Не удалось прочитать кадр. Попробуйте видео меньшего разрешения в формате MP4.");
          }
        }
      }
    }
    finalSample.current = () => sampleFrame(true);
    function sample() {
      if (disposed) return;
      sampleFrame();
      callback = nativeCallbacks
        ? player!.requestVideoFrameCallback(sample)
        : requestAnimationFrame(sample);
    }
    sample();
    return () => {
      disposed = true;
      finalSample.current = null;
      if (nativeCallbacks) player.cancelVideoFrameCallback(callback);
      else cancelAnimationFrame(callback);
      session.current?.stop();
      session.current = null;
      autoPaused.current = false;
      player.pause();
      player.removeAttribute("src");
      URL.revokeObjectURL(url);
    };
  }, [file]);

  function start() {
    const player = video.current;
    if (!player || !file) return;
    session.current?.stop();
    if (player.ended) {
      restarting.current = true;
      player.currentTime = 0;
    }
    setError("");
    setFindings([]);
    setSnapshot(null);
    setAnalyzedThrough(null);
    setBuffering(false);
    autoPaused.current = false;
    lastSample.current = -Infinity;
    setRunning(true);
    setStatus("Сбор кадров во время воспроизведения");
    session.current = new LiveAnalysisSession({
      onEvent: (event) => {
        if (event.type === "stage") setStatus(stages[event.stage]);
        if (event.type === "frame")
          setSnapshot((current) =>
            !current || event.timestamp_sec >= current.timestamp_sec ? event : current,
          );
        if (event.type === "result") {
          setAnalyzedThrough(event.to_sec);
          setFindings((current) => mergeLiveFindings(current, event));
          setStatus("Ожидание следующих кадров");
        }
      },
      onBusy: setBusy,
      onBuffering: (waiting) => {
        setBuffering(waiting);
        if (waiting) {
          autoPaused.current = !player.paused;
          player.pause();
        } else if (autoPaused.current) {
          autoPaused.current = false;
          void player.play().catch(() => stop("Запустите анализ снова, чтобы продолжить."));
        }
      },
      onComplete: () => {
        session.current = null;
        setRunning(false);
        setBusy(false);
        setStatus("Все полученные кадры обработаны.");
      },
      onError: (message) => {
        stop();
        setError(message);
      },
    });
    void player.play().catch(() => {
      stop();
      setError("Браузер не смог воспроизвести видео. Выберите MP4 с видеокодеком H.264.");
    });
  }

  return (
    <section aria-label="Анализ при воспроизведении" className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Кадры отправляются на сервер во время просмотра. События появляются после проверки. Порядок
        действий за весь визит клиента проверяется в полном анализе.
      </p>
      <p className="text-sm text-muted-foreground">
        События хранятся только на этой странице и исчезнут после её закрытия.
      </p>
      <Field>
        <FieldLabel htmlFor="live-file">Видео для просмотра</FieldLabel>
        <Input
          id="live-file"
          type="file"
          accept="video/*,.mov,.mp4,.webm"
          onChange={(event) => {
            stop();
            setFile(event.target.files?.[0] ?? null);
            setFindings([]);
            setSnapshot(null);
            setAnalyzedThrough(null);
            setPlaybackTime(0);
            setError("");
            setStatus("Начните воспроизведение с анализом.");
          }}
        />
      </Field>
      <div className="flex flex-wrap gap-2">
        <Button disabled={!file || running} onClick={start}>
          Начать просмотр и анализ
        </Button>
        <Button variant="outline" disabled={!running} onClick={() => stop()}>
          Остановить анализ
        </Button>
      </div>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      <p role="status" className="text-sm">
        {buffering
          ? "Видео приостановлено, пока анализ обрабатывает очередь кадров. Воспроизведение продолжится автоматически."
          : status}
      </p>
      {file && (
        <div className="grid items-start gap-6 lg:grid-cols-2">
          <div className="min-w-0 space-y-3">
            <video
              ref={video}
              controls
              playsInline
              preload="metadata"
              aria-label="Видео с анализом"
              className="max-h-128 w-full rounded-lg bg-foreground"
              onTimeUpdate={(event) => setPlaybackTime(event.currentTarget.currentTime)}
              onPlay={() => {
                if (autoPaused.current) video.current?.pause();
              }}
              onSeeking={() => {
                if (session.current && !restarting.current)
                  stop(
                    "Анализ остановлен после перемотки. Запустите его снова с выбранного момента.",
                  );
              }}
              onSeeked={() => {
                restarting.current = false;
              }}
              onEnded={() => {
                if (session.current) {
                  setStatus("Обработка последних кадров…");
                  finalSample.current?.();
                  session.current?.finish();
                }
              }}
              onError={() => {
                stop();
                setError("Браузер не смог открыть файл. Выберите MP4 с видеокодеком H.264.");
              }}
            />
            <dl className="flex flex-wrap gap-x-6 gap-y-2 text-sm tabular-nums">
              <div>
                <dt className="text-muted-foreground">Воспроизведение</dt>
                <dd>{liveTimecode(playbackTime)}</dd>
              </div>
              <div>
                <dt className="text-muted-foreground">Проверено до</dt>
                <dd>{analyzedThrough === null ? "Ожидание" : liveTimecode(analyzedThrough)}</dd>
              </div>
              {analyzedThrough !== null && (
                <div>
                  <dt className="text-muted-foreground">Отставание анализа</dt>
                  <dd>{Math.max(0, playbackTime - analyzedThrough).toFixed(1)} с</dd>
                </div>
              )}
            </dl>
            {snapshot && (
              <figure className="space-y-2 rounded-lg border p-3">
                <figcaption className="text-sm">
                  Обработанный кадр: {liveTimecode(snapshot.timestamp_sec)} · Людей:{" "}
                  {snapshot.people_count} · Предметов: {snapshot.objects_count}
                </figcaption>
                <img
                  src={`data:image/jpeg;base64,${snapshot.image_base64}`}
                  alt={`Распознанные люди и предметы, ${liveTimecode(snapshot.timestamp_sec)}`}
                  className="max-h-80 w-full object-contain"
                />
              </figure>
            )}
          </div>
          <section aria-label="События при воспроизведении" className="min-w-0 space-y-3">
            <h2 className="text-lg font-semibold">Найденные события: {findings.length}</h2>
            <p className="text-sm text-muted-foreground">Результаты требуют проверки человеком.</p>
            <p aria-live="polite" aria-atomic="true" className="sr-only">
              Найдено событий: {findings.length}
            </p>
            {!findings.length && (
              <p className="text-sm text-muted-foreground">
                {busy || analyzedThrough === null
                  ? "Ожидание результатов проверки."
                  : "В проверенных окнах событий не найдено."}
              </p>
            )}
            <ol className="max-h-160 space-y-4 overflow-y-auto">
              {findings.map((finding) => (
                <li
                  key={`${finding.type}-${finding.timecode_sec}`}
                  className="space-y-2 rounded-lg border p-3"
                >
                  <h3 className="text-sm font-semibold">
                    {liveTimecode(finding.timecode_sec)} · {finding.type}
                  </h3>
                  <p className="text-sm whitespace-pre-wrap">{finding.description}</p>
                  <img
                    src={`data:image/jpeg;base64,${finding.evidence_frames[0]!.image_base64}`}
                    alt={`Кадр события: ${liveTimecode(finding.evidence_frames[0]!.timestamp_sec)}`}
                    loading="lazy"
                    className="max-h-48 w-full object-contain"
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      stop(
                        "Анализ остановлен для просмотра события. Запустите его снова, чтобы продолжить.",
                      );
                      if (video.current) video.current.currentTime = finding.timecode_sec;
                    }}
                  >
                    Перейти к событию
                  </Button>
                </li>
              ))}
            </ol>
          </section>
        </div>
      )}
    </section>
  );
}
