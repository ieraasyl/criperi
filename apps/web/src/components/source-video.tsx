import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { HugeiconsIcon } from "@hugeicons/react";
import { PlayIcon } from "@hugeicons/core-free-icons";
import { timecode } from "@/lib/review";

export function SourceVideo({
  file,
  onFile,
  seconds,
}: {
  file: File | null;
  onFile: (file: File) => void;
  seconds: number;
}) {
  const [open, setOpen] = useState(false);
  const [error, setError] = useState("");
  const video = useRef<HTMLVideoElement>(null);
  const chooser = useRef<HTMLInputElement>(null);
  useEffect(() => {
    const player = video.current;
    if (!file || !player || !open) return;
    const objectUrl = URL.createObjectURL(file);
    player.src = objectUrl;
    return () => {
      player.removeAttribute("src");
      URL.revokeObjectURL(objectUrl);
    };
  }, [file, open]);
  function seek() {
    const player = video.current;
    if (!player || !Number.isFinite(player.duration)) return;
    if (seconds > player.duration) {
      setError(
        "В выбранном файле нет этого момента. Выберите исходное видео с тем же хронометражем.",
      );
      return;
    }
    setError("");
    player.currentTime = Math.min(seconds, Math.max(0, player.duration - 0.01));
    void player.play().catch(() => {});
  }
  return (
    <div className="space-y-2 border-t pt-3">
      {file && (
        <Button
          variant="outline"
          onClick={() => {
            setOpen(true);
            setError("");
            seek();
          }}
        >
          <HugeiconsIcon icon={PlayIcon} strokeWidth={2} /> Смотреть с{" "}
          {timecode(seconds).replace(/\.0$/, "")}
        </Button>
      )}
      {(!file || error) && (
        <Button variant="outline" onClick={() => chooser.current?.click()}>
          {!file && <HugeiconsIcon icon={PlayIcon} strokeWidth={2} />}
          {file ? "Открыть совместимую копию исходного видео" : "Открыть исходное видео"}
        </Button>
      )}
      {!file && (
        <p className="text-xs text-muted-foreground">
          Для воспроизведения нужен исходный файл. Он не отправляется на сервер.
        </p>
      )}
      <input
        ref={chooser}
        id="source-video"
        aria-label="Исходное видео"
        type="file"
        hidden
        accept="video/*,.mov,.mp4"
        onChange={(event) => {
          const selected = event.target.files?.[0];
          if (selected) {
            onFile(selected);
            setOpen(true);
            setError("");
          }
        }}
      />
      {open && file && (
        <video
          ref={video}
          controls
          preload="metadata"
          className="max-h-96 w-full bg-black"
          onLoadedMetadata={seek}
          onError={() =>
            setError(
              "Браузер не смог воспроизвести файл. Выберите копию исходного видео в MP4 с тем же хронометражем.",
            )
          }
        />
      )}
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
    </div>
  );
}
