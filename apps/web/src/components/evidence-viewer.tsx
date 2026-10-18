import { useEffect, useRef, useState, type CSSProperties } from "react";
import { closestFrame, timecode, type Finding } from "@/lib/review";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogTrigger,
  DialogClose,
} from "@/components/ui/dialog";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  ArrowLeft01Icon,
  ArrowRight01Icon,
  PlusSignIcon,
  MinusSignIcon,
} from "@hugeicons/core-free-icons";

export function EvidenceViewer({
  event,
  selected,
  onSelect,
  title,
}: {
  event: Finding;
  selected: number;
  onSelect: (index: number) => void;
  title: string;
}) {
  const [open, setOpen] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number; left: number; top: number } | null>(null);
  const frames = event.evidence_frames;
  const current = Math.max(0, Math.min(selected, frames.length - 1));
  const frame = frames[current]!;
  const moment = closestFrame(event);
  function select(index: number) {
    onSelect(Math.max(0, Math.min(frames.length - 1, index)));
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }
  useEffect(() => {
    const keydown = (key: KeyboardEvent) => {
      if (key.altKey || key.ctrlKey || key.metaKey) return;
      const target = key.target;
      if (
        target instanceof HTMLElement &&
        target.closest(
          "input,textarea,select,video,[contenteditable=true],[role=combobox],[role=listbox],[role=option],[role=dialog],[role=alertdialog]",
        )
      )
        return;
      if (key.key === "ArrowLeft" || key.key === "ArrowRight") {
        key.preventDefault();
        select(current + (key.key === "ArrowRight" ? 1 : -1));
      }
    };
    window.addEventListener("keydown", keydown);
    return () => window.removeEventListener("keydown", keydown);
  });
  function thumbnails(modal = false) {
    return (
      <div
        role="group"
        aria-label={modal ? "Выбор кадра в просмотре" : "Выбор кадра"}
        className="flex shrink-0 gap-2 overflow-x-auto pb-1"
      >
        {frames.map((item, index) => (
          <Button
            key={index}
            variant="thumbnail"
            size="thumbnail"
            aria-label={`Выбрать кадр ${timecode(item.timestamp_sec)}`}
            aria-pressed={index === current}
            onClick={() => select(index)}
          >
            <img
              className={modal ? "h-28 w-full object-contain" : "h-36 w-full object-contain"}
              src={`data:image/jpeg;base64,${item.image_base64}`}
              alt=""
            />
            <span className="block tabular-nums">{timecode(item.timestamp_sec)}</span>
            {index === moment && <span className="block">● момент события</span>}
          </Button>
        ))}
      </div>
    );
  }
  function arrows() {
    return (
      <>
        <Button
          variant="outline"
          size="icon"
          aria-label="Предыдущий кадр"
          disabled={current === 0}
          onClick={() => select(current - 1)}
        >
          <HugeiconsIcon icon={ArrowLeft01Icon} strokeWidth={2} />
        </Button>
        <Button
          variant="outline"
          size="icon"
          aria-label="Следующий кадр"
          disabled={current === frames.length - 1}
          onClick={() => select(current + 1)}
        >
          <HugeiconsIcon icon={ArrowRight01Icon} strokeWidth={2} />
        </Button>
      </>
    );
  }
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <section aria-label="Кадры события" className="min-w-0 space-y-3">
        <div className="flex items-center justify-between gap-3 text-sm">
          <h4 className="font-medium">Кадры события</h4>
          <span>
            {current + 1} из {frames.length}
          </span>
        </div>
        <figure className="relative">
          <DialogTrigger
            render={<Button variant="image" size="image" className="cursor-zoom-in" />}
            aria-label={`Открыть кадр ${timecode(frame.timestamp_sec)}`}
            aria-haspopup="dialog"
            onClick={() => {
              setZoom(1);
              setPan({ x: 0, y: 0 });
            }}
          >
            <img
              className="mx-auto block size-auto max-h-96 max-w-full xl:max-h-112"
              src={`data:image/jpeg;base64,${frame.image_base64}`}
              alt={`Кадр события: ${timecode(frame.timestamp_sec)}`}
            />
          </DialogTrigger>
          <div className="absolute inset-x-2 top-1/2 flex -translate-y-1/2 justify-between pointer-events-none [&>button]:pointer-events-auto">
            {arrows()}
          </div>
        </figure>
        {thumbnails()}
        <DialogContent
          variant="frame"
          showCloseButton={false}
          aria-label={`${title}. Кадр ${current + 1} из ${frames.length}`}
          onKeyDown={(key) => {
            if (key.altKey || key.ctrlKey || key.metaKey) return;
            if (key.key === "ArrowLeft" || key.key === "ArrowRight") {
              key.preventDefault();
              key.stopPropagation();
              select(current + (key.key === "ArrowRight" ? 1 : -1));
            }
          }}
        >
          <div
            className="flex h-full min-h-0 flex-col gap-3"
            onClick={(key) => {
              if (key.target === key.currentTarget) setOpen(false);
            }}
          >
            <header className="flex shrink-0 flex-wrap items-center justify-between gap-3">
              <div className="min-w-0">
                <DialogTitle>{title}</DialogTitle>
                <p className="text-sm">
                  {timecode(frame.timestamp_sec)} · {current + 1} из {frames.length}
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  variant="outline"
                  size="icon"
                  aria-label="Уменьшить"
                  disabled={zoom <= 1}
                  onClick={() => {
                    setZoom(Math.max(1, zoom - 0.5));
                    setPan({ x: 0, y: 0 });
                  }}
                >
                  <HugeiconsIcon icon={MinusSignIcon} strokeWidth={2} />
                </Button>
                <Button
                  variant="outline"
                  onClick={() => {
                    setZoom(1);
                    setPan({ x: 0, y: 0 });
                  }}
                >
                  {Math.round(zoom * 100)}%
                </Button>
                <Button
                  variant="outline"
                  size="icon"
                  aria-label="Увеличить"
                  disabled={zoom >= 4}
                  onClick={() => setZoom(Math.min(4, zoom + 0.5))}
                >
                  <HugeiconsIcon icon={PlusSignIcon} strokeWidth={2} />
                </Button>
                <DialogClose render={<Button variant="outline" />}>Закрыть кадр</DialogClose>
              </div>
            </header>
            <div
              className="relative min-h-0 flex-1 overflow-hidden"
              onClick={(key) => {
                if (key.target === key.currentTarget) setOpen(false);
              }}
            >
              <img
                draggable={false}
                className={
                  zoom > 1
                    ? "evidence-zoom size-full cursor-grab touch-none object-contain select-none active:cursor-grabbing"
                    : "evidence-zoom size-full cursor-zoom-in object-contain"
                }
                style={
                  {
                    "--frame-x": `${pan.x}px`,
                    "--frame-y": `${pan.y}px`,
                    "--frame-zoom": zoom,
                  } as CSSProperties
                }
                src={`data:image/jpeg;base64,${frame.image_base64}`}
                alt={`Кадр события: ${timecode(frame.timestamp_sec)}`}
                onClick={(key) => {
                  if (zoom !== 1) return;
                  const image = key.currentTarget;
                  const rect = image.getBoundingClientRect();
                  if (!image.naturalWidth || !image.naturalHeight) return;
                  const scale = Math.min(
                    rect.width / image.naturalWidth,
                    rect.height / image.naturalHeight,
                  );
                  const width = image.naturalWidth * scale,
                    height = image.naturalHeight * scale;
                  if (
                    Math.abs(key.clientX - rect.left - rect.width / 2) > width / 2 ||
                    Math.abs(key.clientY - rect.top - rect.height / 2) > height / 2
                  )
                    setOpen(false);
                  else setZoom(2);
                }}
                onPointerDown={(key) => {
                  if (zoom <= 1) return;
                  key.currentTarget.setPointerCapture(key.pointerId);
                  drag.current = { x: key.clientX, y: key.clientY, left: pan.x, top: pan.y };
                }}
                onPointerMove={(key) => {
                  if (!drag.current) return;
                  const image = key.currentTarget;
                  const scale = Math.min(
                    image.clientWidth / image.naturalWidth,
                    image.clientHeight / image.naturalHeight,
                  );
                  const limitX = Math.max(
                    0,
                    (image.naturalWidth * scale * zoom - image.clientWidth) / 2,
                  );
                  const limitY = Math.max(
                    0,
                    (image.naturalHeight * scale * zoom - image.clientHeight) / 2,
                  );
                  setPan({
                    x: Math.max(
                      -limitX,
                      Math.min(limitX, drag.current.left + key.clientX - drag.current.x),
                    ),
                    y: Math.max(
                      -limitY,
                      Math.min(limitY, drag.current.top + key.clientY - drag.current.y),
                    ),
                  });
                }}
                onPointerUp={() => {
                  drag.current = null;
                }}
                onPointerCancel={() => {
                  drag.current = null;
                }}
              />
              <div className="absolute inset-x-2 top-1/2 flex -translate-y-1/2 justify-between pointer-events-none [&>button]:pointer-events-auto">
                {arrows()}
              </div>
            </div>
            <div className="mx-auto max-w-full">{thumbnails(true)}</div>
          </div>
        </DialogContent>
      </section>
    </Dialog>
  );
}
