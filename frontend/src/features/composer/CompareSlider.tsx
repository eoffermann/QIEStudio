import { useRef, useState } from "react";

/** Before/after compare slider for Edit results. */
export function CompareSlider({
  before,
  after,
  className,
}: {
  before: string;
  after: string;
  className?: string;
}) {
  const [pos, setPos] = useState(50);
  const ref = useRef<HTMLDivElement>(null);

  const onMove = (clientX: number) => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const p = ((clientX - rect.left) / rect.width) * 100;
    setPos(Math.max(0, Math.min(100, p)));
  };

  return (
    <div
      ref={ref}
      className={"relative select-none overflow-hidden rounded-xl " + (className ?? "")}
      onMouseMove={(e) => e.buttons === 1 && onMove(e.clientX)}
      onTouchMove={(e) => onMove(e.touches[0].clientX)}
      onClick={(e) => onMove(e.clientX)}
    >
      <img src={after} alt="after" className="block w-full" draggable={false} />
      <div
        className="absolute inset-0 overflow-hidden"
        style={{ width: `${pos}%` }}
      >
        <img
          src={before}
          alt="before"
          className="block h-full w-auto max-w-none object-cover"
          style={{ width: ref.current?.clientWidth ?? "100%" }}
          draggable={false}
        />
        <span className="absolute left-2 top-2 rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-white">
          Before
        </span>
      </div>
      <span className="absolute right-2 top-2 rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-white">
        After
      </span>
      <div
        className="absolute inset-y-0 w-0.5 bg-white shadow"
        style={{ left: `${pos}%` }}
      >
        <div className="absolute top-1/2 left-1/2 grid h-7 w-7 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-white text-black shadow">
          ⇆
        </div>
      </div>
    </div>
  );
}
