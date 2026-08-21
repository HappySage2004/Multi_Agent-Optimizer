"use client";

/**
 * The 3-column split from UI.md §2: fixed-width sidebar, flex-fill centre, fixed-width
 * inspector, with a draggable handle between each.
 *
 * Widths are clamped here rather than by CSS min/max so the drag stops at the bound
 * instead of the pointer drifting out of sync with the panel edge.
 */

import { useCallback, useEffect, useRef, useState } from "react";

const LEFT = { initial: 240, min: 190, max: 360 };
const RIGHT = { initial: 420, min: 300, max: 600 };
/** Keeps the centre column from collapsing under its own min-width. */
const CENTER_MIN = 380;

type Edge = "left" | "right";

export function ResizableLayout({
  sidebar,
  center,
  inspector,
}: {
  sidebar: React.ReactNode;
  center: React.ReactNode;
  inspector: React.ReactNode;
}) {
  const [leftWidth, setLeftWidth] = useState(LEFT.initial);
  const [rightWidth, setRightWidth] = useState(RIGHT.initial);
  const [dragging, setDragging] = useState<Edge | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  // Read in the move handler, which must not be re-created on every pixel.
  const widthsRef = useRef({ left: LEFT.initial, right: RIGHT.initial });
  widthsRef.current = { left: leftWidth, right: rightWidth };

  const resize = useCallback((edge: Edge, clientX: number) => {
    const container = containerRef.current;
    if (!container) return;

    const bounds = container.getBoundingClientRect();
    const { left, right } = widthsRef.current;

    if (edge === "left") {
      const available = bounds.width - right - CENTER_MIN;
      const next = clientX - bounds.left;
      setLeftWidth(clamp(next, LEFT.min, Math.min(LEFT.max, available)));
    } else {
      const available = bounds.width - left - CENTER_MIN;
      const next = bounds.right - clientX;
      setRightWidth(clamp(next, RIGHT.min, Math.min(RIGHT.max, available)));
    }
  }, []);

  useEffect(() => {
    if (!dragging) return;

    const onMove = (event: PointerEvent) => resize(dragging, event.clientX);
    const onUp = () => setDragging(null);

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    // Suppress the text-selection cursor for the whole drag, not just over the handle.
    document.body.dataset.resizing = "true";

    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      delete document.body.dataset.resizing;
    };
  }, [dragging, resize]);

  /** Keyboard resizing, so the handles are not mouse-only. */
  const onHandleKeyDown = useCallback((edge: Edge, event: React.KeyboardEvent) => {
    const step = event.shiftKey ? 32 : 8;
    const delta =
      event.key === "ArrowLeft" ? -step : event.key === "ArrowRight" ? step : null;
    if (delta === null) return;
    event.preventDefault();

    if (edge === "left") {
      setLeftWidth((w) => clamp(w + delta, LEFT.min, LEFT.max));
    } else {
      setRightWidth((w) => clamp(w - delta, RIGHT.min, RIGHT.max));
    }
  }, []);

  return (
    <div ref={containerRef} className="flex h-screen overflow-hidden select-none">
      <div style={{ width: leftWidth }} className="shrink-0">
        {sidebar}
      </div>

      <Handle
        edge="left"
        width={leftWidth}
        dragging={dragging === "left"}
        onStart={() => setDragging("left")}
        onKeyDown={onHandleKeyDown}
      />

      <div className="flex min-w-0 flex-1">{center}</div>

      <Handle
        edge="right"
        width={rightWidth}
        dragging={dragging === "right"}
        onStart={() => setDragging("right")}
        onKeyDown={onHandleKeyDown}
      />

      <div style={{ width: rightWidth }} className="shrink-0">
        {inspector}
      </div>
    </div>
  );
}

function Handle({
  edge,
  width,
  dragging,
  onStart,
  onKeyDown,
}: {
  edge: Edge;
  width: number;
  dragging: boolean;
  onStart: () => void;
  onKeyDown: (edge: Edge, event: React.KeyboardEvent) => void;
}) {
  const label = edge === "left" ? "Resize sessions panel" : "Resize inspector panel";
  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      aria-valuenow={Math.round(width)}
      tabIndex={0}
      data-dragging={dragging}
      className="resizer bg-zinc-200/60 focus:bg-violet-950 focus:outline-none"
      onPointerDown={(event) => {
        event.preventDefault();
        onStart();
      }}
      onKeyDown={(event) => onKeyDown(edge, event)}
    />
  );
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), Math.max(min, max));
}
