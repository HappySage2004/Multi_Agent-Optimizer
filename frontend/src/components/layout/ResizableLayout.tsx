"use client";

/**
 * The 3-column split from UI.md §2: fixed-width sidebar, flex-fill centre, and an
 * inspector that opens at a share of the viewport, with a draggable handle between each.
 *
 * Drag bounds are clamped in JS rather than by CSS min/max so the drag stops at the bound
 * instead of the pointer drifting out of sync with the panel edge.
 */

import { useCallback, useEffect, useRef, useState } from "react";

const LEFT = { initial: 240, min: 190, max: 360 };
/**
 * The inspector opens at a share of the viewport rather than a fixed pixel width — the
 * four stage tabs carry wide tables, and 40% is what they need to read without wrapping.
 *
 * `share` is applied as CSS, not measured on mount: the server and the client then render
 * the identical width (no hydration mismatch, no resize flash), and the panel tracks a
 * window resize until the user drags.
 *
 * The upper drag bound is a share too, not a pixel cap. A pixel cap would quietly clip
 * the default on a wide monitor — the panel would open narrower than `share` and then
 * jump on the first drag — so the two are expressed in the same units and cannot disagree.
 */
const RIGHT = { share: 40, maxShare: 60, min: 300, fallback: 420 };
/** Keeps the centre column from collapsing under its own min-width. */
const CENTER_MIN = 380;

/** The widest the inspector may be dragged, for a container of `containerWidth`. */
function rightMax(containerWidth: number): number {
  return Math.max(RIGHT.min, (containerWidth * RIGHT.maxShare) / 100);
}

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
  /** Null while the inspector is still at its CSS default width; a pixel value once dragged. */
  const [rightWidth, setRightWidth] = useState<number | null>(null);
  const [dragging, setDragging] = useState<Edge | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const inspectorRef = useRef<HTMLDivElement>(null);
  // Read in the move handler, which must not be re-created on every pixel.
  const widthsRef = useRef<{ left: number; right: number | null }>({
    left: LEFT.initial,
    right: null,
  });
  widthsRef.current = { left: leftWidth, right: rightWidth };

  /**
   * The inspector's width as a number. Measured from layout only while it is still at its
   * CSS default — deliberately not read during render, so a streaming run's re-renders do
   * not each force a reflow.
   */
  const measuredRightWidth = useCallback(
    () =>
      widthsRef.current.right ??
      inspectorRef.current?.getBoundingClientRect().width ??
      RIGHT.fallback,
    [],
  );

  const resize = useCallback((edge: Edge, clientX: number) => {
    const container = containerRef.current;
    if (!container) return;

    const bounds = container.getBoundingClientRect();
    const { left } = widthsRef.current;
    const right = measuredRightWidth();

    if (edge === "left") {
      const available = bounds.width - right - CENTER_MIN;
      const next = clientX - bounds.left;
      setLeftWidth(clamp(next, LEFT.min, Math.min(LEFT.max, available)));
    } else {
      const available = bounds.width - left - CENTER_MIN;
      const next = bounds.right - clientX;
      setRightWidth(clamp(next, RIGHT.min, Math.min(rightMax(bounds.width), available)));
    }
  }, [measuredRightWidth]);

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
  const onHandleKeyDown = useCallback(
    (edge: Edge, event: React.KeyboardEvent) => {
      const step = event.shiftKey ? 32 : 8;
      const delta =
        event.key === "ArrowLeft" ? -step : event.key === "ArrowRight" ? step : null;
      if (delta === null) return;
      event.preventDefault();

      if (edge === "left") {
        setLeftWidth((w) => clamp(w + delta, LEFT.min, LEFT.max));
      } else {
        // The first keypress converts the CSS default into a pixel width.
        const containerWidth =
          containerRef.current?.getBoundingClientRect().width ?? window.innerWidth;
        setRightWidth(
          clamp(measuredRightWidth() - delta, RIGHT.min, rightMax(containerWidth)),
        );
      }
    },
    [measuredRightWidth],
  );

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

      {/* A percentage until dragged, a pixel width after. `minWidth` is the floor in both
          cases — flexbox applies it to the used width, so the JS clamp cannot be
          undercut by a narrow viewport. */}
      <div
        ref={inspectorRef}
        style={{ width: rightWidth ?? `${RIGHT.share}%`, minWidth: RIGHT.min }}
        className="shrink-0"
      >
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
  /** Null while the panel is at its CSS default, so there is no pixel value to report. */
  width: number | null;
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
      aria-valuenow={width === null ? undefined : Math.round(width)}
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
