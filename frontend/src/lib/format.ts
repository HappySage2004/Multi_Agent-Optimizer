/** Display formatters. Presentation only — never used to derive a value. */

export function formatCurrency(value: number, fractionDigits = 0): string {
  return value.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  });
}

/** Compact impression/reach counts: 4_250_000 -> "4.25 M". */
export function formatCompact(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(2)} M`;
  if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(1)} K`;
  return value.toFixed(0);
}

export function formatNumber(value: number, fractionDigits = 0): string {
  return value.toLocaleString("en-US", {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  });
}

/** 0.9976 -> "99.8%" */
export function formatPercent(fraction: number, fractionDigits = 1): string {
  return `${(fraction * 100).toFixed(fractionDigits)}%`;
}

/** ISO date -> "01 Oct 2026". Parsed as UTC so the day never shifts by timezone. */
export function formatDate(iso: string): string {
  const date = new Date(`${iso}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });
}

export function formatDateTime(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
}

/** "screen_candidates" -> "Screen Candidates" */
export function titleCase(value: string): string {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .trim();
}
