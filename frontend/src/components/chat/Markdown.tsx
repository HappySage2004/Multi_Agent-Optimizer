"use client";

/**
 * Renders the agent's answer, which is markdown.
 *
 * The Master Agent writes a structured answer — headline, per-section reasoning, bullet
 * lists, occasionally a table — so rendering it as preformatted text loses the structure
 * the prompt asks for.
 *
 * Styled with an explicit element map rather than a typography plugin: the chat column
 * runs at `text-xs` on a zinc palette, and a prose preset would fight it on every rule.
 *
 * Raw HTML is not enabled. `react-markdown` ignores HTML unless `rehype-raw` is added, so
 * model-authored text cannot inject markup — keep it that way.
 */

import type { ComponentPropsWithoutRef } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

type Props<T extends keyof HTMLElementTagNameMap> = ComponentPropsWithoutRef<T>;

/**
 * The agent's answer sections are semantically headings, but they sit inside a chat
 * bubble — so every level renders at a similar weight-led scale rather than the browser's
 * document scale, which would tower over the surrounding text.
 */
const components: Components = {
  h1: ({ children }: Props<"h1">) => (
    <h3 className="mt-4 mb-1.5 text-[13px] font-bold text-zinc-900 first:mt-0">{children}</h3>
  ),
  h2: ({ children }: Props<"h2">) => (
    <h4 className="mt-4 mb-1.5 text-xs font-bold text-zinc-900 first:mt-0">{children}</h4>
  ),
  h3: ({ children }: Props<"h3">) => (
    <h5 className="mt-3 mb-1 text-xs font-bold text-zinc-800 first:mt-0">{children}</h5>
  ),
  h4: ({ children }: Props<"h4">) => (
    <h6 className="mt-3 mb-1 text-[11px] font-bold tracking-wide text-zinc-700 uppercase first:mt-0">
      {children}
    </h6>
  ),

  p: ({ children }: Props<"p">) => (
    <p className="my-1.5 leading-relaxed text-zinc-600 first:mt-0 last:mb-0">{children}</p>
  ),

  ul: ({ children }: Props<"ul">) => (
    <ul className="my-1.5 list-disc space-y-1 pl-4 text-zinc-600 first:mt-0 last:mb-0">
      {children}
    </ul>
  ),
  ol: ({ children }: Props<"ol">) => (
    <ol className="my-1.5 list-decimal space-y-1 pl-4 text-zinc-600 first:mt-0 last:mb-0">
      {children}
    </ol>
  ),
  li: ({ children }: Props<"li">) => <li className="leading-relaxed">{children}</li>,

  strong: ({ children }: Props<"strong">) => (
    <strong className="font-semibold text-zinc-900">{children}</strong>
  ),
  em: ({ children }: Props<"em">) => <em className="italic">{children}</em>,

  // Screen IDs, reason codes and tool names all arrive as inline code.
  code: ({ children }: Props<"code">) => (
    <code className="rounded border border-zinc-200/70 bg-zinc-50 px-1 py-px font-mono text-[11px] text-zinc-700">
      {children}
    </code>
  ),
  pre: ({ children }: Props<"pre">) => (
    <pre className="my-2 overflow-x-auto rounded-lg border border-zinc-200/60 bg-zinc-50 p-3 font-mono text-[11px] leading-relaxed text-zinc-700">
      {children}
    </pre>
  ),

  a: ({ children, href }: Props<"a">) => (
    <a
      href={href}
      target="_blank"
      rel="noreferrer noopener"
      className="font-medium text-violet-950 underline decoration-violet-300 underline-offset-2 hover:decoration-violet-950"
    >
      {children}
    </a>
  ),

  blockquote: ({ children }: Props<"blockquote">) => (
    <blockquote className="my-2 border-l-2 border-zinc-200 pl-3 text-zinc-500 italic">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="my-3 border-zinc-200/70" />,

  // GFM tables — the agent uses these for per-screen and per-block breakdowns.
  table: ({ children }: Props<"table">) => (
    <div className="my-2 overflow-x-auto rounded-lg border border-zinc-200/60">
      <table className="w-full border-collapse text-[11px]">{children}</table>
    </div>
  ),
  thead: ({ children }: Props<"thead">) => <thead className="bg-zinc-50">{children}</thead>,
  th: ({ children }: Props<"th">) => (
    <th className="border-b border-zinc-200/60 px-2.5 py-1.5 text-left font-semibold text-zinc-700">
      {children}
    </th>
  ),
  td: ({ children }: Props<"td">) => (
    <td className="border-b border-zinc-100 px-2.5 py-1.5 text-zinc-600">{children}</td>
  ),
};

export function Markdown({ children }: { children: string }) {
  return (
    <div className="text-xs leading-relaxed">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {children}
      </ReactMarkdown>
    </div>
  );
}
