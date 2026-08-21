import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AgentIQ — Urban Media Engine",
  description:
    "Transit media campaign recommendation system. Natural-language brief in, " +
    "explainable sales-ready media package out.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="h-screen overflow-hidden bg-zinc-50/50 font-sans text-zinc-700 antialiased">
        {children}
      </body>
    </html>
  );
}
