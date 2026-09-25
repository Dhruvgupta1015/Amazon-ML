import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { AppShell } from "@/components/AppShell";
import { ThemeProvider } from "@/components/ThemeProvider";

const inter = Inter({ 
  subsets: ["latin"], 
  variable: "--font-inter",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "RESOLVE.AI | Amazon ML Challenge 2026 Entity Resolution Platform",
  description: "Enterprise Business Entity Resolution Platform — High-recall multi-stage blocking, 28-feature LightGBM ranking, and optimal Macro F0.5 calibration.",
  keywords: "entity resolution, amazon ml challenge, lightgbm, minhash, deduplication, record linkage",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`light ${inter.variable} ${jetbrainsMono.variable}`}>
      <body className="min-h-screen bg-[var(--bg-deep)] text-[var(--text-main)] font-sans antialiased selection:bg-violet-500/20 selection:text-violet-600 dark:selection:text-violet-300">
        <ThemeProvider>
          <AppShell>{children}</AppShell>
        </ThemeProvider>
      </body>
    </html>
  );
}
