"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { 
  LayoutDashboard, 
  GitCompare, 
  LineChart, 
  FileDown, 
  Sparkles, 
  Layers, 
  Cpu, 
  Search, 
  Bell, 
  ShieldCheck, 
  ExternalLink,
  ChevronRight,
  Database,
  Sliders,
  CheckCircle2,
  X,
  Play
} from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";

const navItems = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard, badge: "Live" },
  { href: "/explorer", label: "Match Explorer", icon: GitCompare, badge: "28 Features" },
  { href: "/benchmark", label: "F₀.₅ Lab", icon: LineChart, badge: "β=0.5" },
  { href: "/export", label: "Submission Center", icon: FileDown, badge: "Ready" },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [searchOpen, setSearchOpen] = useState(false);
  const [specOpen, setSpecOpen] = useState(false);
  const [backendLatency, setBackendLatency] = useState<number | null>(24);

  // Keyboard shortcut for Cmd+K / Ctrl+K
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setSearchOpen((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  return (
    <div className="flex min-h-screen bg-[var(--bg-deep)] text-[var(--text-main)] transition-colors duration-200">
      {/* ── Left Sidebar ────────────────────────────────────────────────────────── */}
      <aside className="w-64 bg-white/95 dark:bg-[#090d16]/95 backdrop-blur-xl border-r border-slate-200 dark:border-white/[0.08] flex flex-col fixed h-full z-40 transition-colors duration-200">
        {/* Brand Header */}
        <div className="p-5 border-b border-slate-200 dark:border-white/[0.08]">
          <Link href="/" className="flex items-center gap-3 group">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-violet-600 via-indigo-600 to-cyan-500 p-[1px] shadow-lg shadow-violet-500/20 group-hover:shadow-violet-500/35 transition-all">
              <div className="w-full h-full bg-white dark:bg-[#090d16] rounded-[11px] flex items-center justify-center">
                <Sparkles className="w-4 h-4 text-violet-600 dark:text-violet-400 group-hover:scale-110 transition-transform" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="font-bold text-sm tracking-tight text-slate-900 dark:text-white">RESOLVE.AI</span>
                <span className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded bg-violet-50 dark:bg-violet-500/20 text-violet-700 dark:text-violet-300 border border-violet-200 dark:border-violet-500/30">
                  ML 2026
                </span>
              </div>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 font-medium">Entity Resolution Platform</p>
            </div>
          </Link>
        </div>

        {/* Navigation Links */}
        <div className="px-3 py-4">
          <div className="px-3 mb-2 text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">
            Platform Modules
          </div>
          <nav className="space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-medium transition-all duration-200 group ${
                    isActive
                      ? "bg-violet-50 dark:bg-violet-600/15 text-violet-700 dark:text-white border border-violet-200 dark:border-violet-500/30 shadow-sm"
                      : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-white/[0.04]"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <Icon
                      className={`w-4 h-4 transition-colors ${
                        isActive ? "text-violet-600 dark:text-violet-400" : "text-slate-400 group-hover:text-slate-600 dark:group-hover:text-slate-200"
                      }`}
                    />
                    <span>{item.label}</span>
                  </div>
                  {item.badge && (
                    <span
                      className={`text-[10px] px-1.5 py-0.5 rounded-md font-semibold ${
                        isActive
                          ? "bg-violet-100 dark:bg-violet-500/30 text-violet-800 dark:text-violet-200"
                          : "bg-slate-100 dark:bg-white/[0.05] text-slate-500 dark:text-slate-400 group-hover:text-slate-700 dark:group-hover:text-slate-300"
                      }`}
                    >
                      {item.badge}
                    </span>
                  )}
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Middle Quick Specs */}
        <div className="px-3 py-2 flex-1">
          <div className="px-3 mb-2 text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">
            Pipeline Architecture
          </div>
          <div className="p-3 rounded-xl bg-slate-50 dark:bg-white/[0.02] border border-slate-200 dark:border-white/[0.06] space-y-2 text-[11px]">
            <div className="flex items-center justify-between text-slate-600 dark:text-slate-400">
              <span className="flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-cyan-600 dark:text-cyan-400" /> Model
              </span>
              <span className="font-semibold text-slate-900 dark:text-slate-200">LightGBM (8B-Safe)</span>
            </div>
            <div className="flex items-center justify-between text-slate-600 dark:text-slate-400">
              <span className="flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5 text-violet-600 dark:text-violet-400" /> Blocking
              </span>
              <span className="font-semibold text-slate-900 dark:text-slate-200">MinHash + Dense ANN</span>
            </div>
            <div className="flex items-center justify-between text-slate-600 dark:text-slate-400">
              <span className="flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" /> Metric
              </span>
              <span className="font-semibold text-slate-900 dark:text-slate-200">Macro F₀.₅ (β=0.5)</span>
            </div>
          </div>
        </div>

        {/* Sidebar Footer: System Status */}
        <div className="p-4 border-t border-slate-200 dark:border-white/[0.08] space-y-3">
          <div className="flex items-center justify-between p-2.5 rounded-xl bg-emerald-50 dark:bg-emerald-500/[0.08] border border-emerald-200 dark:border-emerald-500/20">
            <div className="flex items-center gap-2">
              <div className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </div>
              <span className="text-xs font-semibold text-emerald-800 dark:text-emerald-300">FastAPI Pipeline</span>
            </div>
            <span className="text-[10px] font-mono text-emerald-700 dark:text-emerald-400/80">{backendLatency}ms</span>
          </div>

          <button
            onClick={() => setSpecOpen(true)}
            className="w-full flex items-center justify-center gap-2 py-2 px-3 rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-white/[0.04] dark:hover:bg-white/[0.08] text-xs font-medium text-slate-700 dark:text-slate-300 transition-colors border border-slate-200 dark:border-white/[0.06]"
          >
            <span>🎨</span> View Figma Blueprint
          </button>
        </div>
      </aside>

      {/* ── Main Layout Column ─────────────────────────────────────────────────── */}
      <div className="flex-1 ml-64 flex flex-col min-w-0">
        {/* Top Header Bar */}
        <header className="h-16 border-b border-slate-200 dark:border-white/[0.08] bg-white/80 dark:bg-[#06080e]/80 backdrop-blur-md sticky top-0 z-30 px-8 flex items-center justify-between gap-4 transition-colors duration-200">
          {/* Breadcrumb / Title */}
          <div className="flex items-center gap-3">
            <span className="text-xs font-medium text-slate-500 dark:text-slate-400">Amazon ML Challenge 2026</span>
            <ChevronRight className="w-3.5 h-3.5 text-slate-400 dark:text-slate-600" />
            <span className="text-xs font-semibold text-slate-800 dark:text-slate-200 capitalize">
              {pathname === "/" ? "Executive Dashboard" : pathname.replace("/", "")}
            </span>
          </div>

          {/* Center Search Bar */}
          <div className="flex-1 max-w-md mx-4">
            <button
              onClick={() => setSearchOpen(true)}
              className="w-full bg-slate-100 hover:bg-slate-200/70 dark:bg-[#0c111d] dark:hover:bg-[#131a2c] border border-slate-200 dark:border-white/[0.08] text-slate-500 dark:text-slate-400 px-3.5 py-1.5 rounded-xl text-xs flex items-center justify-between transition-colors shadow-inner"
            >
              <span className="flex items-center gap-2">
                <Search className="w-3.5 h-3.5 text-slate-400" />
                <span>Search entities, features, or metrics...</span>
              </span>
              <kbd className="bg-white dark:bg-white/[0.06] text-slate-500 dark:text-slate-400 px-1.5 py-0.5 rounded text-[10px] font-mono border border-slate-200 dark:border-white/[0.08] shadow-xs">
                ⌘K
              </kbd>
            </button>
          </div>

          {/* Right Action Icons & Theme Toggle */}
          <div className="flex items-center gap-3">
            {/* Theme Toggle (Day / Night) */}
            <ThemeToggle />

            <Link
              href="/export"
              className="btn-figma-secondary text-xs py-1.5 px-3 flex items-center gap-1.5"
            >
              <FileDown className="w-3.5 h-3.5 text-violet-600 dark:text-violet-400" />
              <span>Export</span>
            </Link>

            <Link
              href="/?run=new"
              className="btn-figma-primary text-xs py-1.5 px-3.5 flex items-center gap-1.5"
            >
              <Play className="w-3 h-3 fill-current" />
              <span>Run Pipeline</span>
            </Link>

            <div className="h-4 w-[1px] bg-slate-300 dark:bg-white/[0.1] mx-1" />

            <div className="flex items-center gap-2 pl-1">
              <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-violet-600 to-indigo-500 p-[1px] shadow-sm">
                <div className="w-full h-full bg-white dark:bg-[#090d16] rounded-full flex items-center justify-center font-bold text-xs text-violet-600 dark:text-violet-300">
                  ML
                </div>
              </div>
              <div className="hidden lg:block text-left text-xs">
                <div className="font-semibold text-slate-800 dark:text-slate-200 leading-tight">Resolve Team</div>
                <div className="text-[10px] text-slate-500 dark:text-slate-400 leading-tight">Amazon ML 2026</div>
              </div>
            </div>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 p-8 pb-16">{children}</main>
      </div>

      {/* ── Quick Search Modal (Cmd+K) ────────────────────────────────────────── */}
      {searchOpen && (
        <div className="fixed inset-0 bg-black/50 dark:bg-black/70 backdrop-blur-sm z-50 flex items-start justify-center pt-24 p-4 animate-in fade-in duration-200">
          <div className="bg-white dark:bg-[#0c111d] border border-slate-200 dark:border-white/[0.12] rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden">
            <div className="p-4 border-b border-slate-200 dark:border-white/[0.08] flex items-center gap-3">
              <Search className="w-5 h-5 text-violet-600 dark:text-violet-400" />
              <input
                autoFocus
                placeholder="Type to search entities (e.g. S1-925783039), modules, or benchmarks..."
                className="bg-transparent text-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 w-full focus:outline-none"
                onKeyDown={(e) => e.key === "Escape" && setSearchOpen(false)}
              />
              <button
                onClick={() => setSearchOpen(false)}
                className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 p-1"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="p-3 text-xs space-y-1">
              <div className="text-[10px] font-semibold text-slate-400 dark:text-slate-500 px-3 py-1 uppercase">Quick Navigation</div>
              <Link
                href="/"
                onClick={() => setSearchOpen(false)}
                className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-slate-100 dark:hover:bg-white/[0.05] text-slate-700 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white"
              >
                <LayoutDashboard className="w-4 h-4 text-violet-600 dark:text-violet-400" />
                <span>Executive Dashboard & Pipeline Cockpit</span>
              </Link>
              <Link
                href="/explorer"
                onClick={() => setSearchOpen(false)}
                className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-slate-100 dark:hover:bg-white/[0.05] text-slate-700 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white"
              >
                <GitCompare className="w-4 h-4 text-cyan-600 dark:text-cyan-400" />
                <span>Entity Match Explorer & Pairwise 28-Feature Inspector</span>
              </Link>
              <Link
                href="/benchmark"
                onClick={() => setSearchOpen(false)}
                className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-slate-100 dark:hover:bg-white/[0.05] text-slate-700 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white"
              >
                <LineChart className="w-4 h-4 text-amber-600 dark:text-amber-400" />
                <span>F₀.₅ Optimization Lab & Threshold Calibration Curve</span>
              </Link>
              <Link
                href="/export"
                onClick={() => setSearchOpen(false)}
                className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-slate-100 dark:hover:bg-white/[0.05] text-slate-700 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white"
              >
                <FileDown className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                <span>Submission Validator & Output Packaging</span>
              </Link>
            </div>
          </div>
        </div>
      )}

      {/* ── Figma Blueprint Spec Drawer Modal ─────────────────────────────────── */}
      {specOpen && (
        <div className="fixed inset-0 bg-black/60 dark:bg-black/80 backdrop-blur-md z-50 flex items-center justify-center p-6 animate-in fade-in duration-200">
          <div className="bg-white dark:bg-[#0c111d] border border-slate-200 dark:border-violet-500/30 rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
            <div className="p-4 px-6 border-b border-slate-200 dark:border-white/[0.08] flex items-center justify-between bg-violet-50 dark:bg-violet-950/20">
              <div className="flex items-center gap-3">
                <span className="text-xl">🎨</span>
                <div>
                  <h3 className="text-sm font-bold text-slate-900 dark:text-white">Figma Design System & Architecture Blueprint</h3>
                  <p className="text-xs text-slate-500 dark:text-slate-400">Enterprise AI Business Entity Resolution (Amazon ML 2026)</p>
                </div>
              </div>
              <button
                onClick={() => setSpecOpen(false)}
                className="text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-white p-1 rounded-lg hover:bg-slate-100 dark:hover:bg-white/[0.05]"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-6 overflow-y-auto space-y-6 text-sm text-slate-600 dark:text-slate-300">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 rounded-xl bg-slate-50 dark:bg-white/[0.03] border border-slate-200 dark:border-white/[0.06]">
                  <div className="text-xs font-semibold text-violet-600 dark:text-violet-400 uppercase tracking-wider mb-1">Color Tokens</div>
                  <div className="text-xs space-y-1.5 text-slate-600 dark:text-slate-400 font-mono">
                    <div className="flex items-center gap-2"><div className="w-3 h-3 rounded bg-[#f8fafc] border border-slate-300" /> Day Deep (#F8FAFC)</div>
                    <div className="flex items-center gap-2"><div className="w-3 h-3 rounded bg-white border border-slate-300" /> Day Surface (#FFFFFF)</div>
                    <div className="flex items-center gap-2"><div className="w-3 h-3 rounded bg-[#7c3aed]" /> Primary Violet (#7C3AED)</div>
                    <div className="flex items-center gap-2"><div className="w-3 h-3 rounded bg-[#0284c7]" /> Cyan Accent (#0284C7)</div>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-slate-50 dark:bg-white/[0.03] border border-slate-200 dark:border-white/[0.06]">
                  <div className="text-xs font-semibold text-cyan-600 dark:text-cyan-400 uppercase tracking-wider mb-1">Evaluation Metric</div>
                  <div className="text-xs text-slate-700 dark:text-slate-300 space-y-1">
                    <div className="font-semibold text-slate-900 dark:text-white">Macro F₀.₅ Optimization</div>
                    <p className="text-[11px] text-slate-500 dark:text-slate-400">Precision counts 2× more than Recall. Singletons predict empty for 1.0 score.</p>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-slate-50 dark:bg-white/[0.03] border border-slate-200 dark:border-white/[0.06]">
                  <div className="text-xs font-semibold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider mb-1">ML Pipeline Stages</div>
                  <div className="text-[11px] text-slate-700 dark:text-slate-300 space-y-1">
                    <div>1. Open-Set NFKD Normalizer</div>
                    <div>2. MinHash LSH + Dense ANN</div>
                    <div>3. 28 Pairwise Similarity Features</div>
                    <div>4. LightGBM Calibrated Classifier</div>
                  </div>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-100 dark:bg-slate-900/60 border border-slate-200 dark:border-violet-500/20">
                <h4 className="font-semibold text-xs text-slate-800 dark:text-slate-200 uppercase tracking-wider mb-2">Dual Theme Architecture</h4>
                <ul className="text-xs text-slate-600 dark:text-slate-400 space-y-1 list-disc list-inside">
                  <li>Day Theme (Default): Crisp, clean SaaS aesthetic with high contrast and readable data tables.</li>
                  <li>Night Theme: Deep obsidian dark mode with glowing borders and ambient lighting.</li>
                  <li>Quick Theme Switcher in the top header persists your choice across sessions.</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
