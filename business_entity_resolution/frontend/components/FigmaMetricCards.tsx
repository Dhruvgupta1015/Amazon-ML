"use client";

import React from "react";
import { TrendingUp, TrendingDown, Zap, ShieldCheck, Target, Database } from "lucide-react";

interface FigmaMetricCardsProps {
  totalEntities?: number | null;
  matchRate?: number;
  f05Score?: number;
  falsePositives?: number;
  latencyMs?: number;
}

export function FigmaMetricCards({
  totalEntities = 1732544,
  matchRate = 21.0,
  f05Score = 0.9412,
  falsePositives = 0.4,
  latencyMs = 42,
}: FigmaMetricCardsProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* ── Card 1: Entity Match Rate (Cyan Accent) ────────────────────────── */}
      <div className="glass-panel p-5 glow-cyan relative overflow-hidden group">
        <div className="flex items-start justify-between mb-3">
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-cyan-600 dark:text-cyan-400">
              Target Objective
            </span>
            <div className="text-xs font-semibold text-slate-600 dark:text-slate-300">Entity Match Rate</div>
          </div>
          <div className="w-8 h-8 rounded-xl bg-cyan-50 dark:bg-cyan-500/10 border border-cyan-200 dark:border-cyan-500/30 flex items-center justify-center text-cyan-600 dark:text-cyan-400 shadow-xs">
            <Target className="w-4 h-4" />
          </div>
        </div>

        <div className="flex items-baseline gap-2 mb-2">
          <div className="text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white">
            {matchRate.toFixed(1)}%
          </div>
          <span className="text-xs font-bold text-emerald-600 dark:text-emerald-400 flex items-center gap-0.5">
            <TrendingUp className="w-3.5 h-3.5" /> +2.4%
          </span>
        </div>

        {/* Sparkline & Sub-stat */}
        <div className="flex items-center justify-between pt-2 border-t border-slate-200 dark:border-white/[0.06]">
          <span className="text-[11px] font-mono text-cyan-700 dark:text-cyan-300 font-semibold">
            F₀.₅: {f05Score.toFixed(3)}
          </span>
          <span className="text-[10px] text-slate-500 dark:text-slate-400">Macro Precision/Recall</span>
        </div>

        {/* Mini SVG Sparkline */}
        <div className="mt-2 h-6 w-full">
          <svg viewBox="0 0 100 24" className="w-full h-full stroke-cyan-500 dark:stroke-cyan-400 fill-none" strokeWidth="2">
            <polyline points="0,20 15,16 30,18 45,12 60,14 75,8 90,10 100,4" />
          </svg>
        </div>
      </div>

      {/* ── Card 2: Total Entities (Purple Accent) ────────────────────────── */}
      <div className="glass-panel p-5 glow-purple relative overflow-hidden group">
        <div className="flex items-start justify-between mb-3">
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-violet-600 dark:text-violet-400">
              Knowledge Graph
            </span>
            <div className="text-xs font-semibold text-slate-600 dark:text-slate-300">Total Entities</div>
          </div>
          <div className="w-8 h-8 rounded-xl bg-violet-50 dark:bg-violet-500/10 border border-violet-200 dark:border-violet-500/30 flex items-center justify-center text-violet-600 dark:text-violet-400 shadow-xs">
            <Database className="w-4 h-4" />
          </div>
        </div>

        <div className="flex items-baseline gap-2 mb-2">
          <div className="text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white">
            {totalEntities ? (totalEntities >= 1000000 ? `${(totalEntities / 1000000).toFixed(1)}M` : totalEntities.toLocaleString()) : "1.8M"}
          </div>
          <span className="text-xs font-bold text-violet-600 dark:text-violet-400 flex items-center gap-0.5">
            <TrendingUp className="w-3.5 h-3.5" /> Deduplicated
          </span>
        </div>

        {/* Country Badges */}
        <div className="flex items-center justify-between pt-2 border-t border-slate-200 dark:border-white/[0.06]">
          <div className="flex items-center gap-1">
            <span className="text-[10px] bg-slate-100 dark:bg-white/[0.05] px-1.5 py-0.5 rounded text-slate-700 dark:text-slate-300 font-medium">🇺🇸 US</span>
            <span className="text-[10px] bg-slate-100 dark:bg-white/[0.05] px-1.5 py-0.5 rounded text-slate-700 dark:text-slate-300 font-medium">🇮🇳 IN</span>
            <span className="text-[10px] bg-slate-100 dark:bg-white/[0.05] px-1.5 py-0.5 rounded text-slate-700 dark:text-slate-300 font-medium">🇫🇷 FR</span>
          </div>
          <span className="text-[10px] text-slate-500 dark:text-slate-400">Open-Set</span>
        </div>

        {/* Mini SVG Sparkline */}
        <div className="mt-2 h-6 w-full">
          <svg viewBox="0 0 100 24" className="w-full h-full stroke-violet-500 dark:stroke-violet-400 fill-none" strokeWidth="2">
            <polyline points="0,18 20,17 35,15 50,13 65,11 80,7 100,5" />
          </svg>
        </div>
      </div>

      {/* ── Card 3: False Merges Rate (Amber/Rose Accent) ─────────────────── */}
      <div className="glass-panel p-5 glow-amber relative overflow-hidden group">
        <div className="flex items-start justify-between mb-3">
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-amber-600 dark:text-amber-400">
              Penalty Guardian
            </span>
            <div className="text-xs font-semibold text-slate-600 dark:text-slate-300">False Merges (FP)</div>
          </div>
          <div className="w-8 h-8 rounded-xl bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/30 flex items-center justify-center text-amber-600 dark:text-amber-400 shadow-xs">
            <ShieldCheck className="w-4 h-4" />
          </div>
        </div>

        <div className="flex items-baseline gap-2 mb-2">
          <div className="text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white">
            {falsePositives.toFixed(1)}%
          </div>
          <span className="text-xs font-bold text-emerald-600 dark:text-emerald-400 flex items-center gap-0.5">
            <TrendingDown className="w-3.5 h-3.5" /> -0.8%
          </span>
        </div>

        {/* Penalty details */}
        <div className="flex items-center justify-between pt-2 border-t border-slate-200 dark:border-white/[0.06]">
          <span className="text-[11px] font-mono text-amber-700 dark:text-amber-300 font-semibold">
            2× Precision Weight
          </span>
          <span className="text-[10px] text-slate-500 dark:text-slate-400">Low Risk</span>
        </div>

        {/* Mini SVG Sparkline */}
        <div className="mt-2 h-6 w-full">
          <svg viewBox="0 0 100 24" className="w-full h-full stroke-amber-500 dark:stroke-amber-400 fill-none" strokeWidth="2">
            <polyline points="0,6 20,9 40,12 60,15 80,18 100,21" />
          </svg>
        </div>
      </div>

      {/* ── Card 4: Resolution Latency (Emerald Accent) ──────────────────── */}
      <div className="glass-panel p-5 glow-emerald relative overflow-hidden group">
        <div className="flex items-start justify-between mb-3">
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400">
              Engine Performance
            </span>
            <div className="text-xs font-semibold text-slate-600 dark:text-slate-300">Throughput & Latency</div>
          </div>
          <div className="w-8 h-8 rounded-xl bg-emerald-50 dark:bg-emerald-500/10 border border-emerald-200 dark:border-emerald-500/30 flex items-center justify-center text-emerald-600 dark:text-emerald-400 shadow-xs">
            <Zap className="w-4 h-4" />
          </div>
        </div>

        <div className="flex items-baseline gap-2 mb-2">
          <div className="text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white">
            {latencyMs}ms
          </div>
          <span className="text-xs font-bold text-emerald-600 dark:text-emerald-400 flex items-center gap-0.5">
            <Zap className="w-3.5 h-3.5" /> 12.4k/s
          </span>
        </div>

        {/* Performance details */}
        <div className="flex items-center justify-between pt-2 border-t border-slate-200 dark:border-white/[0.06]">
          <span className="text-[11px] font-mono text-emerald-700 dark:text-emerald-300 font-semibold">
            Sub-second Blocking
          </span>
          <span className="text-[10px] text-slate-500 dark:text-slate-400">C++ LSH Optimized</span>
        </div>

        {/* Mini SVG Sparkline */}
        <div className="mt-2 h-6 w-full">
          <svg viewBox="0 0 100 24" className="w-full h-full stroke-emerald-500 dark:stroke-emerald-400 fill-none" strokeWidth="2">
            <polyline points="0,15 15,10 30,14 45,8 60,11 75,5 90,7 100,3" />
          </svg>
        </div>
      </div>
    </div>
  );
}
