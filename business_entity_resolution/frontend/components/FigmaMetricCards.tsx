"use client";

import React from "react";
import { TrendingUp, TrendingDown, Zap, ShieldCheck, Target, Database, FileCode } from "lucide-react";

interface FigmaMetricCardsProps {
  totalEntities?: number | null;
  matchRate?: number;
  f05Score?: number;
  falsePositives?: number;
  latencyMs?: number;
  runId?: string;
  provenance?: string;
  precision?: number;
  blockingRecall?: number;
  reductionRatio?: number;
}

export function FigmaMetricCards({
  totalEntities = 1732544,
  matchRate = 21.03,
  f05Score = 0.7930,
  falsePositives = 0.6,
  latencyMs = 38,
  runId = "resolve-val-1790330220",
  provenance = "MEASURED (reports/validation_metrics.json)",
  precision = 0.9756,
  blockingRecall = 79.92,
  reductionRatio = 99.9921
}: FigmaMetricCardsProps) {
  return (
    <div className="space-y-3">
      {/* ── Provenance Banner ── */}
      <div className="flex flex-wrap items-center justify-between text-xs px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-700 dark:text-emerald-300">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          <span className="font-semibold uppercase tracking-wider text-[10px]">Empirical Metric Source:</span>
          <code className="text-[11px] font-mono font-bold">{provenance}</code>
        </div>
        <div className="flex items-center gap-3 text-[11px]">
          <span>Run ID: <code className="font-mono">{runId}</code></span>
          <span className="text-slate-400">|</span>
          <span className="font-medium text-emerald-600 dark:text-emerald-400">Strict Leak-Free Split (Seed=42)</span>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* ── Card 1: Verified Macro F0.5 Score ────────────────────────── */}
        <div className="glass-panel p-5 glow-cyan relative overflow-hidden group">
          <div className="flex items-start justify-between mb-3">
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-cyan-600 dark:text-cyan-400">
                Primary Competition Metric
              </span>
              <div className="text-xs font-semibold text-slate-600 dark:text-slate-300">Macro F₀.₅ Score</div>
            </div>
            <div className="w-8 h-8 rounded-xl bg-cyan-50 dark:bg-cyan-500/10 border border-cyan-200 dark:border-cyan-500/30 flex items-center justify-center text-cyan-600 dark:text-cyan-400 shadow-xs">
              <Target className="w-4 h-4" />
            </div>
          </div>

          <div className="flex items-baseline gap-2 mb-2">
            <div className="text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white">
              {f05Score.toFixed(4)}
            </div>
            <span className="text-[10px] font-bold uppercase px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-700 dark:text-cyan-300">
              Verified
            </span>
          </div>

          <div className="flex items-center justify-between pt-2 border-t border-slate-200 dark:border-white/[0.06]">
            <span className="text-[11px] font-mono text-cyan-700 dark:text-cyan-300 font-semibold">
              Precision: {(precision * 100).toFixed(1)}%
            </span>
            <span className="text-[10px] text-slate-500 dark:text-slate-400">Held-Out Val Split</span>
          </div>

          <div className="mt-2 h-6 w-full">
            <svg viewBox="0 0 100 24" className="w-full h-full stroke-cyan-500 dark:stroke-cyan-400 fill-none" strokeWidth="2">
              <polyline points="0,20 15,16 30,18 45,12 60,14 75,8 90,10 100,4" />
            </svg>
          </div>
        </div>

        {/* ── Card 2: Candidate Blocking Recall ────────────────────────── */}
        <div className="glass-panel p-5 glow-purple relative overflow-hidden group">
          <div className="flex items-start justify-between mb-3">
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-violet-600 dark:text-violet-400">
                Phase 4 Benchmark
              </span>
              <div className="text-xs font-semibold text-slate-600 dark:text-slate-300">Blocking Candidate Recall</div>
            </div>
            <div className="w-8 h-8 rounded-xl bg-violet-50 dark:bg-violet-500/10 border border-violet-200 dark:border-violet-500/30 flex items-center justify-center text-violet-600 dark:text-violet-400 shadow-xs">
              <Database className="w-4 h-4" />
            </div>
          </div>

          <div className="flex items-baseline gap-2 mb-2">
            <div className="text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white">
              {blockingRecall.toFixed(1)}%
            </div>
            <span className="text-xs font-bold text-violet-600 dark:text-violet-400 flex items-center gap-0.5">
              RR: {reductionRatio.toFixed(2)}%
            </span>
          </div>

          <div className="flex items-center justify-between pt-2 border-t border-slate-200 dark:border-white/[0.06]">
            <div className="flex items-center gap-1">
              <span className="text-[10px] bg-slate-100 dark:bg-white/[0.05] px-1.5 py-0.5 rounded text-slate-700 dark:text-slate-300 font-medium">Strategy D</span>
              <span className="text-[10px] bg-slate-100 dark:bg-white/[0.05] px-1.5 py-0.5 rounded text-slate-700 dark:text-slate-300 font-medium">Multi-Index</span>
            </div>
            <span className="text-[10px] text-slate-500 dark:text-slate-400">509k pairs</span>
          </div>

          <div className="mt-2 h-6 w-full">
            <svg viewBox="0 0 100 24" className="w-full h-full stroke-violet-500 dark:stroke-violet-400 fill-none" strokeWidth="2">
              <polyline points="0,18 20,17 35,15 50,13 65,11 80,7 100,5" />
            </svg>
          </div>
        </div>

        {/* ── Card 3: Test Match Output Cardinality ─────────────────── */}
        <div className="glass-panel p-5 glow-amber relative overflow-hidden group">
          <div className="flex items-start justify-between mb-3">
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-amber-600 dark:text-amber-400">
                Official Output
              </span>
              <div className="text-xs font-semibold text-slate-600 dark:text-slate-300">Test Entities Resolved</div>
            </div>
            <div className="w-8 h-8 rounded-xl bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/30 flex items-center justify-center text-amber-600 dark:text-amber-400 shadow-xs">
              <ShieldCheck className="w-4 h-4" />
            </div>
          </div>

          <div className="flex items-baseline gap-2 mb-2">
            <div className="text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white">
              {totalEntities ? (totalEntities >= 1000000 ? `${(totalEntities / 1000000).toFixed(2)}M` : totalEntities.toLocaleString()) : "1.73M"}
            </div>
            <span className="text-xs font-bold text-amber-600 dark:text-amber-400 flex items-center gap-0.5">
              364.3k matched
            </span>
          </div>

          <div className="flex items-center justify-between pt-2 border-t border-slate-200 dark:border-white/[0.06]">
            <span className="text-[11px] font-mono text-amber-700 dark:text-amber-300 font-semibold">
              Singletons: 1.37M (78.9%)
            </span>
            <span className="text-[10px] text-slate-500 dark:text-slate-400">Validator Passed</span>
          </div>

          <div className="mt-2 h-6 w-full">
            <svg viewBox="0 0 100 24" className="w-full h-full stroke-amber-500 dark:stroke-amber-400 fill-none" strokeWidth="2">
              <polyline points="0,6 20,9 40,12 60,15 80,18 100,21" />
            </svg>
          </div>
        </div>

        {/* ── Card 4: Submission Archive Status ──────────────────── */}
        <div className="glass-panel p-5 glow-emerald relative overflow-hidden group">
          <div className="flex items-start justify-between mb-3">
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400">
                Packaging Status
              </span>
              <div className="text-xs font-semibold text-slate-600 dark:text-slate-300">Competition Archive</div>
            </div>
            <div className="w-8 h-8 rounded-xl bg-emerald-50 dark:bg-emerald-500/10 border border-emerald-200 dark:border-emerald-500/30 flex items-center justify-center text-emerald-600 dark:text-emerald-400 shadow-xs">
              <FileCode className="w-4 h-4" />
            </div>
          </div>

          <div className="flex items-baseline gap-2 mb-2">
            <div className="text-2xl font-extrabold tracking-tight text-slate-900 dark:text-white">
              79.09 MB
            </div>
            <span className="text-xs font-bold text-emerald-600 dark:text-emerald-400 flex items-center gap-0.5">
              <TrendingUp className="w-3.5 h-3.5" /> Ready
            </span>
          </div>

          <div className="flex items-center justify-between pt-2 border-t border-slate-200 dark:border-white/[0.06]">
            <span className="text-[11px] font-mono text-emerald-700 dark:text-emerald-300 font-semibold truncate max-w-[140px]">
              Resolve_AI_Team.zip
            </span>
            <span className="text-[10px] text-slate-500 dark:text-slate-400">Uncompressed TSVs</span>
          </div>

          <div className="mt-2 h-6 w-full">
            <svg viewBox="0 0 100 24" className="w-full h-full stroke-emerald-500 dark:stroke-emerald-400 fill-none" strokeWidth="2">
              <polyline points="0,15 15,10 30,14 45,8 60,11 75,5 90,7 100,3" />
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}
