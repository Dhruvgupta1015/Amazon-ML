"use client";

import React from "react";
import { Sparkles, Target, Zap, ShieldCheck } from "lucide-react";

interface MetricScoreboardProps {
  f05: number | null;
  precision: number | null;
  recall: number | null;
  threshold: number | null;
  label?: string;
}

export function MetricScoreboard({ f05, precision, recall, threshold, label = "Validation" }: MetricScoreboardProps) {
  const fmt = (v: number | null) => (v != null ? v.toFixed(4) : "—");
  return (
    <div className="glass-panel p-5">
      <div className="flex items-center justify-between mb-4">
        <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
          {label} Macro Metrics
        </span>
        <span className="figma-badge figma-badge-purple text-[10px]">β = 0.5</span>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="p-3.5 rounded-xl bg-violet-500/10 border border-violet-500/20 text-center glow-purple">
          <div className="text-2xl font-extrabold text-violet-300 font-mono">{fmt(f05)}</div>
          <div className="text-[11px] text-slate-400 mt-1 flex items-center justify-center gap-1">
            <Target className="w-3 h-3 text-violet-400" /> Macro F₀.₅
          </div>
        </div>
        <div className="p-3.5 rounded-xl bg-cyan-500/10 border border-cyan-500/20 text-center glow-cyan">
          <div className="text-2xl font-extrabold text-cyan-300 font-mono">{fmt(precision)}</div>
          <div className="text-[11px] text-slate-400 mt-1 flex items-center justify-center gap-1">
            <ShieldCheck className="w-3 h-3 text-cyan-400" /> Precision (2×)
          </div>
        </div>
        <div className="p-3.5 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-center">
          <div className="text-2xl font-extrabold text-indigo-300 font-mono">{fmt(recall)}</div>
          <div className="text-[11px] text-slate-400 mt-1 flex items-center justify-center gap-1">
            <Sparkles className="w-3 h-3 text-indigo-400" /> Recall
          </div>
        </div>
        <div className="p-3.5 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-center glow-emerald">
          <div className="text-2xl font-extrabold text-emerald-300 font-mono">
            {threshold != null ? threshold.toFixed(2) : "0.68"}
          </div>
          <div className="text-[11px] text-slate-400 mt-1 flex items-center justify-center gap-1">
            <Zap className="w-3 h-3 text-emerald-400" /> Threshold τ*
          </div>
        </div>
      </div>
    </div>
  );
}
