"use client";

import React from "react";
import { Sliders, Sparkles } from "lucide-react";

interface ThresholdSliderProps {
  value: number;
  onChange: (v: number) => void;
  optimal?: number | null;
}

export function ThresholdSlider({ value, onChange, optimal }: ThresholdSliderProps) {
  const delta = optimal != null ? (value - optimal).toFixed(2) : null;
  const isOptimal = optimal != null && Math.abs(value - optimal) < 0.015;
  return (
    <div className="glass-panel p-4 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
          <Sliders className="w-3.5 h-3.5 text-violet-400" /> Decision Threshold
        </span>
        <div className="flex items-center gap-2">
          <span className="text-sm font-bold font-mono text-violet-300 bg-violet-500/20 px-2 py-0.5 rounded border border-violet-500/30">
            τ = {value.toFixed(2)}
          </span>
          {delta && (
            <span
              className={`figma-badge text-[10px] ${
                isOptimal ? "figma-badge-green" : "figma-badge-amber"
              }`}
            >
              {isOptimal ? "★ Optimal" : `Δ${delta}`}
            </span>
          )}
        </div>
      </div>
      <input
        type="range"
        min="0.05"
        max="0.99"
        step="0.01"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-violet-500 cursor-pointer h-2 bg-slate-800 rounded-lg"
        id="threshold-range-input"
      />
      <div className="flex justify-between text-[11px] text-slate-500 font-mono">
        <span>← High Recall (More Links)</span>
        <span>High Precision (Fewer, Safer Links) →</span>
      </div>
    </div>
  );
}
