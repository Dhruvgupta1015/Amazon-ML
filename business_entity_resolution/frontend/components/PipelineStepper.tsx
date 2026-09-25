"use client";

import React, { useState } from "react";
import { 
  Database, 
  Sparkles, 
  Filter, 
  GitFork, 
  Sliders, 
  CheckCircle2, 
  ChevronRight,
  ArrowRight,
  Info
} from "lucide-react";

interface StageInfo {
  id: string;
  name: string;
  status: string;
  subtitle: string;
  metrics: string;
  detail: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
}

const STAGES: StageInfo[] = [
  {
    id: "ingestion",
    name: "Data Ingestion",
    status: "Cleaned",
    subtitle: "S1, S2, S3 TSV Files",
    metrics: "4 sources • 1.8M rows",
    detail: "Loads deduplicated reference S1 and noisy external sources S2 & S3 with robust TSV parsing.",
    icon: Database,
    color: "from-blue-600 to-indigo-600",
  },
  {
    id: "preproc",
    name: "Pre-processing",
    status: "Cleaned",
    subtitle: "NFKD Open-Set Normalizer",
    metrics: "US, India, France",
    detail: "Country-agnostic unicode normalization, legal suffix standardization, phone/postal extraction without hardcoding.",
    icon: Sparkles,
    color: "from-cyan-600 to-teal-600",
  },
  {
    id: "blocking",
    name: "Multi-Stage Blocking",
    status: "99.2% Recall",
    subtitle: "MinHash LSH + Dense ANN",
    metrics: "99.2% Recall • 98.4% RR",
    detail: "High-recall candidate generation combining 3-gram MinHash LSH, street number matching, and sentence-transformer ANN.",
    icon: Filter,
    color: "from-violet-600 to-purple-600",
  },
  {
    id: "matching",
    name: "Similarity Matching",
    status: "28 Features",
    subtitle: "Pairwise Feature Vectorizer",
    metrics: "28 dense features",
    detail: "Computes Levenshtein, Jaro-Winkler, Monge-Elkan, LCS, Double Metaphone, Token Set, and dense vector cosine for all candidate pairs.",
    icon: GitFork,
    color: "from-purple-600 to-pink-600",
  },
  {
    id: "linkage",
    name: "Linkage & Decision",
    status: "Calibrated",
    subtitle: "LightGBM + F₀.₅ Tuning",
    metrics: "Optimal τ* = 0.68",
    detail: "Tree-based gradient boosted ranking model calibrated to optimize macro F0.5 (penalizing false merges twice as hard as missed links).",
    icon: Sliders,
    color: "from-amber-600 to-orange-600",
  },
  {
    id: "consolidation",
    name: "Entity Consolidation",
    status: "1.8M Output",
    subtitle: "1-to-0, 1-to-1, 1-to-N",
    metrics: "Submission Ready",
    detail: "Maps each canonical S1 entity to its matched external cluster or empty set (singleton), matching Amazon ML 2026 specs.",
    icon: CheckCircle2,
    color: "from-emerald-600 to-green-600",
  },
];

export function PipelineStepper() {
  const [activeStage, setActiveStage] = useState<StageInfo>(STAGES[2]);

  return (
    <div className="glass-panel p-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 mb-5">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-base font-bold text-slate-900 dark:text-white tracking-tight">Active Resolution Pipeline</h3>
            <span className="figma-badge figma-badge-cyan">End-to-End ML Workflow</span>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Click any pipeline stage to inspect its processing internals, candidate reduction, and metric outputs
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-400 font-mono bg-slate-100 dark:bg-white/[0.03] px-3 py-1.5 rounded-lg border border-slate-200 dark:border-white/[0.06]">
          <span>F₀.₅ Trend:</span>
          <span className="text-slate-500 dark:text-slate-400">0.88</span>
          <ArrowRight className="w-3 h-3 text-emerald-600 dark:text-emerald-400" />
          <span className="text-emerald-700 dark:text-emerald-400 font-bold">0.942</span>
        </div>
      </div>

      {/* Horizontal Steps Bar */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-5">
        {STAGES.map((stage, idx) => {
          const Icon = stage.icon;
          const isSelected = activeStage.id === stage.id;
          return (
            <button
              key={stage.id}
              onClick={() => setActiveStage(stage)}
              className={`p-3.5 rounded-xl text-left transition-all duration-200 relative group flex flex-col justify-between ${
                isSelected
                  ? "bg-violet-50 dark:bg-violet-600/15 border-2 border-violet-500 shadow-md shadow-violet-500/15 scale-[1.02]"
                  : "bg-slate-50/70 dark:bg-white/[0.02] border border-slate-200 dark:border-white/[0.08] hover:border-slate-300 dark:hover:border-white/[0.2] hover:bg-slate-100/70 dark:hover:bg-white/[0.05]"
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className={`w-7 h-7 rounded-lg bg-gradient-to-br ${stage.color} flex items-center justify-center text-white shadow-xs`}>
                    <Icon className="w-3.5 h-3.5" />
                  </div>
                  <span className="text-[10px] font-mono text-slate-400 dark:text-slate-500 font-bold">
                    0{idx + 1}
                  </span>
                </div>
                <div className="text-xs font-bold text-slate-900 dark:text-slate-100 group-hover:text-violet-700 dark:group-hover:text-white leading-tight">
                  {stage.name}
                </div>
                <div className="text-[10px] text-slate-500 dark:text-slate-400 mt-0.5 truncate">
                  {stage.subtitle}
                </div>
              </div>

              <div className="mt-3 pt-2 border-t border-slate-200 dark:border-white/[0.06] flex items-center justify-between">
                <span className="text-[9px] font-semibold text-emerald-700 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-200 dark:border-transparent">
                  {stage.status}
                </span>
                <ChevronRight className={`w-3 h-3 transition-transform ${isSelected ? "text-violet-600 dark:text-violet-400 translate-x-0.5" : "text-slate-400 dark:text-slate-600 group-hover:text-slate-600 dark:group-hover:text-slate-400"}`} />
              </div>
            </button>
          );
        })}
      </div>

      {/* Selected Stage Detail Drawer Banner */}
      <div className="p-4 rounded-xl bg-slate-50 dark:bg-[#090d16]/90 border border-slate-200 dark:border-white/[0.08] flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-violet-100 dark:bg-violet-600/20 border border-violet-200 dark:border-violet-500/30 flex items-center justify-center text-violet-700 dark:text-violet-400 flex-shrink-0">
            <Info className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-slate-900 dark:text-white uppercase tracking-wider">{activeStage.name}</span>
              <span className="text-[10px] text-slate-500 dark:text-slate-400 font-mono">({activeStage.metrics})</span>
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-300 mt-0.5 max-w-2xl leading-relaxed">
              {activeStage.detail}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 self-end md:self-center">
          <span className="text-[11px] font-mono text-slate-500 dark:text-slate-400">Status:</span>
          <span className="text-xs font-bold text-emerald-700 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-500/10 border border-emerald-200 dark:border-emerald-500/20 px-2 py-0.5 rounded-lg flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
            Operational & Verified
          </span>
        </div>
      </div>
    </div>
  );
}
