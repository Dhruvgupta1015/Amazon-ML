"use client";

import React, { useState } from "react";
import { Building2, Globe, ShieldCheck, ArrowRight, Sparkles, CheckCircle2, AlertCircle } from "lucide-react";

interface SampleEntity {
  id: string;
  name: string;
  country: string;
  sources: {
    source: string;
    name: string;
    address: string;
    confidence: number;
    matchType: string;
  }[];
  canonical: {
    name: string;
    address: string;
    country: string;
    verified: boolean;
    confidence: number;
  };
  attributes: {
    f05Score: number;
    tokensMatched: number;
    phoneticMatch: boolean;
    denseCosine: number;
  };
}

const PRESET_ENTITIES: Record<string, SampleEntity> = {
  acme: {
    id: "S1-925783039",
    name: "Acme Corporation Inc",
    country: "US",
    canonical: {
      name: "Acme Corporation Inc.",
      address: "273 West Dickinson Ave, Suite 400, Seattle, WA 98101",
      country: "US",
      verified: true,
      confidence: 0.98,
    },
    sources: [
      {
        source: "Source 2 (CRM)",
        name: "Acme Corp US",
        address: "273 W Dickinson Ave Ste 400, Seattle 98101",
        confidence: 0.98,
        matchType: "High-Confidence Match",
      },
      {
        source: "Source 3 (Support)",
        name: "Acme Corp Support Div",
        address: "273 W. Dickinson Ave., Seattle WA",
        confidence: 0.94,
        matchType: "Fuzzy Token Match",
      },
      {
        source: "Source 2 (Sales)",
        name: "Acme Corporation",
        address: "273 West Dickinson Avenue, Seattle",
        confidence: 0.96,
        matchType: "Address Match",
      },
    ],
    attributes: {
      f05Score: 0.978,
      tokensMatched: 5,
      phoneticMatch: true,
      denseCosine: 0.962,
    },
  },
  bistro: {
    id: "S1-419820144",
    name: "Bistro Le Marais Paris",
    country: "FR",
    canonical: {
      name: "SARL Bistro Le Marais",
      address: "14 Rue des Rosiers, 75004 Paris, France",
      country: "FR",
      verified: true,
      confidence: 0.96,
    },
    sources: [
      {
        source: "Source 2 (Directories)",
        name: "Le Marais Bistro SARL",
        address: "14 r. des Rosiers, 75004 Paris",
        confidence: 0.95,
        matchType: "Unicode Normalized Match",
      },
      {
        source: "Source 3 (Social)",
        name: "Bistro Marais Paris 4",
        address: "14 Rue Rosiers, Paris",
        confidence: 0.92,
        matchType: "Dense Cosine Match",
      },
    ],
    attributes: {
      f05Score: 0.954,
      tokensMatched: 4,
      phoneticMatch: true,
      denseCosine: 0.941,
    },
  },
  tata: {
    id: "S1-772910381",
    name: "Tata Consultancy Services Ltd",
    country: "IN",
    canonical: {
      name: "Tata Consultancy Services Limited",
      address: "TCS House, Raveline Street, Fort, Mumbai 400001, Maharashtra",
      country: "IN",
      verified: true,
      confidence: 0.99,
    },
    sources: [
      {
        source: "Source 2 (Registry)",
        name: "TCS Ltd Mumbai",
        address: "Raveline St, Fort, Mumbai 400001",
        confidence: 0.99,
        matchType: "Acronym + Postal Match",
      },
      {
        source: "Source 3 (B2B)",
        name: "Tata Consultancy Services",
        address: "TCS House, Fort, Mumbai",
        confidence: 0.97,
        matchType: "Legal Token Match",
      },
    ],
    attributes: {
      f05Score: 0.992,
      tokensMatched: 6,
      phoneticMatch: true,
      denseCosine: 0.985,
    },
  },
  singleton: {
    id: "S1-100293847",
    name: "Hyperion Quantum Research Lab",
    country: "US",
    canonical: {
      name: "Hyperion Quantum Research Lab LLC",
      address: "100 Innovation Way, Los Alamos, NM 87544",
      country: "US",
      verified: true,
      confidence: 1.0,
    },
    sources: [],
    attributes: {
      f05Score: 1.0,
      tokensMatched: 0,
      phoneticMatch: false,
      denseCosine: 0.0,
    },
  },
};

export function EntityGraphVisualizer() {
  const [selectedKey, setSelectedKey] = useState<string>("acme");
  const entity = PRESET_ENTITIES[selectedKey];

  return (
    <div className="glass-panel p-6 overflow-hidden relative">
      {/* Background ambient glow */}
      <div className="absolute top-0 right-1/4 w-96 h-96 bg-violet-500/5 dark:bg-violet-600/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-0 left-1/4 w-80 h-80 bg-cyan-500/5 dark:bg-cyan-600/10 rounded-full blur-3xl pointer-events-none" />

      {/* Header with Preset Selector */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6 relative z-10">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-base font-bold text-slate-900 dark:text-white tracking-tight">Entity Graph Linking Visualizer</h3>
            <span className="figma-badge figma-badge-purple">Figma Interactive Node View</span>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Interactive multi-source resolution graph showing noisy mentions resolving to canonical S1 entities
          </p>
        </div>

        {/* Entity Preset Tabs */}
        <div className="flex items-center gap-1.5 p-1 bg-slate-100 dark:bg-white/[0.04] border border-slate-200 dark:border-white/[0.08] rounded-xl self-start">
          {[
            { key: "acme", label: "Acme Corp (US)", flag: "🇺🇸" },
            { key: "bistro", label: "Bistro Le Marais (FR)", flag: "🇫🇷" },
            { key: "tata", label: "TCS Mumbai (IN)", flag: "🇮🇳" },
            { key: "singleton", label: "Singleton (1-to-0)", flag: "🔒" },
          ].map((tab) => (
            <button
              key={tab.key}
              onClick={() => setSelectedKey(tab.key)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 flex items-center gap-1.5 ${
                selectedKey === tab.key
                  ? "bg-violet-600 text-white shadow-md shadow-violet-500/25"
                  : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-200/60 dark:hover:bg-white/[0.04]"
              }`}
            >
              <span>{tab.flag}</span>
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Graph Area & Details Split View */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center relative z-10">
        {/* Left / Center Node Graph (8 cols) */}
        <div className="lg:col-span-8 bg-slate-50/90 dark:bg-[#090d16]/80 border border-slate-200 dark:border-white/[0.06] rounded-2xl p-6 min-h-[340px] flex items-center justify-center relative overflow-hidden">
          {/* Subtle grid pattern background */}
          <div className="absolute inset-0 dot-pattern opacity-60 dark:opacity-40" />

          {entity.sources.length === 0 ? (
            /* Singleton State */
            <div className="text-center py-10 relative z-10">
              <div className="w-16 h-16 rounded-2xl bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/30 flex items-center justify-center mx-auto mb-4 text-amber-600 dark:text-amber-400 shadow-xs">
                <ShieldCheck className="w-8 h-8" />
              </div>
              <h4 className="text-sm font-bold text-slate-900 dark:text-white mb-1">True Singleton Entity (No External Mentions)</h4>
              <p className="text-xs text-slate-500 dark:text-slate-400 max-w-sm mx-auto mb-4">
                Our F₀.₅ optimizer predicts an empty match set for this S1 entity. In the Amazon ML challenge,
                predicting empty for a true singleton yields a perfect F₀.₅ score of 1.0!
              </p>
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-50 dark:bg-emerald-500/15 border border-emerald-200 dark:border-emerald-500/30 text-emerald-800 dark:text-emerald-300 text-xs font-semibold">
                <CheckCircle2 className="w-4 h-4" /> Score: F₀.₅ = 1.0000 (Protected from False Merges)
              </div>
            </div>
          ) : (
            /* Multi-Node Resolution Visualizer */
            <div className="w-full flex flex-col md:flex-row items-center justify-between gap-6 relative z-10">
              {/* Left Column: Noisy Ingestion Sources */}
              <div className="space-y-3 w-full md:w-64">
                <div className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 px-1">
                  Incoming Mentions (S2 / S3)
                </div>
                {entity.sources.map((src, i) => (
                  <div
                    key={i}
                    className="p-3 rounded-xl bg-white dark:bg-[#0e1424] border border-cyan-200 dark:border-cyan-500/25 shadow-sm dark:shadow-lg dark:shadow-cyan-500/5 hover:border-cyan-400/50 transition-all group"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[10px] font-bold text-cyan-700 dark:text-cyan-400 uppercase tracking-tight flex items-center gap-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-pulse" />
                        {src.source}
                      </span>
                      <span className="text-[10px] font-mono text-cyan-800 dark:text-cyan-300 bg-cyan-50 dark:bg-cyan-500/10 px-1.5 py-0.5 rounded font-semibold">
                        {(src.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="text-xs font-semibold text-slate-900 dark:text-slate-200 group-hover:text-cyan-700 dark:group-hover:text-white truncate">
                      {src.name}
                    </div>
                    <div className="text-[10px] text-slate-500 dark:text-slate-400 truncate mt-0.5">{src.address}</div>
                  </div>
                ))}
              </div>

              {/* Center Connectors & AI Resolution Hub */}
              <div className="flex flex-col items-center justify-center px-2 py-4">
                <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-violet-600 to-cyan-500 p-[1px] shadow-lg shadow-violet-500/20">
                  <div className="w-full h-full bg-white dark:bg-[#0c111d] rounded-[15px] flex items-center justify-center">
                    <Sparkles className="w-5 h-5 text-violet-600 dark:text-violet-300 animate-pulse" />
                  </div>
                </div>
                <div className="text-[10px] font-mono font-bold text-violet-700 dark:text-violet-300 mt-2 bg-violet-50 dark:bg-violet-500/10 px-2 py-0.5 rounded-full border border-violet-200 dark:border-violet-500/20">
                  LightGBM τ*=0.68
                </div>
                <div className="text-[9px] text-slate-500 dark:text-slate-400 mt-0.5">Macro F₀.₅ Match</div>
              </div>

              {/* Right Column: Canonical Target S1 Entity */}
              <div className="w-full md:w-64">
                <div className="text-[10px] font-bold uppercase tracking-wider text-violet-600 dark:text-violet-400 px-1 mb-3">
                  Canonical S1 Reference
                </div>
                <div className="p-4 rounded-xl bg-violet-50/80 dark:bg-[#120f26] border border-violet-200 dark:border-violet-500/40 shadow-sm dark:shadow-xl dark:shadow-violet-500/10 glow-purple">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[10px] font-mono font-bold text-violet-700 dark:text-violet-300 bg-white dark:bg-violet-500/20 px-2 py-0.5 rounded border border-violet-200 dark:border-violet-500/30">
                      {entity.id}
                    </span>
                    <span className="figma-badge figma-badge-purple text-[10px]">
                      {entity.canonical.country}
                    </span>
                  </div>
                  <div className="text-xs font-bold text-slate-900 dark:text-white mb-1">{entity.canonical.name}</div>
                  <div className="text-[11px] text-slate-600 dark:text-slate-300 leading-snug">{entity.canonical.address}</div>
                  <div className="mt-3 pt-3 border-t border-violet-200 dark:border-violet-500/20 flex items-center justify-between text-[11px]">
                    <span className="text-slate-500 dark:text-slate-400">Match Confidence</span>
                    <span className="font-bold text-emerald-700 dark:text-emerald-400">
                      {(entity.canonical.confidence * 100).toFixed(0)}% Match
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Detail Card (4 cols) */}
        <div className="lg:col-span-4 bg-slate-50/80 dark:bg-[#090d16]/80 border border-slate-200 dark:border-white/[0.08] rounded-2xl p-5 space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-200 dark:border-white/[0.06]">
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">Resolved Entity</span>
              <div className="text-sm font-bold text-slate-900 dark:text-white">{entity.name}</div>
            </div>
            <span className="figma-badge figma-badge-green font-mono">
              F₀.₅: {entity.attributes.f05Score.toFixed(3)}
            </span>
          </div>

          {/* Metric dials */}
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="p-2.5 rounded-xl bg-white dark:bg-white/[0.02] border border-slate-200 dark:border-white/[0.05]">
              <div className="text-[10px] text-slate-500 dark:text-slate-400">Token Overlap</div>
              <div className="text-sm font-bold text-violet-700 dark:text-violet-400 mt-0.5">
                {entity.attributes.tokensMatched} / 6 Words
              </div>
            </div>
            <div className="p-2.5 rounded-xl bg-white dark:bg-white/[0.02] border border-slate-200 dark:border-white/[0.05]">
              <div className="text-[10px] text-slate-500 dark:text-slate-400">Dense Cosine</div>
              <div className="text-sm font-bold text-cyan-700 dark:text-cyan-400 mt-0.5">
                {entity.attributes.denseCosine.toFixed(3)}
              </div>
            </div>
          </div>

          {/* Feature explanations */}
          <div className="space-y-2 pt-1 text-xs">
            <div className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Pairwise Features
            </div>
            <div className="flex items-center justify-between text-slate-700 dark:text-slate-300 py-1 border-b border-slate-200 dark:border-white/[0.04]">
              <span>Phonetic Metaphone</span>
              <span className="text-emerald-700 dark:text-emerald-400 font-semibold">
                {entity.attributes.phoneticMatch ? "Exact Phonetic Match" : "Distinct"}
              </span>
            </div>
            <div className="flex items-center justify-between text-slate-700 dark:text-slate-300 py-1 border-b border-slate-200 dark:border-white/[0.04]">
              <span>Address Digit Matching</span>
              <span className="text-emerald-700 dark:text-emerald-400 font-semibold">Verified Number/ZIP</span>
            </div>
            <div className="flex items-center justify-between text-slate-700 dark:text-slate-300 py-1">
              <span>Decision Verdict</span>
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-100 dark:bg-emerald-500/20 text-emerald-800 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-500/30">
                MERGE APPROVED
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
