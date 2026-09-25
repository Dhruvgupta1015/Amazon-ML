"use client";

import React from "react";
import { Building2, MapPin, Globe, CheckCircle2, XCircle } from "lucide-react";

interface EntityCardProps {
  entityId: string;
  businessName: string;
  address: string;
  country: string;
  source: "S1" | "S2" | "S3";
  confidence?: number;
  isMatch?: boolean;
  highlightTokens?: string[];
}

function highlightText(text: string, tokens: string[]): React.ReactNode {
  if (!tokens.length) return <span>{text}</span>;
  const tokenSet = new Set(tokens.map((t) => t.toLowerCase()));
  return (
    <span>
      {text.split(/(\s+)/).map((w, i) =>
        tokenSet.has(w.toLowerCase().replace(/[\s,.-]/g, "")) ? (
          <mark key={i} className="bg-emerald-500/20 text-emerald-300 rounded px-1 py-0.5 font-medium">
            {w}
          </mark>
        ) : (
          <span key={i} className="text-slate-300">
            {w}
          </span>
        )
      )}
    </span>
  );
}

export function EntityCard({
  entityId,
  businessName,
  address,
  country,
  source,
  confidence,
  isMatch,
  highlightTokens = [],
}: EntityCardProps) {
  const sourceThemes = {
    S1: "border-violet-500/30 glow-purple bg-violet-950/15",
    S2: "border-cyan-500/30 glow-cyan bg-cyan-950/15",
    S3: "border-indigo-500/30 bg-indigo-950/15",
  };

  return (
    <div className={`glass-panel p-5 ${sourceThemes[source]} transition-all duration-200`}>
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-white/[0.08]">
        <div className="flex items-center gap-2">
          <span className="figma-badge figma-badge-purple font-mono">{entityId}</span>
          <span className="text-[10px] font-bold uppercase text-slate-400">
            {source === "S1" ? "Reference Source 1" : `Mention Source ${source}`}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="figma-badge figma-badge-cyan text-[10px]">{country}</span>
          {confidence != null && (
            <span className="figma-badge figma-badge-green text-[10px]">
              P = {confidence.toFixed(2)}
            </span>
          )}
          {isMatch != null && (
            <span
              className={`figma-badge text-[10px] ${
                isMatch ? "figma-badge-green" : "figma-badge-red"
              }`}
            >
              {isMatch ? "Match" : "No Match"}
            </span>
          )}
        </div>
      </div>

      <div className="space-y-3">
        <div>
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block mb-1">
            Business Name
          </span>
          <div className="text-base font-bold text-white leading-snug">
            {highlightText(businessName, highlightTokens)}
          </div>
        </div>

        <div>
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block mb-1">
            Address
          </span>
          <div className="text-xs text-slate-300 leading-relaxed">
            {highlightText(address, highlightTokens)}
          </div>
        </div>
      </div>
    </div>
  );
}
