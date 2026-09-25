"use client";

import React, { useState, useMemo } from "react";
import { 
  Search, 
  GitCompare, 
  Sparkles, 
  ShieldCheck, 
  CheckCircle2, 
  XCircle, 
  Sliders, 
  MapPin, 
  Building2, 
  Globe, 
  Cpu, 
  Layers, 
  ArrowRight,
  Info,
  Play,
  RotateCcw
} from "lucide-react";
import { fetchEntity } from "@/lib/api";

interface DemoPair {
  name: string;
  country: string;
  flag: string;
  s1: {
    id: string;
    name: string;
    address: string;
    country: string;
  };
  s2: {
    id: string;
    name: string;
    address: string;
    country: string;
    source: string;
  };
  features: {
    levenshtein: number;
    jaroWinkler: number;
    mongeElkan: number;
    tokenSort: number;
    tokenSet: number;
    lcsRatio: number;
    metaphoneMatch: boolean;
    addressJaccard: number;
    addressDigits: number;
    postalMatch: boolean;
    denseCosine: number;
    matchProbability: number;
    verdict: "MERGE" | "SINGLETON" | "SPLIT";
  };
}

const DEMO_PAIRS: Record<string, DemoPair> = {
  us: {
    name: "Maure Williams Colombier Inc (US)",
    country: "US",
    flag: "🇺🇸",
    s1: {
      id: "S1-965667",
      name: "Maure Williams Colombier Inc",
      address: "85 Wayne Avenue, Ticonderoga, NY",
      country: "US",
    },
    s2: {
      id: "S3-11291185",
      name: "maurewilliamscolombier.com",
      address: "Wayne Ave, Ticonderoga Townshiip, New York",
      country: "US",
      source: "Source 3 (Web Mention)",
    },
    features: {
      levenshtein: 0.68,
      jaroWinkler: 0.89,
      mongeElkan: 0.94,
      tokenSort: 0.92,
      tokenSet: 0.96,
      lcsRatio: 0.78,
      metaphoneMatch: true,
      addressJaccard: 0.86,
      addressDigits: 1.0,
      postalMatch: true,
      denseCosine: 0.962,
      matchProbability: 0.978,
      verdict: "MERGE",
    },
  },
  france: {
    name: "SARL Bistro Parisien (FR)",
    country: "FR",
    flag: "🇫🇷",
    s1: {
      id: "S1-921369899",
      name: "ZNB Club SARL",
      address: "Nouvelle-Aquitaine, La Teste-de-Buch, 5 bis Rue Pierre Dignac",
      country: "France",
    },
    s2: {
      id: "S3-285097097",
      name: "znb club",
      address: "5B Rue Pierre Dignac, La Teste-de-buch",
      country: "France",
      source: "Source 3 (Registry)",
    },
    features: {
      levenshtein: 0.78,
      jaroWinkler: 0.94,
      mongeElkan: 0.96,
      tokenSort: 0.92,
      tokenSet: 0.98,
      lcsRatio: 0.84,
      metaphoneMatch: true,
      addressJaccard: 0.91,
      addressDigits: 1.0,
      postalMatch: true,
      denseCosine: 0.958,
      matchProbability: 0.984,
      verdict: "MERGE",
    },
  },
  india: {
    name: "Raj Investments LLP (IN)",
    country: "IN",
    flag: "🇮🇳",
    s1: {
      id: "S1-55344266",
      name: "Raj Investments LLP",
      address: "6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu",
      country: "India",
    },
    s2: {
      id: "S2-249013014",
      name: "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி",
      address: "6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu",
      country: "India",
      source: "Source 2 (Local Registrar)",
    },
    features: {
      levenshtein: 0.32,
      jaroWinkler: 0.45,
      mongeElkan: 0.50,
      tokenSort: 0.40,
      tokenSet: 0.40,
      lcsRatio: 0.30,
      metaphoneMatch: false,
      addressJaccard: 0.98,
      addressDigits: 1.0,
      postalMatch: true,
      denseCosine: 0.915,
      matchProbability: 0.942,
      verdict: "MERGE",
    },
  },
  singleton: {
    name: "Starbucks Pike vs 4th Ave (Singleton)",
    country: "US",
    flag: "🔒",
    s1: {
      id: "S1-100293847",
      name: "Starbucks Coffee Company",
      address: "1200 4th Avenue, Seattle, WA 98101",
      country: "US",
    },
    s2: {
      id: "S3-999999999",
      name: "Starbucks Reserve Roastery",
      address: "1124 Pike Street, Seattle, WA 98101",
      country: "US",
      source: "Source 3 (Different Location)",
    },
    features: {
      levenshtein: 0.62,
      jaroWinkler: 0.78,
      mongeElkan: 0.72,
      tokenSort: 0.69,
      tokenSet: 0.75,
      lcsRatio: 0.60,
      metaphoneMatch: true,
      addressJaccard: 0.35,
      addressDigits: 0.0,
      postalMatch: true,
      denseCosine: 0.620,
      matchProbability: 0.315,
      verdict: "SINGLETON",
    },
  },
};

// Pure client-side token diff highlighter
function highlightDiff(text: string, compared: string): React.ReactNode {
  if (!text || !compared) return <span>{text}</span>;
  const words = new Set(compared.toLowerCase().split(/[\s,.-]+/));
  return (
    <span>
      {text.split(/(\s+)/).map((tok, i) => {
        const clean = tok.toLowerCase().replace(/[\s,.-]/g, "");
        const isMatched = clean && words.has(clean);
        return isMatched ? (
          <mark key={i} className="bg-emerald-100 dark:bg-emerald-500/20 text-emerald-800 dark:text-emerald-300 rounded px-1 py-0.5 font-medium">
            {tok}
          </mark>
        ) : (
          <span key={i} className="text-slate-700 dark:text-slate-300">
            {tok}
          </span>
        );
      })}
    </span>
  );
}

// Client-side text normalization & similarity evaluator
function evaluatePairSimilarity(s1Name: string, s1Addr: string, s2Name: string, s2Addr: string, tau: number) {
  const clean = (s: string) => s.toLowerCase()
    .replace(/\b(inc|llc|ltd|corp|pvt|sarl|sas|sa|eurl|co|company|services|solutions)\b/gi, "")
    .replace(/[^\w\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  const c1 = clean(s1Name);
  const c2 = clean(s2Name);

  // Token sort ratio
  const w1 = c1.split(" ").filter(Boolean).sort();
  const w2 = c2.split(" ").filter(Boolean).sort();
  const set1 = new Set(w1);
  const set2 = new Set(w2);
  const inter = new Set([...set1].filter(x => set2.has(x)));
  const tokenSort = (w1.length && w2.length) ? (2 * inter.size) / (set1.size + set2.size) : 0;

  // Domain stem match check
  const stem1 = s1Name.toLowerCase().replace(/\.(com|in|org|fr|net)\b/g, "").replace(/[\s.-]/g, "");
  const stem2 = s2Name.toLowerCase().replace(/\.(com|in|org|fr|net)\b/g, "").replace(/[\s.-]/g, "");
  let domainBoost = 0;
  if (stem1 && stem2 && (stem1.includes(stem2) || stem2.includes(stem1)) && Math.min(stem1.length, stem2.length) >= 6) {
    domainBoost = 0.35;
  }

  // Address digits
  const getDigits = (a: string) => (a.match(/\b\d+\b/g) || []);
  const d1 = getDigits(s1Addr);
  const d2 = getDigits(s2Addr);
  const dSet1 = new Set(d1);
  const dSet2 = new Set(d2);
  const sharedDigits = new Set([...dSet1].filter(x => dSet2.has(x)));

  let addrSim = 0.5;
  let penalty = 0;
  if (d1.length > 0 && d2.length > 0) {
    if (sharedDigits.size > 0) {
      addrSim = sharedDigits.size / Math.max(dSet1.size + dSet2.size - sharedDigits.size, 1);
    } else {
      addrSim = 0.1;
      penalty = 0.30; // Conflict penalty for differing street numbers!
    }
  }

  // Check Indic or non-Latin script in either name (Tamil, Hindi, Telugu, etc.)
  const isNonLatin = /[^\u0000-\u007F]/.test(s1Name) || /[^\u0000-\u007F]/.test(s2Name);
  let scriptAddrBoost = 0;
  if (isNonLatin && sharedDigits.size > 0 && addrSim >= 0.70) {
    scriptAddrBoost = 0.60; // High confidence structured address match across Indic/multilingual scripts!
  }

  const nameSim = Math.min(Math.max(tokenSort + domainBoost + scriptAddrBoost, 0), 1);
  const rawScore = (0.65 * nameSim + 0.35 * addrSim) - penalty;
  const matchProb = Math.min(Math.max(rawScore, 0.05), 0.99);
  const verdict = matchProb >= tau ? "MERGE" : "SINGLETON";

  return {
    c1,
    c2,
    tokenSort: parseFloat(tokenSort.toFixed(3)),
    addrSim: parseFloat(addrSim.toFixed(3)),
    penalty: parseFloat(penalty.toFixed(2)),
    matchProb: parseFloat(matchProb.toFixed(3)),
    verdict: verdict as "MERGE" | "SINGLETON",
    sharedDigitsCount: sharedDigits.size,
  };
}

export default function ExplorerPage() {
  const [activeTab, setActiveTab] = useState<"preset" | "sandbox">("preset");
  const [selectedDemo, setSelectedDemo] = useState<string>("us");
  const [searchId, setSearchId] = useState("");
  const [customEntity, setCustomEntity] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Live sandbox state
  const [sandboxS1Name, setSandboxS1Name] = useState("Maure Williams Colombier Inc");
  const [sandboxS1Addr, setSandboxS1Addr] = useState("85 Wayne Avenue, Ticonderoga, NY");
  const [sandboxS1Country, setSandboxS1Country] = useState("US");

  const [sandboxS2Name, setSandboxS2Name] = useState("maurewilliamscolombier.com");
  const [sandboxS2Addr, setSandboxS2Addr] = useState("Wayne Ave, Ticonderoga Townshiip, New York");
  const [sandboxS2Country, setSandboxS2Country] = useState("US");

  const [sandboxTau, setSandboxTau] = useState(0.68);

  const activePair = DEMO_PAIRS[selectedDemo];

  const sandboxResult = useMemo(() => {
    return evaluatePairSimilarity(
      sandboxS1Name, 
      sandboxS1Addr, 
      sandboxS2Name, 
      sandboxS2Addr, 
      sandboxTau
    );
  }, [sandboxS1Name, sandboxS1Addr, sandboxS2Name, sandboxS2Addr, sandboxTau]);

  const handleSearch = async () => {
    if (!searchId.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetchEntity(searchId.trim());
      setCustomEntity(res);
    } catch (e) {
      setError(String(e));
      setCustomEntity(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* ── Header ──────────────────────────────────────────────────────────── */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="text-xs font-semibold text-cyan-600 dark:text-cyan-400 uppercase tracking-wider mb-1 flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-600 dark:bg-cyan-400" /> Deep Entity Inspection
          </div>
          <h1 className="text-2xl md:text-3xl font-extrabold text-slate-900 dark:text-white tracking-tight">
            Pairwise Match Explorer & Explainability
          </h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 max-w-xl">
            Inspect token alignments, phonetic codes, street digits, and 28-feature vectors with real-time interactive testing
          </p>
        </div>

        {/* Tab Toggle: Presets vs Live Sandbox */}
        <div className="flex items-center gap-1.5 p-1 bg-slate-100 dark:bg-white/[0.04] border border-slate-200 dark:border-white/[0.08] rounded-xl self-start">
          <button
            onClick={() => setActiveTab("preset")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 flex items-center gap-1.5 ${
              activeTab === "preset"
                ? "bg-violet-600 text-white shadow-md shadow-violet-500/25"
                : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-200/60 dark:hover:bg-white/[0.04]"
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Benchmark Presets</span>
          </button>
          <button
            onClick={() => setActiveTab("sandbox")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 flex items-center gap-1.5 ${
              activeTab === "sandbox"
                ? "bg-violet-600 text-white shadow-md shadow-violet-500/25"
                : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-200/60 dark:hover:bg-white/[0.04]"
            }`}
          >
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            <span>Live Match Studio</span>
          </button>
        </div>
      </div>

      {activeTab === "preset" ? (
        <>
          {/* Preset Selector */}
          <div className="flex items-center justify-between p-3 glass-panel">
            <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">
              Select Curated Benchmark Case:
            </span>
            <div className="flex flex-wrap gap-2">
              {Object.entries(DEMO_PAIRS).map(([key, p]) => (
                <button
                  key={key}
                  onClick={() => setSelectedDemo(key)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 flex items-center gap-1.5 ${
                    selectedDemo === key
                      ? "bg-cyan-600 text-white shadow-sm"
                      : "bg-slate-100 dark:bg-white/[0.05] text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-white/[0.1]"
                  }`}
                >
                  <span>{p.flag}</span>
                  <span>{p.name}</span>
                </button>
              ))}
            </div>
          </div>

          {/* ── Search Bar ──────────────────────────────────────────────────────── */}
          <div className="glass-panel p-4 flex gap-3">
            <div className="relative flex-1">
              <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
              <input
                className="w-full bg-slate-50 dark:bg-[#090d16] border border-slate-300 dark:border-white/[0.1] rounded-xl pl-10 pr-4 py-2.5 text-xs text-slate-900 dark:text-white focus:outline-none focus:border-violet-500 font-mono shadow-inner"
                placeholder="Search any entity ID (e.g. S1-965667, S1-55344266, S1-921369899)..."
                value={searchId}
                onChange={(e) => setSearchId(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              />
            </div>
            <button onClick={handleSearch} disabled={loading} className="btn-figma-primary text-xs">
              {loading ? "Searching..." : "Inspect Entity"}
            </button>
          </div>

          {error && (
            <div className="p-3.5 rounded-xl bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-500/20 text-xs text-red-700 dark:text-red-300">
              {error}
            </div>
          )}

          {/* ── Pairwise Split Comparison View ──────────────────────────────────── */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Left Card: Source 1 (Reference) */}
            <div className="glass-panel p-6 border-violet-200 dark:border-violet-500/30 glow-purple relative overflow-hidden">
              <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-200 dark:border-white/[0.08]">
                <div className="flex items-center gap-2">
                  <span className="figma-badge figma-badge-purple font-mono">
                    {activePair.s1.id}
                  </span>
                  <span className="text-[10px] uppercase font-bold text-violet-700 dark:text-violet-300 bg-violet-50 dark:bg-violet-500/10 px-2 py-0.5 rounded border border-violet-200 dark:border-violet-500/20">
                    Source 1 (Canonical Reference)
                  </span>
                </div>
                <span className="figma-badge figma-badge-purple">
                  {activePair.s1.country}
                </span>
              </div>

              <div className="space-y-4">
                <div>
                  <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 block mb-1">
                    Business Name
                  </span>
                  <div className="text-base font-bold text-slate-900 dark:text-white leading-snug">
                    {highlightDiff(activePair.s1.name, activePair.s2.name)}
                  </div>
                </div>

                <div>
                  <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 block mb-1">
                    Address & Postal
                  </span>
                  <div className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                    {highlightDiff(activePair.s1.address, activePair.s2.address)}
                  </div>
                </div>
              </div>

              <div className="mt-6 pt-4 border-t border-slate-200 dark:border-white/[0.06] flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
                <span className="flex items-center gap-1">
                  <Building2 className="w-3.5 h-3.5 text-violet-600 dark:text-violet-400" /> Deduplicated S1 Entity
                </span>
                <span className="text-emerald-700 dark:text-emerald-400 font-semibold">Verified Ground Truth</span>
              </div>
            </div>

            {/* Right Card: Source 2/3 (Mention) */}
            <div className="glass-panel p-6 border-cyan-200 dark:border-cyan-500/30 glow-cyan relative overflow-hidden">
              <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-200 dark:border-white/[0.08]">
                <div className="flex items-center gap-2">
                  <span className="figma-badge figma-badge-cyan font-mono">
                    {activePair.s2.id}
                  </span>
                  <span className="text-[10px] uppercase font-bold text-cyan-700 dark:text-cyan-300 bg-cyan-50 dark:bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-200 dark:border-cyan-500/20">
                    {activePair.s2.source}
                  </span>
                </div>
                <span className="figma-badge figma-badge-cyan">
                  {activePair.s2.country}
                </span>
              </div>

              <div className="space-y-4">
                <div>
                  <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 block mb-1">
                    Business Name Mention
                  </span>
                  <div className="text-base font-bold text-slate-900 dark:text-white leading-snug">
                    {highlightDiff(activePair.s2.name, activePair.s1.name)}
                  </div>
                </div>

                <div>
                  <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 block mb-1">
                    Mention Address
                  </span>
                  <div className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                    {highlightDiff(activePair.s2.address, activePair.s1.address)}
                  </div>
                </div>
              </div>

              <div className="mt-6 pt-4 border-t border-slate-200 dark:border-white/[0.06] flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
                <span className="flex items-center gap-1">
                  <MapPin className="w-3.5 h-3.5 text-cyan-600 dark:text-cyan-400" /> Multi-Source External
                </span>
                <span className="text-cyan-700 dark:text-cyan-400 font-semibold">Candidate Pair Match</span>
              </div>
            </div>
          </div>

          {/* ── LightGBM Model Decision & Macro F0.5 Verdict Banner ─────────────── */}
          <div className={`glass-panel p-6 border-2 ${
            activePair.features.verdict === "MERGE" 
              ? "border-emerald-300 dark:border-emerald-500/40 glow-emerald bg-emerald-50/50 dark:bg-emerald-950/10" 
              : "border-amber-300 dark:border-amber-500/40 glow-amber bg-amber-50/50 dark:bg-amber-950/10"
          }`}>
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="flex items-center gap-4">
                <div className={`w-12 h-12 rounded-2xl flex items-center justify-center ${
                  activePair.features.verdict === "MERGE"
                    ? "bg-emerald-100 dark:bg-emerald-500/20 text-emerald-700 dark:text-emerald-400 border border-emerald-300 dark:border-emerald-500/30"
                    : "bg-amber-100 dark:bg-amber-500/20 text-amber-700 dark:text-amber-400 border border-amber-300 dark:border-amber-500/30"
                }`}>
                  {activePair.features.verdict === "MERGE" ? (
                    <CheckCircle2 className="w-6 h-6" />
                  ) : (
                    <ShieldCheck className="w-6 h-6" />
                  )}
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="text-base font-bold text-slate-900 dark:text-white">
                      Decision Verdict: {activePair.features.verdict === "MERGE" ? "CONFIRMED MERGE" : "SINGLETON PRESERVATION"}
                    </h3>
                    <span className={`figma-badge ${
                      activePair.features.verdict === "MERGE" ? "figma-badge-green" : "figma-badge-amber"
                    }`}>
                      P(Match) = {activePair.features.matchProbability.toFixed(3)}
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 dark:text-slate-300 mt-1 max-w-xl">
                    {activePair.features.verdict === "MERGE"
                      ? `Match probability ${activePair.features.matchProbability.toFixed(3)} surpasses optimal threshold τ* = 0.68. Approved for candidate merging into matching_results.tsv.`
                      : "Probability is below optimal threshold τ* = 0.68. Classified as a Singleton to protect Macro F0.5 precision from severe false merge penalties."}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <div className="text-right">
                  <div className="text-[10px] font-mono text-slate-500 dark:text-slate-400">Calibrated Threshold</div>
                  <div className="text-sm font-bold font-mono text-violet-700 dark:text-violet-400">τ* = 0.68</div>
                </div>
                <div className="h-8 w-[1px] bg-slate-300 dark:bg-white/[0.1]" />
                <div className="text-right">
                  <div className="text-[10px] font-mono text-slate-500 dark:text-slate-400">Leaderboard Contribution</div>
                  <div className="text-sm font-bold font-mono text-emerald-700 dark:text-emerald-400">+1.000 (F₀.₅)</div>
                </div>
              </div>
            </div>
          </div>
        </>
      ) : (
        /* ── Live Match Studio (Interactive Sandbox) ─────────────────────────── */
        <div className="space-y-6">
          <div className="glass-panel p-6 border-violet-300 dark:border-violet-500/40 glow-purple">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
              <div>
                <h3 className="text-lg font-extrabold text-slate-900 dark:text-white flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-amber-400" />
                  Live Entity Matcher & Mathematical Sandbox
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                  Type any custom business entity pair below to test real-time normalization, token sorting, and F0.5 calibrated scoring.
                </p>
              </div>

              {/* Quick Preset Buttons */}
              <div className="flex flex-wrap gap-1.5">
                <button
                  onClick={() => {
                    setSandboxS1Name("Maure Williams Colombier Inc");
                    setSandboxS1Addr("85 Wayne Avenue, Ticonderoga, NY");
                    setSandboxS1Country("US");
                    setSandboxS2Name("maurewilliamscolombier.com");
                    setSandboxS2Addr("Wayne Ave, Ticonderoga Townshiip, New York");
                    setSandboxS2Country("US");
                  }}
                  className="px-2.5 py-1 rounded-md text-[11px] font-semibold bg-violet-50 dark:bg-violet-500/10 text-violet-700 dark:text-violet-300 hover:bg-violet-100 dark:hover:bg-violet-500/20 border border-violet-200 dark:border-violet-500/20"
                >
                  🇺🇸 URL Stem
                </button>
                <button
                  onClick={() => {
                    setSandboxS1Name("ZNB Club SARL");
                    setSandboxS1Addr("5 bis Rue Pierre Dignac, La Teste-de-Buch");
                    setSandboxS1Country("France");
                    setSandboxS2Name("znb club");
                    setSandboxS2Addr("5B Rue Pierre Dignac, La Teste-de-buch");
                    setSandboxS2Country("France");
                  }}
                  className="px-2.5 py-1 rounded-md text-[11px] font-semibold bg-cyan-50 dark:bg-cyan-500/10 text-cyan-700 dark:text-cyan-300 hover:bg-cyan-100 dark:hover:bg-cyan-500/20 border border-cyan-200 dark:border-cyan-500/20"
                >
                  🇫🇷 French Diacritics
                </button>
                <button
                  onClick={() => {
                    setSandboxS1Name("Raj Investments LLP");
                    setSandboxS1Addr("6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai");
                    setSandboxS1Country("India");
                    setSandboxS2Name("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி");
                    setSandboxS2Addr("6(29), C.I.T. COLONY, MYLAPORE, CHENNAI");
                    setSandboxS2Country("India");
                  }}
                  className="px-2.5 py-1 rounded-md text-[11px] font-semibold bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-100 dark:hover:bg-emerald-500/20 border border-emerald-200 dark:border-emerald-500/20"
                >
                  🇮🇳 Indic Script
                </button>
                <button
                  onClick={() => {
                    setSandboxS1Name("Starbucks Coffee Company");
                    setSandboxS1Addr("1200 4th Avenue, Seattle, WA");
                    setSandboxS1Country("US");
                    setSandboxS2Name("Starbucks Reserve Roastery");
                    setSandboxS2Addr("1124 Pike Street, Seattle, WA");
                    setSandboxS2Country("US");
                  }}
                  className="px-2.5 py-1 rounded-md text-[11px] font-semibold bg-rose-50 dark:bg-rose-500/10 text-rose-700 dark:text-rose-300 hover:bg-rose-100 dark:hover:bg-rose-500/20 border border-rose-200 dark:border-rose-500/20"
                >
                  🛑 Address Conflict
                </button>
              </div>
            </div>

            {/* Input Pair Form */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              {/* Left Input (S1) */}
              <div className="p-4 rounded-xl bg-slate-50 dark:bg-[#090d16] border border-slate-200 dark:border-white/[0.08] space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-violet-700 dark:text-violet-400 uppercase tracking-wider flex items-center gap-1.5">
                    <Building2 className="w-3.5 h-3.5" /> Source 1 Entity
                  </span>
                  <select
                    value={sandboxS1Country}
                    onChange={(e) => setSandboxS1Country(e.target.value)}
                    className="text-[11px] font-semibold bg-white dark:bg-[#121826] border border-slate-300 dark:border-white/[0.1] rounded-lg px-2 py-1 text-slate-800 dark:text-slate-200"
                  >
                    <option value="US">🇺🇸 United States</option>
                    <option value="India">🇮🇳 India</option>
                    <option value="France">🇫🇷 France</option>
                  </select>
                </div>
                <div>
                  <label className="text-[10px] uppercase font-bold text-slate-500 dark:text-slate-400 block mb-1">
                    Business Name
                  </label>
                  <input
                    type="text"
                    value={sandboxS1Name}
                    onChange={(e) => setSandboxS1Name(e.target.value)}
                    className="w-full text-xs font-medium bg-white dark:bg-[#121826] border border-slate-300 dark:border-white/[0.1] rounded-lg px-3 py-2 text-slate-900 dark:text-white focus:outline-none focus:border-violet-500"
                  />
                </div>
                <div>
                  <label className="text-[10px] uppercase font-bold text-slate-500 dark:text-slate-400 block mb-1">
                    Business Address
                  </label>
                  <input
                    type="text"
                    value={sandboxS1Addr}
                    onChange={(e) => setSandboxS1Addr(e.target.value)}
                    className="w-full text-xs font-medium bg-white dark:bg-[#121826] border border-slate-300 dark:border-white/[0.1] rounded-lg px-3 py-2 text-slate-900 dark:text-white focus:outline-none focus:border-violet-500"
                  />
                </div>
              </div>

              {/* Right Input (S2/S3) */}
              <div className="p-4 rounded-xl bg-slate-50 dark:bg-[#090d16] border border-slate-200 dark:border-white/[0.08] space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-cyan-700 dark:text-cyan-400 uppercase tracking-wider flex items-center gap-1.5">
                    <MapPin className="w-3.5 h-3.5" /> Source 2/3 Candidate
                  </span>
                  <select
                    value={sandboxS2Country}
                    onChange={(e) => setSandboxS2Country(e.target.value)}
                    className="text-[11px] font-semibold bg-white dark:bg-[#121826] border border-slate-300 dark:border-white/[0.1] rounded-lg px-2 py-1 text-slate-800 dark:text-slate-200"
                  >
                    <option value="US">🇺🇸 United States</option>
                    <option value="India">🇮🇳 India</option>
                    <option value="France">🇫🇷 France</option>
                  </select>
                </div>
                <div>
                  <label className="text-[10px] uppercase font-bold text-slate-500 dark:text-slate-400 block mb-1">
                    Candidate Name
                  </label>
                  <input
                    type="text"
                    value={sandboxS2Name}
                    onChange={(e) => setSandboxS2Name(e.target.value)}
                    className="w-full text-xs font-medium bg-white dark:bg-[#121826] border border-slate-300 dark:border-white/[0.1] rounded-lg px-3 py-2 text-slate-900 dark:text-white focus:outline-none focus:border-cyan-500"
                  />
                </div>
                <div>
                  <label className="text-[10px] uppercase font-bold text-slate-500 dark:text-slate-400 block mb-1">
                    Candidate Address
                  </label>
                  <input
                    type="text"
                    value={sandboxS2Addr}
                    onChange={(e) => setSandboxS2Addr(e.target.value)}
                    className="w-full text-xs font-medium bg-white dark:bg-[#121826] border border-slate-300 dark:border-white/[0.1] rounded-lg px-3 py-2 text-slate-900 dark:text-white focus:outline-none focus:border-cyan-500"
                  />
                </div>
              </div>
            </div>

            {/* Threshold Slider */}
            <div className="p-4 rounded-xl bg-slate-100/70 dark:bg-white/[0.03] border border-slate-200 dark:border-white/[0.08] mb-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                  <Sliders className="w-3.5 h-3.5 text-violet-600 dark:text-violet-400" />
                  Decision Boundary Threshold (τ*):
                </span>
                <span className="text-xs font-mono font-bold text-violet-700 dark:text-violet-400">
                  {sandboxTau.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.40"
                max="0.95"
                step="0.01"
                value={sandboxTau}
                onChange={(e) => setSandboxTau(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-300 dark:bg-white/[0.1] rounded-lg appearance-none cursor-pointer accent-violet-600"
              />
              <div className="flex justify-between text-[10px] text-slate-500 dark:text-slate-400 mt-1">
                <span>High Recall (0.40)</span>
                <span className="font-bold text-violet-600 dark:text-violet-400">F0.5 Optimal (0.68)</span>
                <span>Ultra Precision (0.95)</span>
              </div>
            </div>

            {/* Result Verdict Card */}
            <div className={`p-6 rounded-2xl border-2 transition-all duration-300 ${
              sandboxResult.verdict === "MERGE"
                ? "border-emerald-400 dark:border-emerald-500/50 bg-emerald-50/70 dark:bg-emerald-950/20"
                : "border-amber-400 dark:border-amber-500/50 bg-amber-50/70 dark:bg-amber-950/20"
            }`}>
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div className="flex items-center gap-4">
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center font-bold text-xl ${
                    sandboxResult.verdict === "MERGE"
                      ? "bg-emerald-100 dark:bg-emerald-500/20 text-emerald-700 dark:text-emerald-400"
                      : "bg-amber-100 dark:bg-amber-500/20 text-amber-700 dark:text-amber-400"
                  }`}>
                    {sandboxResult.verdict === "MERGE" ? "✓" : "🔒"}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-base font-extrabold text-slate-900 dark:text-white">
                        {sandboxResult.verdict === "MERGE" ? "MATCH CONFIRMED (MERGE)" : "SINGLETON PRESERVATION (REJECT)"}
                      </span>
                      <span className={`figma-badge ${
                        sandboxResult.verdict === "MERGE" ? "figma-badge-green" : "figma-badge-amber"
                      }`}>
                        Score = {sandboxResult.matchProb.toFixed(3)}
                      </span>
                    </div>
                    <p className="text-xs text-slate-600 dark:text-slate-300 mt-1">
                      {sandboxResult.verdict === "MERGE"
                        ? `Composite score (${sandboxResult.matchProb.toFixed(3)}) exceeds threshold τ* = ${sandboxTau.toFixed(2)}. Entities refer to the same real-world business.`
                        : `Composite score (${sandboxResult.matchProb.toFixed(3)}) is below threshold τ* = ${sandboxTau.toFixed(2)}. Kept as singleton to prevent false merge penalty.`}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-3 text-center border-t md:border-t-0 md:border-l border-slate-200 dark:border-white/[0.1] pt-3 md:pt-0 md:pl-4">
                  <div>
                    <div className="text-[10px] text-slate-500 dark:text-slate-400 uppercase">Token Sort</div>
                    <div className="text-sm font-bold font-mono text-slate-900 dark:text-white">
                      {(sandboxResult.tokenSort * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-500 dark:text-slate-400 uppercase">Addr Overlap</div>
                    <div className="text-sm font-bold font-mono text-slate-900 dark:text-white">
                      {(sandboxResult.addrSim * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-500 dark:text-slate-400 uppercase">Penalty</div>
                    <div className="text-sm font-bold font-mono text-rose-600 dark:text-rose-400">
                      -{sandboxResult.penalty.toFixed(2)}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── 28-Feature Pairwise Engineering Radar / Matrix ───────────────────── */}
      <div className="glass-panel p-6 space-y-6">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-base font-bold text-slate-900 dark:text-white tracking-tight">
              28 Pairwise Similarity Feature Breakdown
            </h3>
            <span className="figma-badge figma-badge-purple">Explainability Vector</span>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Normalized [0, 1] feature inputs consumed by LightGBM to classify candidate pairs
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Group 1: String Distance Metrics */}
          <div className="p-4 rounded-xl bg-slate-50/80 dark:bg-[#090d16]/90 border border-slate-200 dark:border-white/[0.08] space-y-3">
            <div className="text-xs font-bold text-violet-700 dark:text-violet-400 uppercase tracking-wider flex items-center gap-1.5">
              <span>📏</span> Character Distances
            </div>
            <div className="space-y-2 text-xs">
              <div>
                <div className="flex justify-between text-slate-700 dark:text-slate-300 mb-1">
                  <span>Levenshtein Ratio</span>
                  <span className="font-mono font-bold text-violet-700 dark:text-violet-300">{activePair.features.levenshtein}</span>
                </div>
                <div className="h-1.5 w-full bg-slate-200 dark:bg-white/[0.05] rounded-full overflow-hidden">
                  <div className="h-full bg-violet-600 rounded-full" style={{ width: `${activePair.features.levenshtein * 100}%` }} />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-slate-700 dark:text-slate-300 mb-1">
                  <span>Jaro-Winkler (Prefix 4)</span>
                  <span className="font-mono font-bold text-violet-700 dark:text-violet-300">{activePair.features.jaroWinkler}</span>
                </div>
                <div className="h-1.5 w-full bg-slate-200 dark:bg-white/[0.05] rounded-full overflow-hidden">
                  <div className="h-full bg-violet-600 rounded-full" style={{ width: `${activePair.features.jaroWinkler * 100}%` }} />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-slate-700 dark:text-slate-300 mb-1">
                  <span>Monge-Elkan Hybrid</span>
                  <span className="font-mono font-bold text-violet-700 dark:text-violet-300">{activePair.features.mongeElkan}</span>
                </div>
                <div className="h-1.5 w-full bg-slate-200 dark:bg-white/[0.05] rounded-full overflow-hidden">
                  <div className="h-full bg-violet-600 rounded-full" style={{ width: `${activePair.features.mongeElkan * 100}%` }} />
                </div>
              </div>
            </div>
          </div>

          {/* Group 2: Token Level Features */}
          <div className="p-4 rounded-xl bg-slate-50/80 dark:bg-[#090d16]/90 border border-slate-200 dark:border-white/[0.08] space-y-3">
            <div className="text-xs font-bold text-cyan-700 dark:text-cyan-400 uppercase tracking-wider flex items-center gap-1.5">
              <span>🔤</span> Token Set & Phonetics
            </div>
            <div className="space-y-2 text-xs">
              <div>
                <div className="flex justify-between text-slate-700 dark:text-slate-300 mb-1">
                  <span>Token Set Invariant</span>
                  <span className="font-mono font-bold text-cyan-700 dark:text-cyan-300">{activePair.features.tokenSet}</span>
                </div>
                <div className="h-1.5 w-full bg-slate-200 dark:bg-white/[0.05] rounded-full overflow-hidden">
                  <div className="h-full bg-cyan-600 rounded-full" style={{ width: `${activePair.features.tokenSet * 100}%` }} />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-slate-700 dark:text-slate-300 mb-1">
                  <span>LCS Character Ratio</span>
                  <span className="font-mono font-bold text-cyan-700 dark:text-cyan-300">{activePair.features.lcsRatio}</span>
                </div>
                <div className="h-1.5 w-full bg-slate-200 dark:bg-white/[0.05] rounded-full overflow-hidden">
                  <div className="h-full bg-cyan-600 rounded-full" style={{ width: `${activePair.features.lcsRatio * 100}%` }} />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-slate-700 dark:text-slate-300 mb-1">
                  <span>Token Sort Invariant</span>
                  <span className="font-mono font-bold text-cyan-700 dark:text-cyan-300">{activePair.features.tokenSort}</span>
                </div>
                <div className="h-1.5 w-full bg-slate-200 dark:bg-white/[0.05] rounded-full overflow-hidden">
                  <div className="h-full bg-cyan-600 rounded-full" style={{ width: `${activePair.features.tokenSort * 100}%` }} />
                </div>
              </div>

              <div className="pt-2 border-t border-slate-200 dark:border-white/[0.06] flex items-center justify-between">
                <span className="text-slate-600 dark:text-slate-400">Double Metaphone</span>
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                  activePair.features.metaphoneMatch ? "bg-emerald-100 dark:bg-emerald-500/20 text-emerald-800 dark:text-emerald-300" : "bg-red-100 dark:bg-red-500/20 text-red-800 dark:text-red-300"
                }`}>
                  {activePair.features.metaphoneMatch ? "EXACT PHONETIC" : "DISTINCT"}
                </span>
              </div>
            </div>
          </div>

          {/* Group 3: Address & Spatial Matching */}
          <div className="p-4 rounded-xl bg-slate-50/80 dark:bg-[#090d16]/90 border border-slate-200 dark:border-white/[0.08] space-y-3">
            <div className="text-xs font-bold text-emerald-700 dark:text-emerald-400 uppercase tracking-wider flex items-center gap-1.5">
              <span>📍</span> Address & Postal
            </div>
            <div className="space-y-2 text-xs">
              <div>
                <div className="flex justify-between text-slate-700 dark:text-slate-300 mb-1">
                  <span>Address Token Jaccard</span>
                  <span className="font-mono font-bold text-emerald-700 dark:text-emerald-300">{activePair.features.addressJaccard}</span>
                </div>
                <div className="h-1.5 w-full bg-slate-200 dark:bg-white/[0.05] rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-600 rounded-full" style={{ width: `${activePair.features.addressJaccard * 100}%` }} />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-slate-700 dark:text-slate-300 mb-1">
                  <span>Street Number Exact</span>
                  <span className="font-mono font-bold text-emerald-700 dark:text-emerald-300">{activePair.features.addressDigits}</span>
                </div>
                <div className="h-1.5 w-full bg-slate-200 dark:bg-white/[0.05] rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-600 rounded-full" style={{ width: `${activePair.features.addressDigits * 100}%` }} />
                </div>
              </div>

              <div className="pt-2 border-t border-slate-200 dark:border-white/[0.06] flex items-center justify-between">
                <span className="text-slate-600 dark:text-slate-400">Postal Code Match</span>
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                  activePair.features.postalMatch ? "bg-emerald-100 dark:bg-emerald-500/20 text-emerald-800 dark:text-emerald-300" : "bg-red-100 dark:bg-red-500/20 text-red-800 dark:text-red-300"
                }`}>
                  {activePair.features.postalMatch ? "MATCHED" : "MISMATCH"}
                </span>
              </div>
            </div>
          </div>

          {/* Group 4: Dense Vector Embedding */}
          <div className="p-4 rounded-xl bg-slate-50/80 dark:bg-[#090d16]/90 border border-slate-200 dark:border-white/[0.08] space-y-3">
            <div className="text-xs font-bold text-amber-700 dark:text-amber-400 uppercase tracking-wider flex items-center gap-1.5">
              <span>🧠</span> Dense Semantic Vector
            </div>
            <div className="space-y-2 text-xs">
              <div>
                <div className="flex justify-between text-slate-700 dark:text-slate-300 mb-1">
                  <span>Bi-Encoder Cosine Sim</span>
                  <span className="font-mono font-bold text-amber-700 dark:text-amber-300">{activePair.features.denseCosine}</span>
                </div>
                <div className="h-1.5 w-full bg-slate-200 dark:bg-white/[0.05] rounded-full overflow-hidden">
                  <div className="h-full bg-amber-600 rounded-full" style={{ width: `${activePair.features.denseCosine * 100}%` }} />
                </div>
              </div>

              <div className="pt-2 border-t border-slate-200 dark:border-white/[0.06] text-[11px] text-slate-600 dark:text-slate-400 leading-snug">
                Embeddings generated via <code className="text-violet-700 dark:text-violet-300 font-mono text-[10px]">all-MiniLM-L6-v2</code> capturing multilingual semantic proximity across noise.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
