"use client";

import React from "react";
import { Sun, Moon } from "lucide-react";
import { useTheme } from "./ThemeProvider";

export function ThemeToggle({ className = "" }: { className?: string }) {
  const { theme, toggleTheme } = useTheme();

  return (
    <button
      onClick={toggleTheme}
      type="button"
      aria-label="Toggle Day / Night theme"
      title={theme === "light" ? "Switch to Night theme (Dark)" : "Switch to Day theme (Light)"}
      className={`relative inline-flex items-center gap-1.5 p-1.5 px-2.5 rounded-xl border transition-all duration-200 cursor-pointer ${
        theme === "light"
          ? "bg-slate-100 hover:bg-slate-200 border-slate-300 text-slate-800 shadow-sm"
          : "bg-white/[0.06] hover:bg-white/[0.12] border-white/[0.12] text-amber-300 shadow-inner"
      } ${className}`}
    >
      {theme === "light" ? (
        <>
          <Sun className="w-4 h-4 text-amber-500 animate-spin-slow" />
          <span className="text-xs font-semibold text-slate-700">Day Theme</span>
        </>
      ) : (
        <>
          <Moon className="w-4 h-4 text-violet-300" />
          <span className="text-xs font-semibold text-slate-200">Night Theme</span>
        </>
      )}
    </button>
  );
}
