#!/usr/bin/env python3
"""
benchmark_holdout_blocking_and_pruning.py — Phase 6 & 7 Blocker and Pruning Engine.
Evaluates individual blocking channels and candidate pruning depths (K in [50, 75, 100, 130, 150, 200])
on the STRICT 30,000 Grouped Final Holdout.
Saves:
  - reports/recovery/blocker_ablation_holdout.csv
  - reports/recovery/pruning_benchmark_holdout.csv
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DIAG_DIR = ROOT / "data" / "diagnostic_200k"
RECOVERY_DIR = ROOT / "reports" / "recovery"
RECOVERY_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BlockerHoldout")

RE_PUNCT = re.compile(r"[^a-z0-9\s]")
RE_DIGITS = re.compile(r"\b\d+\b")
RE_ALPHA = re.compile(r"[^a-z0-9]")

def fast_clean(s: str) -> str:
    if not s: return ""
    return RE_PUNCT.sub(" ", str(s).lower()).strip()

def fast_alpha(s: str) -> str:
    if not s: return ""
    return RE_ALPHA.sub("", str(s).lower()).strip()

def get_char_ngrams(text: str, n: int) -> set:
    s = text.replace(" ", "")
    if len(s) < n: return set()
    return {s[i:i+n] for i in range(len(s)-n+1)}

class CompactCand:
    __slots__ = ("eid", "clean_name", "clean_addr", "alpha_name", "name_tokens")
    def __init__(self, eid: str, clean_name: str, clean_addr: str, alpha_name: str, name_tokens: tuple):
        self.eid = eid
        self.clean_name = clean_name
        self.clean_addr = clean_addr
        self.alpha_name = alpha_name
        self.name_tokens = name_tokens

def main():
    logger.info("=" * 80)
    logger.info("PHASE 6 & 7: BLOCKER ABLATION & RANKED PRUNING BENCHMARK ON STRICT 30K HOLDOUT")
    logger.info("=" * 80)

    # 1. Load 30K Holdout IDs
    with open(DIAG_DIR / "split_holdout_30k_s1_ids.json", "r", encoding="utf-8") as f:
        holdout_ids = json.load(f)
    holdout_set = set(holdout_ids)

    # 2. Load Ground Truth
    gt = defaultdict(set)
    total_true_links = 0
    with open(DIAG_DIR / "ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[1]:
                m_set = set(x.strip() for x in parts[1].split(",") if x.strip())
                gt[parts[0]] = m_set
                if parts[0] in holdout_set:
                    total_true_links += len(m_set)

    logger.info("Holdout entities: %d | Total true links: %d", len(holdout_ids), total_true_links)

    # 3. Load S1 records
    s1_dict = {}
    with open(DIAG_DIR / "source1.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4 and parts[0] in holdout_set:
                eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                cn = fast_clean(name)
                ca = fast_clean(addr)
                alpha = fast_alpha(name)
                s1_dict[eid] = {
                    "id": eid, "clean_name": cn, "clean_addr": ca, "alpha_name": alpha,
                    "country": country, "name_tokens": set(cn.split()), "addr_tokens": set(ca.split()),
                    "digits": set(RE_DIGITS.findall(ca)),
                    "ng3": get_char_ngrams(cn, 3), "ng4": get_char_ngrams(cn, 4)
                }

    # 4. Load S2/S3 records into CompactCand
    logger.info("Loading S2/S3 Candidate records...")
    cands_by_country = defaultdict(dict)
    for src in ["source2.tsv", "source3.tsv"]:
        with open(DIAG_DIR / src, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    cid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                    cn = fast_clean(name)
                    ca = fast_clean(addr)
                    alpha = fast_alpha(name)
                    cands_by_country[country][cid] = CompactCand(
                        eid=cid, clean_name=cn, clean_addr=ca, alpha_name=alpha,
                        name_tokens=tuple(cn.split())
                    )

    # 5. Build Comprehensive Multi-Channel Indexes
    logger.info("Building Multi-Channel Inverted Indexes...")
    indexes = {}
    for country, pool in cands_by_country.items():
        alpha_idx = defaultdict(list)
        token_idx = defaultdict(list)
        rare_idx = defaultdict(list)
        addr_idx = defaultdict(list)
        digit_idx = defaultdict(list)
        ng3_idx = defaultdict(list)
        ng4_idx = defaultdict(list)

        token_counts = defaultdict(int)
        for c in pool.values():
            for t in c.name_tokens: token_counts[t] += 1

        for cid, c in pool.items():
            if c.alpha_name and len(c.alpha_name) >= 3:
                p = alpha_idx[c.alpha_name]
                if len(p) < 150: p.append(cid)

            for t in c.name_tokens:
                if len(t) >= 3:
                    freq = token_counts[t]
                    p = token_idx[t]
                    if len(p) < (60 if freq > 200 else 150): p.append(cid)
                    if freq < 40:
                        p_r = rare_idx[t]
                        if len(p_r) < 80: p_r.append(cid)

            for at in c.clean_addr.split():
                if len(at) >= 3:
                    p = addr_idx[at]
                    if len(p) < 80: p.append(cid)

            for d in RE_DIGITS.findall(c.clean_addr):
                p = digit_idx[d]
                if len(p) < 80: p.append(cid)

            for ng in get_char_ngrams(c.clean_name, 3):
                p = ng3_idx[ng]
                if len(p) < 60: p.append(cid)

            for ng in get_char_ngrams(c.clean_name, 4):
                p = ng4_idx[ng]
                if len(p) < 60: p.append(cid)

        indexes[country] = {
            "alpha": alpha_idx,
            "token": token_idx,
            "rare": rare_idx,
            "addr": addr_idx,
            "digit": digit_idx,
            "ng3": ng3_idx,
            "ng4": ng4_idx
        }
        logger.info("[%s] Indexed: alpha=%d, token=%d, rare=%d, addr=%d, digit=%d, ng3=%d, ng4=%d",
                    country, len(alpha_idx), len(token_idx), len(rare_idx),
                    len(addr_idx), len(digit_idx), len(ng3_idx), len(ng4_idx))

    # Phase 6: Measure each channel individually on 30K holdout
    logger.info("=" * 80)
    logger.info("MEASURING INDIVIDUAL BLOCKING CHANNELS ON 30K HOLDOUT")
    logger.info("=" * 80)

    channels = [
        ("1. Exact Alpha Name", ["alpha"]),
        ("2. Name Token Overlap", ["token"]),
        ("3. Rare Name Tokens", ["rare"]),
        ("4. Address Tokens", ["addr"]),
        ("5. Numeric Digits (Postal/Street)", ["digit"]),
        ("6. Character 3-Grams", ["ng3"]),
        ("7. Character 4-Grams", ["ng4"]),
        ("8. Multi-Channel Union (All Combined)", ["alpha", "token", "rare", "addr", "digit", "ng3", "ng4"]),
    ]

    channel_results = []
    union_raw_cands = {}

    for ch_name, ch_keys in channels:
        t_start = time.time()
        hits = 0
        perfect_entities = 0
        non_singleton_entities = 0
        total_cands_retrieved = 0

        for eid in holdout_ids:
            s1 = s1_dict[eid]
            c = s1["country"]
            idx = indexes[c]
            true_m = gt.get(eid, set())
            is_non_singleton = len(true_m) > 0

            if is_non_singleton:
                non_singleton_entities += 1

            retrieved = set()
            if "alpha" in ch_keys and s1["alpha_name"]:
                retrieved.update(idx["alpha"].get(s1["alpha_name"], []))
            if "token" in ch_keys:
                for t in s1["name_tokens"]:
                    if len(t) >= 3: retrieved.update(idx["token"].get(t, []))
            if "rare" in ch_keys:
                for t in s1["name_tokens"]:
                    if len(t) >= 3: retrieved.update(idx["rare"].get(t, []))
            if "addr" in ch_keys:
                for at in s1["addr_tokens"]:
                    if len(at) >= 3: retrieved.update(idx["addr"].get(at, []))
            if "digit" in ch_keys:
                for d in s1["digits"]:
                    retrieved.update(idx["digit"].get(d, []))
            if "ng3" in ch_keys:
                for ng in s1["ng3"]:
                    retrieved.update(idx["ng3"].get(ng, []))
            if "ng4" in ch_keys:
                for ng in s1["ng4"]:
                    retrieved.update(idx["ng4"].get(ng, []))

            total_cands_retrieved += len(retrieved)

            if is_non_singleton:
                matched_hits = len(retrieved & true_m)
                hits += matched_hits
                if matched_hits == len(true_m):
                    perfect_entities += 1

            if ch_keys == ["alpha", "token", "rare", "addr", "digit", "ng3", "ng4"]:
                union_raw_cands[eid] = retrieved

        pair_recall = hits / total_true_links if total_true_links else 0.0
        perfect_recall = perfect_entities / non_singleton_entities if non_singleton_entities else 0.0
        avg_cands = total_cands_retrieved / len(holdout_ids)
        elapsed = time.time() - t_start

        logger.info("%-40s | Pair Rec: %.2f%% | Entity Rec: %.2f%% | Avg Cands: %.1f | Time: %.2fs",
                    ch_name, pair_recall * 100, perfect_recall * 100, avg_cands, elapsed)

        channel_results.append({
            "Channel": ch_name,
            "Pair_Recall": round(pair_recall * 100, 2),
            "Entity_Perfect_Recall": round(perfect_recall * 100, 2),
            "Avg_Candidates": round(avg_cands, 1),
            "Total_Hits": hits,
            "Runtime_Seconds": round(elapsed, 2)
        })

    pd.DataFrame(channel_results).to_csv(RECOVERY_DIR / "blocker_ablation_holdout.csv", index=False)
    logger.info("Saved channel ablation to %s", RECOVERY_DIR / "blocker_ablation_holdout.csv")

    # Phase 7: Candidate Pruning Benchmark across K in [50, 75, 100, 130, 150, 200]
    logger.info("=" * 80)
    logger.info("PHASE 7: CANDIDATE PRUNING BENCHMARK (K in [50, 75, 100, 130, 150, 200])")
    logger.info("=" * 80)

    # Pre-rank all union candidates once using composite similarity
    logger.info("Ranking union candidates using composite similarity...")
    ranked_candidates_by_s1 = {}
    for eid in holdout_ids:
        s1 = s1_dict[eid]
        c = s1["country"]
        pool = cands_by_country[c]
        raw_set = union_raw_cands.get(eid, set())

        s1_alpha = s1["alpha_name"]
        s1_nt = s1["name_tokens"]
        s1_at = s1["addr_tokens"]
        s1_digits = s1["digits"]

        scored = []
        for cid in raw_set:
            cand = pool.get(cid)
            if not cand: continue
            score = 0.0
            if s1_alpha and cand.alpha_name and s1_alpha == cand.alpha_name:
                score += 5.0
            c_nt = set(cand.name_tokens)
            inter_n = len(s1_nt & c_nt)
            if inter_n:
                score += 3.0 * (inter_n / (len(s1_nt) + len(c_nt) - inter_n))
            c_at = set(cand.clean_addr.split())
            inter_a = len(s1_at & c_at)
            if inter_a:
                score += 2.0 * (inter_a / (len(s1_at) + len(c_at) - inter_a))
            c_digits = set(RE_DIGITS.findall(cand.clean_addr))
            if s1_digits & c_digits:
                score += 2.0

            scored.append((score, cid))

        scored.sort(key=lambda x: x[0], reverse=True)
        ranked_candidates_by_s1[eid] = [cid for _, cid in scored]

    k_values = [50, 75, 100, 130, 150, 200]
    pruning_results = []

    for k in k_values:
        hits = 0
        perfect = 0
        non_sing = 0
        total_k_cands = 0

        for eid in holdout_ids:
            true_m = gt.get(eid, set())
            is_non_sing = len(true_m) > 0
            if is_non_sing: non_sing += 1

            top_k_set = set(ranked_candidates_by_s1[eid][:k])
            total_k_cands += len(top_k_set)

            if is_non_sing:
                m_hits = len(top_k_set & true_m)
                hits += m_hits
                if m_hits == len(true_m):
                    perfect += 1

        pair_rec = hits / total_true_links if total_true_links else 0.0
        ent_rec = perfect / non_sing if non_sing else 0.0

        logger.info("K = %-3d | Pair Recall: %.2f%% (%d / %d) | Entity-Perfect: %.2f%% | Avg Cands: %.1f",
                    k, pair_rec * 100, hits, total_true_links, ent_rec * 100, total_k_cands / len(holdout_ids))

        pruning_results.append({
            "K": k,
            "Pair_Recall": round(pair_rec * 100, 2),
            "Entity_Perfect_Recall": round(ent_rec * 100, 2),
            "Avg_Candidates": round(total_k_cands / len(holdout_ids), 1),
            "Total_Hits": hits,
            "Total_True": total_true_links
        })

    pd.DataFrame(pruning_results).to_csv(RECOVERY_DIR / "pruning_benchmark_holdout.csv", index=False)
    logger.info("Saved pruning benchmark to %s", RECOVERY_DIR / "pruning_benchmark_holdout.csv")

if __name__ == "__main__":
    main()
