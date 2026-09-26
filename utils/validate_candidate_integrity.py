#!/usr/bin/env python3
"""
ML Challenge 2026 — Memory-Safe Disk-Backed Candidate Integrity Validator

Verifies that for every S1 entity:
1. candidate_pairs has exactly 1,732,544 S1 rows
2. matching_results has exactly 1,732,544 S1 rows
3. No duplicate S1 rows in either file
4. No missing or extraneous S1 IDs relative to test_source1.tsv
5. No duplicate candidate IDs within a row (intra-list)
6. No duplicate matched IDs within a row (intra-list)
7. Matched IDs have valid S2-/S3- prefixes (no self-matches)
8. Candidate IDs have valid S2-/S3- prefixes (no self-matches)
9. Every matched_entity_id ⊆ candidate_entity_ids for that exact S1 entity

Uses a temporary SQLite database on disk for S1 index/coverage verification.
Peak RAM: < 150 MB (far below 2 GB limit).
Runtime: ~30-60 seconds.
Temporary database is deleted immediately after completion.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

# Ensure UTF-8 output even on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DELIM = "\t"
EXPECTED_S1_COUNT = 1_732_544
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def find_test_source1() -> str:
    candidates = [
        "data_raw/student_resource/dataset/test/test_source1.tsv",
        "dataset/test/test_source1.tsv",
        "data/test/test_source1.tsv",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"test_source1.tsv not found in any of: {candidates}")


def main() -> int:
    matching_path = "output/matching_results.tsv"
    candidate_path = "output/candidate_pairs.tsv"

    if not os.path.isfile(matching_path):
        print(f"FATAL: Missing {matching_path}")
        return 1
    if not os.path.isfile(candidate_path):
        print(f"FATAL: Missing {candidate_path}")
        return 1

    try:
        test_s1_path = find_test_source1()
    except FileNotFoundError as e:
        print(f"FATAL: {e}")
        return 1

    tracemalloc.start()
    t_start = time.time()

    # -------------------------------------------------------------
    # 1. Create temporary SQLite database on disk
    # -------------------------------------------------------------
    db_fd, db_path = tempfile.mkstemp(suffix=".db", prefix="val_candidate_integrity_")
    os.close(db_fd)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA synchronous = OFF")
    cur.execute("PRAGMA journal_mode = MEMORY")
    cur.execute("PRAGMA temp_store = MEMORY")

    cur.execute("CREATE TABLE expected_s1 (id TEXT PRIMARY KEY)")
    cur.execute("CREATE TABLE s1_cand     (id TEXT, line_num INTEGER)")
    cur.execute("CREATE TABLE s1_match    (id TEXT, line_num INTEGER)")
    conn.commit()

    # -------------------------------------------------------------
    # 2. Ingest expected S1 IDs from test_source1.tsv
    # -------------------------------------------------------------
    print(f"Loading expected S1 IDs from {test_s1_path}...")
    batch_exp = []
    with open(test_s1_path, "r", encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            parts = line.split(DELIM, 1)
            if parts and parts[0].strip():
                batch_exp.append((parts[0].strip(),))
                if len(batch_exp) >= 100_000:
                    cur.executemany("INSERT OR IGNORE INTO expected_s1 VALUES (?)", batch_exp)
                    batch_exp = []
        if batch_exp:
            cur.executemany("INSERT OR IGNORE INTO expected_s1 VALUES (?)", batch_exp)
            batch_exp = []
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM expected_s1")
    n_expected = cur.fetchone()[0]
    print(f"Expected unique S1 entities: {n_expected:,}")

    # -------------------------------------------------------------
    # 3. Stream candidate_pairs and matching_results in lockstep
    # -------------------------------------------------------------
    print(f"\nStreaming & validating {matching_path} and {candidate_path}...")

    # Tracking metrics
    # Candidate integrity
    c_row_count = 0
    c_intra_dupes = 0
    c_bad_prefixes = 0
    c_bad_prefix_examples: list[str] = []

    # Match format
    m_row_count = 0
    m_empty_count = 0
    m_nonempty_count = 0
    m_intra_dupes = 0
    m_bad_prefixes = 0
    m_bad_prefix_examples: list[str] = []

    # Match subset of candidates
    mismatch_examples: list[tuple[str, str, str]] = []  # (s1, mid, missing_cand)
    total_mismatches = 0
    mismatched_s1_count = 0

    batch_cand_s1 = []
    batch_match_s1 = []
    BATCH_SIZE = 50_000

    header_c_ok = True
    header_m_ok = True

    with open(candidate_path, "r", encoding="utf-8") as f_c, \
         open(matching_path, "r", encoding="utf-8") as f_m:

        # Header verification
        h_c = f_c.readline()
        h_m = f_m.readline()

        cols_c = [c.strip().lower() for c in h_c.rstrip("\r\n").split(DELIM)]
        cols_m = [c.strip().lower() for c in h_m.rstrip("\r\n").split(DELIM)]

        if cols_c != CANDIDATE_HEADER:
            header_c_ok = False
            print(f"ERROR: candidate header invalid: {cols_c}, expected {CANDIDATE_HEADER}")
        if cols_m != MATCHING_HEADER:
            header_m_ok = False
            print(f"ERROR: matching header invalid: {cols_m}, expected {MATCHING_HEADER}")

        line_num = 1
        while True:
            line_c = f_c.readline()
            line_m = f_m.readline()

            if not line_c and not line_m:
                break
            line_num += 1

            # Candidate row parsing
            s1_c = None
            cand_ids = []
            if line_c:
                c_row_count += 1
                s1_c, tab_c, rest_c = line_c.partition(DELIM)
                s1_c = s1_c.strip()
                batch_cand_s1.append((s1_c, line_num))

                raw_cand = rest_c.rstrip("\r\n")
                if raw_cand:
                    cand_ids = [x.strip() for x in raw_cand.split(",") if x.strip()]

                # Intra-list dupes
                if len(cand_ids) != len(set(cand_ids)):
                    c_intra_dupes += 1

                # Prefix check
                for cid in cand_ids:
                    if not cid.startswith(("S2-", "S3-")):
                        c_bad_prefixes += 1
                        if len(c_bad_prefix_examples) < 10:
                            c_bad_prefix_examples.append(f"{s1_c} -> {cid}")

            # Match row parsing
            s1_m = None
            match_ids = []
            if line_m:
                m_row_count += 1
                s1_m, tab_m, rest_m = line_m.partition(DELIM)
                s1_m = s1_m.strip()
                batch_match_s1.append((s1_m, line_num))

                raw_m = rest_m.rstrip("\r\n")
                if not raw_m:
                    m_empty_count += 1
                else:
                    m_nonempty_count += 1
                    match_ids = [x.strip() for x in raw_m.split(",") if x.strip()]

                    # Intra-list dupes
                    if len(match_ids) != len(set(match_ids)):
                        m_intra_dupes += 1

                    # Prefix check
                    for mid in match_ids:
                        if not mid.startswith(("S2-", "S3-")):
                            m_bad_prefixes += 1
                            if len(m_bad_prefix_examples) < 10:
                                m_bad_prefix_examples.append(f"{s1_m} -> {mid}")

            # Verification: match ⊆ candidates for this row
            if s1_m and s1_c and s1_m == s1_c and match_ids:
                cand_set = set(cand_ids)
                has_row_mismatch = False
                for mid in match_ids:
                    if mid not in cand_set:
                        total_mismatches += 1
                        has_row_mismatch = True
                        if len(mismatch_examples) < 20:
                            mismatch_examples.append((s1_m, mid, f"NOT IN CANDIDATES (cand_count={len(cand_ids)})"))
                if has_row_mismatch:
                    mismatched_s1_count += 1
            elif s1_m != s1_c and (s1_m or s1_c):
                # Row alignment divergence
                total_mismatches += 1
                if len(mismatch_examples) < 20:
                    mismatch_examples.append((
                        s1_m or "<missing>",
                        "ALIGNMENT_MISMATCH",
                        f"Candidate S1={s1_c or '<missing>'}"
                    ))

            # Batch insert S1 into SQLite
            if len(batch_cand_s1) >= BATCH_SIZE:
                cur.executemany("INSERT INTO s1_cand VALUES (?, ?)", batch_cand_s1)
                batch_cand_s1 = []
            if len(batch_match_s1) >= BATCH_SIZE:
                cur.executemany("INSERT INTO s1_match VALUES (?, ?)", batch_match_s1)
                batch_match_s1 = []

            if c_row_count % 500_000 == 0:
                print(f"  ... verified {c_row_count:,} rows")

    if batch_cand_s1:
        cur.executemany("INSERT INTO s1_cand VALUES (?, ?)", batch_cand_s1)
        batch_cand_s1 = []
    if batch_match_s1:
        cur.executemany("INSERT INTO s1_match VALUES (?, ?)", batch_match_s1)
        batch_match_s1 = []
    conn.commit()

    print(f"Stream verification complete: {c_row_count:,} candidate rows, {m_row_count:,} match rows.")

    # -------------------------------------------------------------
    # 4. Disk-Backed SQL Verification of S1 Coverage & Uniqueness
    # -------------------------------------------------------------
    print("Running SQL coverage & uniqueness verification...")

    cur.execute("CREATE INDEX idx_s1_cand ON s1_cand (id)")
    cur.execute("CREATE INDEX idx_s1_match ON s1_match (id)")
    conn.commit()

    # Distinct S1 counts
    cur.execute("SELECT COUNT(DISTINCT id) FROM s1_cand")
    n_distinct_cand = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT id) FROM s1_match")
    n_distinct_match = cur.fetchone()[0]

    # Duplicate S1 rows in candidate_pairs
    cur.execute("SELECT id, COUNT(*) FROM s1_cand GROUP BY id HAVING COUNT(*) > 1 LIMIT 10")
    cand_dups = cur.fetchall()

    # Duplicate S1 rows in matching_results
    cur.execute("SELECT id, COUNT(*) FROM s1_match GROUP BY id HAVING COUNT(*) > 1 LIMIT 10")
    match_dups = cur.fetchall()

    # S1 in expected but missing from candidate_pairs
    cur.execute("""
        SELECT e.id FROM expected_s1 e
        LEFT JOIN s1_cand c ON e.id = c.id
        WHERE c.id IS NULL
        LIMIT 10
    """)
    missing_cand_s1 = [r[0] for r in cur.fetchall()]

    # S1 in candidate_pairs but not in expected
    cur.execute("""
        SELECT c.id FROM s1_cand c
        LEFT JOIN expected_s1 e ON c.id = e.id
        WHERE e.id IS NULL
        LIMIT 10
    """)
    extra_cand_s1 = [r[0] for r in cur.fetchall()]

    # S1 in expected but missing from matching_results
    cur.execute("""
        SELECT e.id FROM expected_s1 e
        LEFT JOIN s1_match m ON e.id = m.id
        WHERE m.id IS NULL
        LIMIT 10
    """)
    missing_match_s1 = [r[0] for r in cur.fetchall()]

    # S1 in matching_results but not in expected
    cur.execute("""
        SELECT m.id FROM s1_match m
        LEFT JOIN expected_s1 e ON m.id = e.id
        WHERE e.id IS NULL
        LIMIT 10
    """)
    extra_match_s1 = [r[0] for r in cur.fetchall()]

    # -------------------------------------------------------------
    # 5. Clean up temporary SQLite database
    # -------------------------------------------------------------
    conn.close()
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
    except OSError:
        pass

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = time.time() - t_start

    # -------------------------------------------------------------
    # 6. Evaluate Results & Print Final Report
    # -------------------------------------------------------------
    # CANDIDATE INTEGRITY
    cand_integrity_pass = (
        header_c_ok
        and c_row_count == EXPECTED_S1_COUNT
        and len(cand_dups) == 0
        and len(missing_cand_s1) == 0
        and len(extra_cand_s1) == 0
        and c_intra_dupes == 0
        and c_bad_prefixes == 0
    )

    # S1 COVERAGE
    s1_coverage_pass = (
        c_row_count == EXPECTED_S1_COUNT
        and m_row_count == EXPECTED_S1_COUNT
        and n_distinct_cand == EXPECTED_S1_COUNT
        and n_distinct_match == EXPECTED_S1_COUNT
        and len(missing_cand_s1) == 0
        and len(extra_cand_s1) == 0
        and len(missing_match_s1) == 0
        and len(extra_match_s1) == 0
    )

    # MATCH FORMAT
    match_format_pass = (
        header_m_ok
        and m_row_count == EXPECTED_S1_COUNT
        and len(match_dups) == 0
        and len(missing_match_s1) == 0
        and len(extra_match_s1) == 0
        and m_intra_dupes == 0
        and m_bad_prefixes == 0
    )

    # MATCH ⊆ CANDIDATES
    match_subset_pass = (total_mismatches == 0)

    print()
    print("=" * 64)
    print("      FINAL PRE-SUBMISSION INTEGRITY REPORT")
    print("=" * 64)
    print()
    print(f"CANDIDATE INTEGRITY:    {'PASS' if cand_integrity_pass else 'FAIL'}")
    print(f"S1 COVERAGE:            {'PASS' if s1_coverage_pass else 'FAIL'}")
    print(f"MATCH FORMAT:           {'PASS' if match_format_pass else 'FAIL'}")
    print(f"MATCH ⊆ CANDIDATES:     {'PASS' if match_subset_pass else 'FAIL'}")
    print(f"PEAK RAM:               {peak_mem / (1024 * 1024):.2f} MB")
    print(f"RUNTIME:                {elapsed:.2f} s")
    print()

    # Detailed statistics
    print("Detailed Statistics:")
    print(f"  Expected S1 count:         {EXPECTED_S1_COUNT:,}")
    print(f"  Candidate rows:            {c_row_count:,} (distinct: {n_distinct_cand:,})")
    print(f"  Matching rows:             {m_row_count:,} (distinct: {n_distinct_match:,})")
    print(f"  Empty predictions:         {m_empty_count:,}")
    print(f"  Non-empty predictions:     {m_nonempty_count:,} ({m_nonempty_count / m_row_count * 100:.2f}%)")
    print(f"  Candidate intra-list dups: {c_intra_dupes}")
    print(f"  Match intra-list dups:     {m_intra_dupes}")
    print(f"  Candidate bad prefixes:    {c_bad_prefixes}")
    print(f"  Match bad prefixes:        {m_bad_prefixes}")
    print(f"  Candidate duplicate rows:  {len(cand_dups)}")
    print(f"  Match duplicate rows:      {len(match_dups)}")
    print(f"  Missing S1 in candidates:  {len(missing_cand_s1)}")
    print(f"  Missing S1 in matches:     {len(missing_match_s1)}")
    print(f"  Mismatched S1 rows:        {mismatched_s1_count}")
    print(f"  Total subset violations:   {total_mismatches}")

    if mismatch_examples:
        print()
        print("Mismatches found (up to 20 examples):")
        print(f"{'S1 ID':<20} {'Matched ID':<20} {'Missing Candidate ID / Detail':<30}")
        print("-" * 70)
        for s1, mid, detail in mismatch_examples[:20]:
            print(f"{s1:<20} {mid:<20} {detail:<30}")

    all_passed = (
        cand_integrity_pass
        and s1_coverage_pass
        and match_format_pass
        and match_subset_pass
    )
    print()
    if all_passed:
        print("ALL CRITICAL CONDITIONS VERIFIED: SUBMISSION IS READY AND SAFE.")
        return 0
    else:
        print("VALIDATION FAILED: Issues must be resolved before submission.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
