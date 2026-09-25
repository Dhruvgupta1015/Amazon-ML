"""validator.py - Submission Validator.

Mirrors utils/validate_submission.py logic, integrated into the backend
so the API can run validation checks and return structured results to the dashboard.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

DELIM = "\t"
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _read_ids(path: str) -> set[str]:
    """Return set of first-column entity IDs from a source TSV."""
    ids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if line:
                ids.add(line.split(DELIM, 1)[0].strip())
    return ids


def _validate_id_list_file(
    path: str,
    expected_header: list[str],
    col_label: str,
    required: set[str],
    valid_ids: Optional[set[str]],
    errors: list[str],
    warnings: list[str],
) -> Optional[dict[str, set[str]]]:
    """Validate one TSV submission file."""
    if not os.path.isfile(path):
        errors.append(f"File not found: {path}")
        return None

    name = os.path.basename(path)
    mapping: dict[str, set[str]] = {}
    seen: set[str] = set()
    dup_rows: set[str] = set()
    intra_dupes: set[str] = set()
    self_matches: set[str] = set()
    wrong_prefix: set[str] = set()
    unknown: set[str] = set()
    n_rows = empties = 0

    with open(path, encoding="utf-8") as f:
        header = f.readline()
        if not header:
            errors.append(f"{name} is empty.")
            return None

        if DELIM not in header and "," in header:
            errors.append(
                f"{name}: header has no TAB but has commas -- looks CSV. "
                "Submissions must be TAB-separated (.tsv)."
            )
            return None

        cols = [c.strip().lower() for c in header.rstrip("\n").split(DELIM)]
        if cols != expected_header:
            errors.append(
                f"{name}: unexpected header {cols}. "
                f"Expected exactly {expected_header} (tab-separated)."
            )
            return None

        for line_num, line in enumerate(f, start=2):
            s1, tab, rest = line.partition(DELIM)
            if not tab:
                if s1.strip():
                    errors.append(
                        f"{name}: malformed row (no tab) at line {line_num}: {line.rstrip()!r}"
                    )
                continue

            n_rows += 1
            if s1 in seen:
                dup_rows.add(s1)
            seen.add(s1)

            ids = rest.rstrip("\n").split(",") if rest.strip() else []
            if not ids:
                empties += 1
                mapping[s1] = set()
                continue

            if len(ids) != len(set(ids)):
                intra_dupes.add(s1)

            id_set = set(ids)
            mapping[s1] = id_set

            for mid in id_set:
                if mid.startswith("S1-"):
                    self_matches.add(mid)
                elif not mid.startswith(("S2-", "S3-")):
                    wrong_prefix.add(mid)
                elif valid_ids is not None and mid not in valid_ids:
                    unknown.add(mid)

    def _ex(items: set[str]) -> str:
        sorted_items = sorted(items)
        shown = ", ".join(sorted_items[:5])
        return f"{len(items)} total, e.g. {shown}, ..." if len(items) > 5 else shown

    findings = [
        (dup_rows, f"{name}: duplicate source1_entity_id rows: {_ex(dup_rows)}"),
        (intra_dupes, f"{name}: repeated ID inside {col_label} list for: {_ex(intra_dupes)}"),
        (self_matches, f"{name}: {col_label} contains S1-IDs (self-matches): {_ex(self_matches)}"),
        (wrong_prefix, f"{name}: {col_label} contains IDs without S2-/S3- prefix: {_ex(wrong_prefix)}"),
        (unknown, f"{name}: {col_label} references IDs not in test set: {_ex(unknown)}"),
        (required - seen, f"{name}: required S1 entities missing: {_ex(required - seen)}"),
        (seen - required, f"{name}: rows with S1 ID not in test set: {_ex(seen - required)}"),
    ]
    for offenders, message in findings:
        if offenders:
            errors.append(message)

    return mapping


def validate_submission(
    matching_path: str,
    candidate_path: Optional[str],
    test_source1_path: str,
    test_source2_path: Optional[str] = None,
    test_source3_path: Optional[str] = None,
    check_ids: bool = False,
) -> ValidationResult:
    """Full submission validation. Returns structured ValidationResult."""
    errors: list[str] = []
    warnings: list[str] = []

    if not os.path.isfile(test_source1_path):
        return ValidationResult(
            passed=False,
            errors=[f"test_source1 not found: {test_source1_path}"]
        )

    required = _read_ids(test_source1_path)

    valid_ids: Optional[set[str]] = None
    if check_ids and test_source2_path and test_source3_path:
        valid_ids = set()
        for p in [test_source2_path, test_source3_path]:
            if os.path.isfile(p):
                valid_ids |= _read_ids(p)
    else:
        warnings.append(
            "ID-existence check OFF -- matched IDs not verified against test S2/S3 files. "
            "Pass check_ids=True to enable (memory-intensive)."
        )

    matched = _validate_id_list_file(
        matching_path, MATCHING_HEADER, "matched_entity_ids",
        required, valid_ids, errors, warnings,
    )

    candidate = None
    if candidate_path and os.path.isfile(candidate_path):
        candidate = _validate_id_list_file(
            candidate_path, CANDIDATE_HEADER, "candidate_entity_ids",
            required, valid_ids, errors, warnings,
        )
    elif candidate_path:
        warnings.append(
            f"{candidate_path} not found -- skipping candidate_pairs.tsv checks. "
            "Final submission zip must include output/candidate_pairs.tsv."
        )

    if matched is not None and candidate is not None:
        offenders = {
            s1 for s1, mids in matched.items() if mids - candidate.get(s1, set())
        }
        if offenders:
            warnings.append(
                f"{len(offenders)} S1 entities have matched IDs not in candidate_pairs.tsv. "
                "Final matches normally come from your blocking candidates."
            )

    stats = {
        "required_s1_count": len(required),
        "valid_ids_count": len(valid_ids) if valid_ids else None,
        "n_errors": len(errors),
        "n_warnings": len(warnings),
    }

    return ValidationResult(passed=len(errors) == 0, errors=errors, warnings=warnings, stats=stats)
