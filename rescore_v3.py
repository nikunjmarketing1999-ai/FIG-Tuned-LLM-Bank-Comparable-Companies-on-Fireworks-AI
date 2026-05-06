"""
rescore_v3.py — FIG analyst-grade rubric applied to existing eval data

WHY THIS EXISTS
---------------
The v2 rubric had a critical flaw: it counted "EV/EBITDA: N/A — not applicable
for banks" as a violation because the literal string "EV/EBITDA" appeared. A
real FIG analyst would score that as a CORRECT ANSWER — the model is properly
rejecting an industrials multiple.

This script re-scores the same model outputs with a 5-axis industry-grade
rubric. It does NOT call any APIs. It does NOT cost money. It re-reads
eval_results_v2.json and produces corrected scores.

THE 5-AXIS RUBRIC
-----------------
1. Format Correctness        25 pts   Right multiples; wrong multiples either
                                       absent OR explicitly rejected
2. Numerical Sanity          25 pts   Numbers in defensible ranges for the
                                       bank's specific sub-vertical
3. Sub-vertical Awareness    20 pts   Recognized when standard metrics
                                       don't apply (trust/IB/payments)
4. Citation Quality          15 pts   Source named, ideally with vintage
5. Format Integrity          15 pts   Table structure, all metrics labeled

KEY DESIGN PRINCIPLE
--------------------
Every scoring decision is one a senior FIG analyst could defend in a peer
review. No fuzzy thresholds. No string matching that ignores context.

USAGE
-----
    python3 rescore_v3.py

Reads:  eval_results_v2.json
Writes: eval_results_v2_rescored.json
        eval_summary_v2_rescored.md
"""

import json
import re
from collections import defaultdict
from pathlib import Path

INPUT_PATH = "eval_results_v2.json"
OUTPUT_JSON = "eval_results_v2_rescored.json"
OUTPUT_MD = "eval_summary_v2_rescored.md"

# === BANK SUB-VERTICAL CLASSIFICATION ===
# This drives sanity bounds and "should caveat" expectations.
# All test set banks plus our training banks classified.
BANK_SUBTYPES = {
    # Money-center commercial
    "C": "money_center", "JPM": "money_center", "BAC": "money_center",
    "WFC": "money_center",

    # Investment bank (hybrid)
    "GS": "investment_bank",

    # Trust/custody (NIM not meaningful, no NPL)
    "BK": "trust_custody", "STT": "trust_custody", "NTRS": "trust_wealth",

    # Payments-heavy (NIM structurally elevated)
    "COF": "payments_credit_card",

    # Super-regional commercial
    "USB": "super_regional", "PNC": "super_regional", "TFC": "super_regional",
    "FITB": "super_regional", "RF": "super_regional", "KEY": "super_regional",
    "MTB": "super_regional", "HBAN": "super_regional", "CFG": "super_regional",
    "SNV": "super_regional", "ONB": "super_regional", "FNB": "super_regional",
    "SSB": "super_regional",

    # Mid-cap regional
    "WAL": "mid_cap_regional", "EWBC": "mid_cap_regional",
    "CFR": "mid_cap_regional", "PB": "mid_cap_regional",
    "FHN": "mid_cap_regional", "WBS": "mid_cap_regional",
    "FCNCA": "mid_cap_regional", "UMBF": "mid_cap_regional",
    "BPOP": "mid_cap_regional", "BOKF": "mid_cap_regional",

    # Community
    "ABCB": "community", "CVBF": "community", "FFBC": "community",
    "INDB": "community", "NBHC": "community",
}


# === SUB-VERTICAL-AWARE NUMERICAL BOUNDS ===
# Each tuple is (low, high) for "defensible range." Outside these is suspect.
NIM_BOUNDS = {
    "money_center":         (1.5, 3.5),
    "investment_bank":      (0.5, 3.0),  # often N/A
    "trust_custody":        (0.8, 2.5),
    "trust_wealth":         (1.0, 2.5),
    "payments_credit_card": (5.0, 10.0),  # COF is structurally high
    "super_regional":       (2.5, 4.0),
    "mid_cap_regional":     (2.5, 4.5),
    "community":            (3.0, 5.0),
}

ROA_BOUNDS = {  # ROA in %
    "money_center":         (0.5, 1.5),
    "investment_bank":      (0.5, 1.5),
    "trust_custody":        (0.8, 2.0),
    "trust_wealth":         (0.8, 2.0),
    "payments_credit_card": (0.8, 2.5),
    "super_regional":       (0.7, 1.6),
    "mid_cap_regional":     (0.8, 1.8),
    "community":            (0.7, 2.0),
}

ROE_BOUNDS = {  # ROE in %
    "money_center":         (5.0, 20.0),
    "investment_bank":      (8.0, 22.0),
    "trust_custody":        (8.0, 25.0),
    "trust_wealth":         (8.0, 20.0),
    "payments_credit_card": (5.0, 25.0),
    "super_regional":       (7.0, 18.0),
    "mid_cap_regional":     (7.0, 20.0),
    "community":            (5.0, 18.0),
}

CET1_BOUNDS = {  # CET1 in %, regulatory floor 7%, peer ceiling ~18%
    "all": (8.0, 18.0),
}

EFFICIENCY_BOUNDS = {  # %
    "money_center":         (45, 75),
    "investment_bank":      (50, 75),
    "trust_custody":        (60, 80),  # higher structurally
    "trust_wealth":         (60, 80),
    "payments_credit_card": (40, 65),
    "super_regional":       (45, 70),
    "mid_cap_regional":     (40, 70),
    "community":            (40, 70),
}


# === FIG-CORRECT METRICS (must be present for full format score) ===
REQUIRED_FIG_METRICS = {
    "p_tbv": [r"p\s*/\s*tbv", r"price\s*to\s*tangible\s*book", r"p\s*/\s*tbvps"],
    "p_e":   [r"p\s*/\s*e", r"price\s*to\s*earnings", r"\bpe\s*ratio"],
    "nim":   [r"\bnim\b", r"net\s*interest\s*margin"],
    "efficiency": [r"efficiency\s*ratio", r"overhead\s*ratio"],
    "roa":   [r"\broa\b", r"return\s*on\s*(?:average\s+)?assets"],
    "roe":   [r"\broe\b", r"return\s*on\s*(?:average\s+)?(?:common\s+)?equity"],
    "cet1":  [r"cet[\s-]?1", r"common\s*equity\s*tier\s*1"],
    "npl":   [r"\bnpl\b", r"non[-\s]?performing\s*loan", r"nonperforming"],
    "div_yield": [r"dividend\s*yield"],
}

# === INDUSTRIALS METRICS — context-aware detection ===
# These need to be checked WITH context: is it being USED or REJECTED?
INDUSTRIALS_PATTERNS = [
    r"ev\s*/\s*ebitda",
    r"ev\s*/\s*sales",
    r"ev\s*/\s*revenue",
    r"enterprise\s*value\s*to\s*ebitda",
    r"enterprise\s*value\s*to\s*sales",
    r"ebitda\s*margin",
    r"\bp\s*/\s*s\s*\b",
    r"price\s*to\s*sales",
]

# Phrases nearby that indicate the model is REJECTING the metric (not using it)
REJECTION_MARKERS = [
    "n/a", "not applicable", "not meaningful", "doesn't apply",
    "do not use", "should not", "inappropriate", "structurally wrong",
    "we avoid", "industrials", "not the right", "irrelevant",
    "not used", "skip", "exclude", "doesn't make sense",
    "deposits aren't debt", "deposits are not debt",
]


# === CITATION TIERS ===
# Tier 1: generic ("source", "filing")
# Tier 2: specific source type ("10-K", "earnings release")
# Tier 3: source + date ("4Q25 earnings release", "2025 annual report")
CITATION_TIER_1 = [r"\bsource\b", r"\bfiling\b", r"\breport\b"]
CITATION_TIER_2 = [r"10-?k", r"10-?q", r"earnings\s+release",
                   r"annual\s+report", r"earnings\s+supplement",
                   r"investor\s+presentation", r"proxy"]
CITATION_TIER_3_DATE = [r"\b(?:fy|q[1-4]|[1-4]q)\s*'?20?2[4-6]\b",
                        r"20[2-3][4-6]\s+annual\s+report",
                        r"4q25", r"4q\s+2025", r"fourth\s+quarter\s+2025",
                        r"december\s+(?:31,?\s+)?2025", r"year[-\s]end\s+2025"]


# === SUB-VERTICAL CAVEAT EXPECTATIONS ===
# These banks should explicitly note when standard metrics don't apply
SUBVERTICAL_CAVEAT_EXPECTATIONS = {
    "trust_custody": {
        "should_caveat_nim": True,
        "should_caveat_npl": True,
        "should_emphasize_fee_or_auc": True,
    },
    "trust_wealth": {
        "should_caveat_nim": True,
        "should_caveat_npl": True,
        "should_emphasize_fee_or_auc": True,
    },
    "investment_bank": {
        "should_caveat_nim": True,
        "should_caveat_npl": True,
    },
    "payments_credit_card": {
        "should_caveat_nim_elevated": True,
    },
}


# ============================================================
# SCORING FUNCTIONS — each returns (score_0_to_max, details)
# ============================================================

def _find_industrials_mentions(text):
    """Return list of (matched_pattern, context_window) tuples."""
    text_lower = text.lower()
    hits = []
    for pattern in INDUSTRIALS_PATTERNS:
        for m in re.finditer(pattern, text_lower):
            start = max(0, m.start() - 80)
            end = min(len(text_lower), m.end() + 80)
            context = text_lower[start:end]
            hits.append((m.group(), context))
    return hits


def _is_rejection_context(context):
    """Determine if an industrials mention is in a REJECTION context."""
    return any(marker in context for marker in REJECTION_MARKERS)


def score_format_correctness(text):
    """25 points. Required FIG metrics present + no industrials *misuse*."""
    text_lower = text.lower()

    # Check FIG metrics present (15 of 25 points)
    found = set()
    for key, patterns in REQUIRED_FIG_METRICS.items():
        if any(re.search(p, text_lower) for p in patterns):
            found.add(key)
    coverage = len(found) / len(REQUIRED_FIG_METRICS)
    metric_score = round(coverage * 15, 1)  # max 15

    # Industrials handling (10 of 25 points)
    industrials_hits = _find_industrials_mentions(text)
    if not industrials_hits:
        # No mention at all — clean
        industrials_score = 10
        misuse = []
        rejection = []
    else:
        # Mentions exist — check if used or rejected
        misuse = []
        rejection = []
        for term, ctx in industrials_hits:
            if _is_rejection_context(ctx):
                rejection.append(term)
            else:
                misuse.append(term)
        if misuse:
            industrials_score = 0  # Any actual misuse = 0
        else:
            # All mentions are rejections — that's actually GOOD, full credit
            industrials_score = 10

    return metric_score + industrials_score, {
        "fig_metrics_found": sorted(found),
        "fig_metrics_missing": sorted(set(REQUIRED_FIG_METRICS) - found),
        "industrials_misuse": misuse,
        "industrials_correct_rejection": rejection,
        "metric_score": metric_score,
        "industrials_score": industrials_score,
    }


def _extract_metric_value(text, metric_patterns):
    """Find first numerical % value associated with a metric."""
    text_lower = text.lower()
    for pattern in metric_patterns:
        # Look for "metric ... NN.N%" within ~50 chars
        for m in re.finditer(pattern, text_lower):
            window = text_lower[m.end():m.end() + 100]
            num_match = re.search(r"[~]?\s*([\d.]+)\s*%", window)
            if num_match:
                try:
                    return float(num_match.group(1))
                except ValueError:
                    continue
    return None


def score_numerical_sanity(text, ticker):
    """25 points. Sub-vertical-aware numerical bounds."""
    subtype = BANK_SUBTYPES.get(ticker, "super_regional")
    flags = []

    # NIM check
    nim = _extract_metric_value(text, [r"\bnim\b", r"net\s*interest\s*margin"])
    if nim is not None:
        low, high = NIM_BOUNDS.get(subtype, (1.5, 5.0))
        if not (low <= nim <= high):
            flags.append(f"NIM={nim}% out of {subtype} range [{low}-{high}]")

    # ROA check
    roa = _extract_metric_value(text, [r"\broa\b", r"return\s*on\s*assets"])
    if roa is not None:
        low, high = ROA_BOUNDS.get(subtype, (0.5, 2.0))
        if not (low <= roa <= high):
            flags.append(f"ROA={roa}% out of {subtype} range [{low}-{high}]")

    # ROE check
    roe = _extract_metric_value(text, [r"\broe\b", r"return\s*on\s*equity"])
    if roe is not None:
        low, high = ROE_BOUNDS.get(subtype, (5.0, 20.0))
        if not (low <= roe <= high):
            flags.append(f"ROE={roe}% out of {subtype} range [{low}-{high}]")

    # CET1 check
    cet1 = _extract_metric_value(text, [r"cet[\s-]?1"])
    if cet1 is not None:
        low, high = CET1_BOUNDS["all"]
        if not (low <= cet1 <= high):
            flags.append(f"CET1={cet1}% out of regulatory range [{low}-{high}]")

    # Efficiency check
    eff = _extract_metric_value(text, [r"efficiency\s*ratio"])
    if eff is not None:
        low, high = EFFICIENCY_BOUNDS.get(subtype, (40, 75))
        if not (low <= eff <= high):
            flags.append(f"Efficiency={eff}% out of {subtype} range [{low}-{high}]")

    # Score: deduct 5 per flag, floor 0
    sanity_score = max(0, 25 - 5 * len(flags))
    return sanity_score, {
        "subtype": subtype,
        "flags": flags,
        "values_found": {
            "nim": nim, "roa": roa, "roe": roe, "cet1": cet1, "efficiency": eff
        },
    }


def score_subvertical_awareness(text, ticker):
    """20 points. Did model recognize when standard metrics don't apply?"""
    subtype = BANK_SUBTYPES.get(ticker, "super_regional")
    text_lower = text.lower()
    expectations = SUBVERTICAL_CAVEAT_EXPECTATIONS.get(subtype, {})

    if not expectations:
        # Standard commercial bank — no special caveats expected
        return 20, {"subtype": subtype, "expectations": "none — standard bank"}

    points_earned = 0
    points_possible = 0
    notes = []

    if expectations.get("should_caveat_nim"):
        points_possible += 7
        nim_caveat = bool(re.search(
            r"nim.*?(?:not\s+meaningful|n/a|structural|low|fee|caveat|"
            r"not\s+(?:the\s+)?primary|less\s+relevant)",
            text_lower, re.DOTALL
        ))
        if nim_caveat:
            points_earned += 7
            notes.append("✓ NIM caveat present")
        else:
            notes.append("✗ NIM caveat missing for sub-vertical")

    if expectations.get("should_caveat_npl"):
        points_possible += 7
        npl_caveat = bool(re.search(
            r"npl.*?(?:n/a|not\s+meaningful|minimal|trivial|no\s+commercial)",
            text_lower, re.DOTALL
        ))
        if npl_caveat:
            points_earned += 7
            notes.append("✓ NPL caveat present")
        else:
            notes.append("✗ NPL caveat missing for sub-vertical")

    if expectations.get("should_emphasize_fee_or_auc"):
        points_possible += 6
        fee_emphasis = any(t in text_lower for t in [
            "fee income", "auc/a", "auc", "asset servicing",
            "custody", "fee revenue", "fee-driven"
        ])
        if fee_emphasis:
            points_earned += 6
            notes.append("✓ Fee/AUC business model noted")
        else:
            notes.append("✗ Fee/AUC business model not noted")

    if expectations.get("should_caveat_nim_elevated"):
        points_possible += 20  # this is THE distinguishing feature for COF
        elevated_note = bool(re.search(
            r"nim.*?(?:elevated|high|credit\s*card|structurally|"
            r"not\s+comparable)",
            text_lower, re.DOTALL
        ))
        if elevated_note:
            points_earned += 20
            notes.append("✓ Elevated NIM context noted (credit card)")
        else:
            notes.append("✗ Elevated NIM context missing (COF risk)")

    # Normalize to 20-point max
    if points_possible > 0:
        score = round(20 * points_earned / points_possible, 1)
    else:
        score = 20

    return score, {
        "subtype": subtype,
        "points_earned": points_earned,
        "points_possible": points_possible,
        "notes": notes,
    }


def score_citation_quality(text):
    """15 points. Tiered citation depth."""
    text_lower = text.lower()

    has_tier3 = any(re.search(p, text_lower) for p in CITATION_TIER_3_DATE)
    has_tier2 = any(re.search(p, text_lower) for p in CITATION_TIER_2)
    has_tier1 = any(re.search(p, text_lower) for p in CITATION_TIER_1)

    if has_tier3:
        return 15, {"tier": 3, "note": "Source + specific date/period"}
    elif has_tier2:
        return 10, {"tier": 2, "note": "Specific source type"}
    elif has_tier1:
        return 5, {"tier": 1, "note": "Generic source mention"}
    else:
        return 0, {"tier": 0, "note": "No citation"}


def score_format_integrity(text):
    """15 points. Markdown table + structure."""
    has_pipes = text.count("|") >= 6  # at least a few rows
    has_header_row = bool(re.search(r"\|[\s\-]+\|[\s\-]+\|", text))
    has_metric_labels = sum(1 for m in [
        "p/tbv", "p/e", "nim", "roa", "roe", "cet1"
    ] if m in text.lower()) >= 4

    score = 0
    if has_pipes:
        score += 6
    if has_header_row:
        score += 5
    if has_metric_labels:
        score += 4

    return score, {
        "has_table_pipes": has_pipes,
        "has_header_row": has_header_row,
        "has_metric_labels": has_metric_labels,
    }


def rescore_response(text, ticker):
    """Apply full 5-axis rubric. Return (composite_0_to_100, breakdown)."""
    fc, fc_d = score_format_correctness(text)
    ns, ns_d = score_numerical_sanity(text, ticker)
    sa, sa_d = score_subvertical_awareness(text, ticker)
    cq, cq_d = score_citation_quality(text)
    fi, fi_d = score_format_integrity(text)

    composite = fc + ns + sa + cq + fi

    return composite, {
        "composite_score": round(composite, 1),
        "format_correctness": {"score": fc, "max": 25, **fc_d},
        "numerical_sanity":   {"score": ns, "max": 25, **ns_d},
        "subvertical_awareness": {"score": sa, "max": 20, **sa_d},
        "citation_quality":   {"score": cq, "max": 15, **cq_d},
        "format_integrity":   {"score": fi, "max": 15, **fi_d},
    }


# ============================================================
# MAIN — re-score existing eval data
# ============================================================

def main():
    src = Path(INPUT_PATH)
    if not src.exists():
        print(f"ERROR: {INPUT_PATH} not found in current directory.")
        print("Run from the same folder as your eval_results_v2.json file.")
        return

    data = json.loads(src.read_text())
    runs = data["runs"]

    rescored_runs = []
    for r in runs:
        if "error" in r or "response" not in r:
            rescored_runs.append(r)
            continue
        new_score, breakdown = rescore_response(r["response"], r["ticker"])
        r_new = dict(r)
        r_new["score_v3"] = breakdown
        rescored_runs.append(r_new)

    # Aggregate by model
    by_model = defaultdict(list)
    for r in rescored_runs:
        if "score_v3" in r:
            by_model[r["model"]].append(r)

    summary = {}
    for model_key, model_runs in by_model.items():
        scores = [r["score_v3"]["composite_score"] for r in model_runs]
        misuse_count = sum(
            1 for r in model_runs
            if r["score_v3"]["format_correctness"]["industrials_misuse"]
        )
        correct_rejection_count = sum(
            1 for r in model_runs
            if r["score_v3"]["format_correctness"]["industrials_correct_rejection"]
        )
        sanity_flags_total = sum(
            len(r["score_v3"]["numerical_sanity"]["flags"]) for r in model_runs
        )
        tier3_citations = sum(
            1 for r in model_runs
            if r["score_v3"]["citation_quality"]["tier"] == 3
        )
        tier2_citations = sum(
            1 for r in model_runs
            if r["score_v3"]["citation_quality"]["tier"] == 2
        )
        any_citation = sum(
            1 for r in model_runs
            if r["score_v3"]["citation_quality"]["tier"] >= 1
        )

        summary[model_key] = {
            "n_runs": len(model_runs),
            "avg_composite": round(sum(scores) / len(scores), 2),
            "min_score": round(min(scores), 1),
            "max_score": round(max(scores), 1),
            "industrials_misuse_rate": (
                f"{misuse_count}/{len(model_runs)} = "
                f"{100*misuse_count/len(model_runs):.0f}%"
            ),
            "industrials_correct_rejection_rate": (
                f"{correct_rejection_count}/{len(model_runs)} = "
                f"{100*correct_rejection_count/len(model_runs):.0f}%"
            ),
            "numerical_sanity_flags_total": sanity_flags_total,
            "citation_tier3_rate": (
                f"{tier3_citations}/{len(model_runs)} = "
                f"{100*tier3_citations/len(model_runs):.0f}%"
            ),
            "citation_any_rate": (
                f"{any_citation}/{len(model_runs)} = "
                f"{100*any_citation/len(model_runs):.0f}%"
            ),
            "avg_breakdown": {
                "format_correctness": round(sum(
                    r["score_v3"]["format_correctness"]["score"]
                    for r in model_runs
                ) / len(model_runs), 1),
                "numerical_sanity": round(sum(
                    r["score_v3"]["numerical_sanity"]["score"]
                    for r in model_runs
                ) / len(model_runs), 1),
                "subvertical_awareness": round(sum(
                    r["score_v3"]["subvertical_awareness"]["score"]
                    for r in model_runs
                ) / len(model_runs), 1),
                "citation_quality": round(sum(
                    r["score_v3"]["citation_quality"]["score"]
                    for r in model_runs
                ) / len(model_runs), 1),
                "format_integrity": round(sum(
                    r["score_v3"]["format_integrity"]["score"]
                    for r in model_runs
                ) / len(model_runs), 1),
            },
        }

    out = {"runs": rescored_runs, "summary_v3_rubric": summary,
           "rubric_version": "v3 — FIG analyst-grade",
           "rubric_axes": {
               "format_correctness": "25 pts — FIG metrics present, no industrials misuse",
               "numerical_sanity":   "25 pts — sub-vertical-aware numerical bounds",
               "subvertical_awareness": "20 pts — caveats for trust/IB/payments",
               "citation_quality":   "15 pts — tiered: 0/5/10/15",
               "format_integrity":   "15 pts — table structure, labels",
           }}
    Path(OUTPUT_JSON).write_text(json.dumps(out, indent=2, default=str))

    # Markdown summary
    md = ["# FIG Eval — Rescored with Industry Rubric (v3)\n"]
    md.append("## Why this rescore exists\n")
    md.append("The original v2 rubric counted phrases like *\"EV/EBITDA: N/A — "
              "not applicable for banks\"* as a violation because the literal "
              "string \"EV/EBITDA\" appeared in the response. A real FIG analyst "
              "would score that as a **correct answer**. This v3 rubric uses "
              "context-aware detection — distinguishing *misuse* from "
              "*correct rejection* — and adds sub-vertical-aware numerical "
              "sanity bounds.\n")
    md.append("\n## Headline results\n")
    md.append("| Model | Avg Score | Industrials *Misuse* | Industrials *Correct Rejection* | Tier-3 Citations | Any Citation |")
    md.append("|---|---|---|---|---|---|")
    for k in ("fine_tuned", "gpt4", "claude"):
        if k not in summary:
            continue
        s = summary[k]
        md.append(
            f"| {k} | {s['avg_composite']} | {s['industrials_misuse_rate']} | "
            f"{s['industrials_correct_rejection_rate']} | "
            f"{s['citation_tier3_rate']} | {s['citation_any_rate']} |"
        )

    md.append("\n## Score breakdown by axis (avg across N=15)\n")
    md.append("| Axis (max) | Fine-tuned | GPT-4 | Claude |")
    md.append("|---|---|---|---|")
    axes = [
        ("Format Correctness (25)", "format_correctness"),
        ("Numerical Sanity (25)", "numerical_sanity"),
        ("Sub-vertical Awareness (20)", "subvertical_awareness"),
        ("Citation Quality (15)", "citation_quality"),
        ("Format Integrity (15)", "format_integrity"),
    ]
    for label, key in axes:
        ft = summary.get("fine_tuned", {}).get("avg_breakdown", {}).get(key, "—")
        g = summary.get("gpt4", {}).get("avg_breakdown", {}).get(key, "—")
        c = summary.get("claude", {}).get("avg_breakdown", {}).get(key, "—")
        md.append(f"| {label} | {ft} | {g} | {c} |")

    Path(OUTPUT_MD).write_text("\n".join(md) + "\n")

    print("=" * 60)
    print("RESCORED with FIG analyst-grade rubric (v3)")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {OUTPUT_JSON}")
    print(f"Wrote: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
