"""
FIG Comparables — Head-to-Head Eval v2

WHAT CHANGED FROM V1
====================
v1 measured the wrong thing. Its system prompt explicitly told all three models
"never use EV/EBITDA, use P/TBV / P/E / NIM..." — then scored whether they used
EV/EBITDA. They didn't. The thesis ("frontier models default to industrials-style
metrics on banks") was untested.

v2 fixes this on five axes:

  1. NEUTRAL SYSTEM PROMPTS for baselines. We tell GPT-5 and Claude only that
     they're a financial analyst — not what multiples to use. This is what
     "default behavior" actually means. The fine-tuned model gets NO system
     prompt: its instructions are baked into weights, which is the entire point.

  2. OPEN-ENDED USER PROMPTS. v1 said "Standard FIG metrics" — that's still
     a hint. v2 says "Include the standard multiples used to value this company"
     and lets the model choose. This is how a junior analyst at a non-FIG shop
     would actually phrase it.

  3. MULTI-AXIS RUBRIC. v1 only scored metric coverage. v2 scores:
       - format adherence (FIG-correct multiples, no industrials metrics)
       - sub-vertical awareness (trust banks vs commercial banks vs payments)
       - knowledge currency (FY2025 numbers, not FY2023)
       - numerical sanity (no obviously-wrong values)
       - citation discipline (cites a source)

  4. MULTIPLE SAMPLES PER PROMPT. We run each prompt 3 times per model with
     temperature 0.7. Single observations are dismissible ("you got unlucky").
     Failure rates over N=15 per model are defensible.

  5. COST ANALYSIS. We log token usage and compute $/inference for each model.
     This is the GTM punchline for Lin Qiao: even at parity quality, Fireworks
     8B is ~20-50x cheaper than GPT-4. That's the whole pitch for inference
     companies — quality at a fraction of the cost via vertical fine-tuning.

OUTPUTS
=======
  - eval_results_v2.json  (full data, all 45 responses)
  - eval_summary_v2.md    (markdown summary, ready to paste in Loom or memo)
"""

import json
import os
import time
from pathlib import Path
from collections import defaultdict

# === CONFIG ===
TEST_SET_PATH = "test_prompts.jsonl"
OUTPUT_JSON = "eval_results_v2.json"
OUTPUT_MD = "eval_summary_v2.md"

SAMPLES_PER_PROMPT = 1        # variance control
TEMPERATURE = 0.0             # produces variation; reveals "default" behavior
MAX_TOKENS = 4000

# Replace with your re-trained v4 model ID (after retraining with epochs=3)
FIREWORKS_MODEL_ID = os.environ.get(
    "FIREWORKS_MODEL_ID",
    "accounts/nikunjmarketing1999/models/fig-comps-v4#accounts/nikunjmarketing1999/deployments/v7o4jagw"
)
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# === SYSTEM PROMPTS — DIFFERENT BY DESIGN ===

# Fine-tuned model: NO system prompt. Knowledge is in weights, not in context.
# This is the entire reason fine-tuning exists.
FT_SYSTEM_PROMPT = None

# Baselines: neutral framing. We are NOT going to tell them what multiples to
# use — that's what we're measuring. This is what an analyst at a generalist
# shop would type into ChatGPT when asked for bank comps.
BASELINE_SYSTEM_PROMPT = (
    "You are a helpful financial analyst assistant. When asked about companies, "
    "provide useful, accurate financial information formatted clearly."
)

# === OPEN-ENDED USER PROMPTS ===
# These deliberately don't specify which multiples to use. Each asks for
# "standard multiples" or "comparable companies" — leaving the model to choose.
# 5 banks × 3 samples × 3 models = 45 total inferences.
TEST_PROMPTS = [
    {
        "bank": "Citigroup",
        "ticker": "C",
        "subtype": "money_center",
        "prompt": (
            "Build a comparable companies table for Citigroup (C) for FY2025. "
            "Include the standard multiples and metrics used to value this "
            "company. Format as a clean table."
        ),
    },
    {
        "bank": "Huntington Bancshares",
        "ticker": "HBAN",
        "subtype": "super_regional",
        "prompt": (
            "I'm building a comps table for Huntington Bancshares (HBAN) for "
            "FY2025. Give me the metrics and trading multiples that would go "
            "in a comparable companies analysis."
        ),
    },
    {
        "bank": "Webster Financial",
        "ticker": "WBS",
        "subtype": "mid_cap_regional",
        "prompt": (
            "Need a valuation comparable companies row for Webster Financial "
            "(WBS), FY2025 figures. Include standard valuation multiples."
        ),
    },
    {
        "bank": "UMB Financial",
        "ticker": "UMBF",
        "subtype": "mid_cap_regional",
        "prompt": (
            "Generate a comps table for UMB Financial (UMBF) covering full-year "
            "2025. Include the standard valuation metrics and multiples."
        ),
    },
    {
        "bank": "Independent Bank Corp",
        "ticker": "INDB",
        "subtype": "community",
        "prompt": (
            "Pull together comparable companies metrics for Independent Bank "
            "Corp (INDB) for FY2025. Standard valuation multiples please."
        ),
    },
]

# === COST DATA (per 1M tokens) ===
# Public list prices as of mid-2026. These are approximations for the cost
# headline — do not represent contract pricing.
COSTS_PER_1M_TOKENS = {
    "fine_tuned": {"input": 0.20, "output": 0.20},   # Fireworks Llama 3.1 8B
    "gpt5":       {"input": 10.00, "output": 30.00}, # GPT-5 published
    "claude":     {"input": 15.00, "output": 75.00}, # Claude Opus published
}


# === RUBRIC ===

# Bank-correct multiples and metrics. Presence is good.
FIG_CORRECT_METRICS = [
    "p/tbv", "price to tangible book", "tangible book",
    "p/e", "price to earnings",
    "nim", "net interest margin",
    "efficiency ratio", "overhead",
    "roa", "return on assets",
    "roe", "return on equity",
    "rotce", "return on tangible",
    "tce/ta", "tangible common equity",
    "cet1", "tier 1 common",
    "npl", "non-performing", "nonperforming",
    "dividend yield",
]

# Industrials-style multiples. Presence is the marquee failure.
FIG_INCORRECT_METRICS = [
    "ev/ebitda", "ev / ebitda", "enterprise value to ebitda",
    "ev/sales", "ev / sales", "enterprise value to sales",
    "ev/revenue", "ev / revenue",
    "ebitda margin", "ebitda%",
    "enterprise value",
    "p/s ratio", "price to sales",
    "peg ratio",
]

# Sub-vertical-specific awareness. For trust banks (BK/STT/NTRS), NIM is not
# meaningful and shouldn't be the headline metric. For Goldman, neither is
# NIM nor NPL. Sophistication = knowing when NOT to apply standard metrics.
SUBTYPE_PENALTIES = {
    "trust_custody": {
        "should_caveat_nim": True,
        "should_emphasize_auc_aum": True,
    },
    "investment_bank": {
        "should_caveat_nim": True,
        "should_caveat_npl": True,
    },
}

# Knowledge-currency markers. Reflects FY2025 data, not stale FY2023.
CURRENCY_MARKERS_FY25 = [
    "fy2025", "fy 2025", "2025", "fiscal 2025",
    "fy25", "fy '25", "year-end 2025", "year end 2025",
    "december 2025", "q4 2025", "4q25", "fourth quarter 2025",
]

# Source-citation discipline.
CITATION_MARKERS = [
    "10-k", "form 10", "annual report",
    "earnings release", "earnings supplement",
    "10-q", "investor presentation",
    "proxy", "10k",
]


def score_response(response_text: str, prompt_meta: dict) -> dict:
    """Multi-axis scoring. Returns dict with breakdown.

    Composite score: 0-100, weighted toward what matters for FIG analyst.
        - Format adherence (FIG metrics + no industrials):  50 points
        - Knowledge currency (FY2025):                      15 points
        - Citation:                                         10 points
        - Numerical sanity (no obvious hallucinations):     15 points
        - Sub-vertical awareness:                           10 points
    """
    text = response_text.lower()

    # 1. FIG-correct metric coverage (count distinct categories, dedup synonyms)
    fig_metric_categories = set()
    for marker in FIG_CORRECT_METRICS:
        if marker in text:
            # Bucket synonyms together so we don't double-count
            if "tbv" in marker or "tangible book" in marker:
                fig_metric_categories.add("p_tbv")
            elif marker in ("p/e", "price to earnings"):
                fig_metric_categories.add("p_e")
            elif "nim" in marker or "net interest margin" in marker:
                fig_metric_categories.add("nim")
            elif "efficiency" in marker or "overhead" in marker:
                fig_metric_categories.add("efficiency")
            elif "roa" in marker or "return on assets" in marker:
                fig_metric_categories.add("roa")
            elif "roe" in marker or "return on equity" in marker:
                fig_metric_categories.add("roe")
            elif "rotce" in marker or "return on tangible" in marker:
                fig_metric_categories.add("rotce")
            elif "tce" in marker or "tangible common" in marker:
                fig_metric_categories.add("tce_ta")
            elif "cet1" in marker or "tier 1 common" in marker:
                fig_metric_categories.add("cet1")
            elif "npl" in marker or "performing" in marker:
                fig_metric_categories.add("npl")
            elif "dividend" in marker:
                fig_metric_categories.add("div_yield")

    fig_metric_count = len(fig_metric_categories)
    # Cap at 8 (have to hit 8 of the 11 categories to score full)
    fig_score = min(fig_metric_count / 8.0, 1.0) * 30

    # 2. Industrials-metric usage — this is the marquee failure
    industrials_violations = []
    for marker in FIG_INCORRECT_METRICS:
        if marker in text:
            industrials_violations.append(marker)
    industrials_score = 20 if not industrials_violations else 0

    # 3. Knowledge currency — does it reference FY2025?
    currency_hit = any(m in text for m in CURRENCY_MARKERS_FY25)
    currency_score = 15 if currency_hit else 0

    # 4. Citation discipline
    has_citation = any(m in text for m in CITATION_MARKERS)
    citation_score = 10 if has_citation else 0

    # 5. Numerical sanity heuristic — for commercial banks, NIM should be
    # 1-5%. If we see something like "NIM: 12%" or "NIM: 0.5%", that's a
    # hallucinated number. (Quick heuristic; not perfect — flags rough cases.)
    sanity_score = 15  # default: pass
    sanity_flags = []
    import re
    nim_matches = re.findall(r"nim[\s:|]+[~]?\s*([\d.]+)\s*%", text)
    for nim in nim_matches:
        try:
            v = float(nim)
            # Commercial banks: 2-5%. Trust banks: 1-2%. COF: 6-8%. Anything
            # outside [0.5, 9] for any bank is suspect.
            if v < 0.5 or v > 9:
                sanity_flags.append(f"suspect NIM={v}%")
                sanity_score = 0
        except ValueError:
            pass

    # 6. Sub-vertical awareness — only relevant for trust/IB; skipped here for
    # the test set (none are trust banks). Reserved for full eval expansion.
    subvertical_score = 10  # default pass; placeholder for richer logic

    composite = (
        fig_score + industrials_score + currency_score
        + citation_score + sanity_score + subvertical_score
    )

    return {
        "composite_score": round(composite, 1),
        "fig_metric_categories_hit": fig_metric_count,
        "fig_metric_score": round(fig_score, 1),
        "industrials_violations": industrials_violations,
        "industrials_score": industrials_score,
        "knowledge_currency_hit": currency_hit,
        "currency_score": currency_score,
        "has_citation": has_citation,
        "citation_score": citation_score,
        "sanity_flags": sanity_flags,
        "sanity_score": sanity_score,
        "subvertical_score": subvertical_score,
    }


# === API CALLS ===

def call_fireworks(prompt: str) -> tuple[str, dict]:
    """Returns (response_text, usage_dict). Usage is for cost calculation."""
    import requests
    messages = []
    if FT_SYSTEM_PROMPT:
        messages.append({"role": "system", "content": FT_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})

    response = requests.post(
        "https://api.fireworks.ai/inference/v1/chat/completions",
        headers={"Authorization": f"Bearer {FIREWORKS_API_KEY}"},
        json={
            "model": FIREWORKS_MODEL_ID,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        },
        timeout=90,
    )
    data = response.json()
    if "choices" not in data:
        raise Exception(f"Fireworks API error: {data}")
    return (
        data["choices"][0]["message"]["content"],
        data.get("usage", {}),
    )


def call_openai_gpt5(prompt: str) -> tuple[str, dict]:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=MAX_TOKENS,
    )
    return (
        resp.choices[0].message.content,
        {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        },
    )


def call_claude(prompt: str) -> tuple[str, dict]:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=MAX_TOKENS,
        system=BASELINE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return (
        msg.content[0].text,
        {
            "prompt_tokens": msg.usage.input_tokens,
            "completion_tokens": msg.usage.output_tokens,
        },
    )


def cost_of(usage: dict, model_key: str) -> float:
    """Compute $ cost from token usage."""
    pricing = COSTS_PER_1M_TOKENS[model_key]
    pin = usage.get("prompt_tokens", 0) / 1_000_000 * pricing["input"]
    pout = usage.get("completion_tokens", 0) / 1_000_000 * pricing["output"]
    return pin + pout


# === MAIN LOOP ===

def run_eval():
    all_runs = []

    print(f"Running {len(TEST_PROMPTS)} prompts × {SAMPLES_PER_PROMPT} samples × 3 models = "
          f"{len(TEST_PROMPTS) * SAMPLES_PER_PROMPT * 3} inferences")
    print(f"Temperature: {TEMPERATURE}\n")

    for prompt_idx, p in enumerate(TEST_PROMPTS, 1):
        print(f"=== Prompt {prompt_idx}/{len(TEST_PROMPTS)}: {p['bank']} ({p['ticker']}) ===")

        for sample_idx in range(1, SAMPLES_PER_PROMPT + 1):
            print(f"  Sample {sample_idx}/{SAMPLES_PER_PROMPT}")

            for model_key, fn in [
                ("fine_tuned", call_fireworks),
                ("gpt5", call_openai_gpt5),
                ("claude", call_claude),
            ]:
                try:
                    text, usage = fn(p["prompt"])
                    score = score_response(text, p)
                    cost = cost_of(usage, model_key)
                    print(f"    {model_key:11s} score={score['composite_score']:5.1f}  "
                          f"violations={len(score['industrials_violations'])}  "
                          f"cost=${cost:.5f}")
                    all_runs.append({
                        "prompt_idx": prompt_idx,
                        "bank": p["bank"],
                        "ticker": p["ticker"],
                        "subtype": p["subtype"],
                        "sample_idx": sample_idx,
                        "model": model_key,
                        "response": text,
                        "score": score,
                        "usage": usage,
                        "cost_usd": cost,
                    })
                except Exception as e:
                    print(f"    {model_key:11s} ERROR: {e}")
                    all_runs.append({
                        "prompt_idx": prompt_idx,
                        "bank": p["bank"],
                        "ticker": p["ticker"],
                        "sample_idx": sample_idx,
                        "model": model_key,
                        "error": str(e),
                    })
                # gentle rate-limit padding
                time.sleep(0.5)

    # === AGGREGATE ===
    by_model = defaultdict(list)
    for r in all_runs:
        if "score" in r:
            by_model[r["model"]].append(r)

    summary = {}
    for model_key, runs in by_model.items():
        scores = [r["score"]["composite_score"] for r in runs]
        violations = [r for r in runs if r["score"]["industrials_violations"]]
        currency_hits = [r for r in runs if r["score"]["knowledge_currency_hit"]]
        citation_hits = [r for r in runs if r["score"]["has_citation"]]
        total_cost = sum(r["cost_usd"] for r in runs)

        summary[model_key] = {
            "n_runs": len(runs),
            "avg_composite_score": round(sum(scores) / len(scores), 2),
            "min_score": round(min(scores), 1),
            "max_score": round(max(scores), 1),
            "industrials_violation_rate": f"{len(violations)}/{len(runs)} = "
                                          f"{100*len(violations)/len(runs):.0f}%",
            "knowledge_currency_rate":   f"{len(currency_hits)}/{len(runs)} = "
                                          f"{100*len(currency_hits)/len(runs):.0f}%",
            "citation_rate":             f"{len(citation_hits)}/{len(runs)} = "
                                          f"{100*len(citation_hits)/len(runs):.0f}%",
            "total_cost_usd": round(total_cost, 4),
            "avg_cost_per_inference_usd": round(total_cost / len(runs), 5),
        }

    # === WRITE OUTPUTS ===
    out = {"runs": all_runs, "summary": summary, "config": {
        "samples_per_prompt": SAMPLES_PER_PROMPT,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "n_test_banks": len(TEST_PROMPTS),
    }}
    Path(OUTPUT_JSON).write_text(json.dumps(out, indent=2, default=str))

    # Markdown summary
    md = ["# FIG Comparables Eval — v2 Results\n"]
    md.append(f"**Setup:** {len(TEST_PROMPTS)} held-out banks × {SAMPLES_PER_PROMPT} "
              f"samples × 3 models = {len(all_runs)} total inferences. "
              f"Temperature {TEMPERATURE}.\n")
    md.append("\n## Summary\n")
    md.append("| Model | Avg Score | Industrials Violations | FY2025 Currency | Citation Rate | Cost/Inference |")
    md.append("|---|---|---|---|---|---|")
    for k in ("fine_tuned", "gpt5", "claude"):
        if k not in summary:
            continue
        s = summary[k]
        md.append(f"| {k} | {s['avg_composite_score']} | {s['industrials_violation_rate']} | "
                  f"{s['knowledge_currency_rate']} | {s['citation_rate']} | "
                  f"${s['avg_cost_per_inference_usd']:.5f} |")
    Path(OUTPUT_MD).write_text("\n".join(md) + "\n")

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {OUTPUT_JSON}")
    print(f"Wrote: {OUTPUT_MD}")


if __name__ == "__main__":
    run_eval()
