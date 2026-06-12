"""Self-test rescore_v3 rubric on synthetic responses representing failure modes."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rescore_v3 import rescore_response

print("=" * 72)
print("RUBRIC SELF-TEST: 4 synthetic responses, expected scoring behavior")
print("=" * 72)

# CASE 1: Model correctly REJECTS EV/EBITDA. Should NOT be penalized.
case1 = """**Citigroup (C) — FY2025 Comparable Companies Analysis**

| Metric | FY2025 |
|---|---|
| Market Cap | ~$155B |
| P/TBV | ~1.0x |
| P/E (LTM) | ~13.0x |
| Dividend Yield | ~2.9% |
| NIM | 2.65% |
| Efficiency Ratio | 65.0% |
| ROA | ~0.55% |
| ROE | 6.8% |
| ROTCE | 7.7% |
| CET1 Ratio | 13.2% |
| NPL Ratio | ~0.6% |

Note: EV/EBITDA is N/A — not applicable for banks because deposits are
operating liabilities, not financing capital.

Source: Citigroup 4Q25 earnings release.
"""

# CASE 2: Model INCORRECTLY USES EV/EBITDA as a real metric. Should be penalized.
case2 = """**Huntington Bancshares Comparable Analysis**

| Metric | Value |
|---|---|
| EV/EBITDA | 12.5x |
| EBITDA Margin | 35% |
| P/E | 11x |
| ROE | 12.4% |
| NIM | 3.13% |

This shows HBAN's enterprise value relative to its earnings.
"""

# CASE 3: Model uses correct metrics, no citation, hallucinated NIM.
case3 = """**Webster Financial (WBS)**

| Metric | Value |
|---|---|
| P/TBV | 1.7x |
| P/E | 10.7x |
| NIM | 8.5% |
| Efficiency Ratio | 46.0% |
| ROA | 1.24% |
| ROE | 11% |
| CET1 | 11.22% |
| NPL | 0.88% |
"""
# NIM 8.5% for a mid-cap regional should fail sanity check.

# CASE 4: Trust bank without sub-vertical caveats.
case4 = """**Bank of New York Mellon**

| Metric | Value |
|---|---|
| P/TBV | 3.0x |
| P/E | 14x |
| NIM | 1.33% |
| ROA | 1.3% |
| ROE | 13.9% |
| CET1 | 11.9% |
| NPL | 0.6% |

Source: 2025 annual report.
"""
# Should lose subvertical awareness points — no caveat that NIM is structural,
# no mention of fee/AUC business model.

cases = [
    ("Case 1 — Correctly REJECTS EV/EBITDA", case1, "C"),
    ("Case 2 — Incorrectly USES EV/EBITDA",  case2, "HBAN"),
    ("Case 3 — Hallucinated NIM (8.5% for mid-cap)", case3, "WBS"),
    ("Case 4 — Trust bank without caveats", case4, "BK"),
]

for label, text, ticker in cases:
    print(f"\n{'─'*72}")
    print(f"  {label}")
    print(f"  Ticker: {ticker}")
    print('─'*72)
    score, breakdown = rescore_response(text, ticker)
    print(f"  COMPOSITE: {score}/100")
    print(f"  Format Correctness:    {breakdown['format_correctness']['score']}/25")
    print(f"    misuse:    {breakdown['format_correctness']['industrials_misuse']}")
    print(f"    rejection: {breakdown['format_correctness']['industrials_correct_rejection']}")
    print(f"  Numerical Sanity:      {breakdown['numerical_sanity']['score']}/25")
    print(f"    flags: {breakdown['numerical_sanity']['flags']}")
    print(f"  Sub-vertical Aware:    {breakdown['subvertical_awareness']['score']}/20")
    print(f"    notes: {breakdown['subvertical_awareness'].get('notes', [])}")
    print(f"  Citation Quality:      {breakdown['citation_quality']['score']}/15  "
          f"(tier {breakdown['citation_quality']['tier']})")
    print(f"  Format Integrity:      {breakdown['format_integrity']['score']}/15")

print("\n" + "=" * 72)
print("EXPECTED BEHAVIOR:")
print("  Case 1 should score HIGH (~75-90) — correct rejection, good citation")
print("  Case 2 should score LOW  (~20-40) — actual misuse penalized")
print("  Case 3 should be PENALIZED on numerical sanity (NIM=8.5% is wrong)")
print("  Case 4 should LOSE points on sub-vertical awareness (BK trust caveats)")
print("=" * 72)
