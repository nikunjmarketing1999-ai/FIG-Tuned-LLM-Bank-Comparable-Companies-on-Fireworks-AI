# FIG Eval — Rescored with Industry Rubric (v3)

## Why this rescore exists

The original v2 rubric counted phrases like *"EV/EBITDA: N/A — not applicable for banks"* as a violation because the literal string "EV/EBITDA" appeared in the response. A real FIG analyst would score that as a **correct answer**. This v3 rubric uses context-aware detection — distinguishing *misuse* from *correct rejection* — and adds sub-vertical-aware numerical sanity bounds.


## Headline results

| Model | Avg Score | Industrials *Misuse* | Industrials *Correct Rejection* | Tier-3 Citations | Any Citation |
|---|---|---|---|---|---|
| fine_tuned | 77.06 | 1/5 = 20% | 0/5 = 0% | 5/5 = 100% | 5/5 = 100% |
| claude | 87.0 | 2/5 = 40% | 1/5 = 20% | 4/5 = 80% | 4/5 = 80% |

## Score breakdown by axis (avg across N=15)

| Axis (max) | Fine-tuned | GPT-4 | Claude |
|---|---|---|---|
| Format Correctness (25) | 12.7 | — | 18.0 |
| Numerical Sanity (25) | 25.0 | — | 22.0 |
| Sub-vertical Awareness (20) | 20.0 | — | 20.0 |
| Citation Quality (15) | 15.0 | — | 12.0 |
| Format Integrity (15) | 4.4 | — | 15.0 |
