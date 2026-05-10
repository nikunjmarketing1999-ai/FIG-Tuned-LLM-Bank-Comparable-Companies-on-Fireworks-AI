# FIG-Tuned LLM — Bank Comparable Companies on Fireworks AI

> A 5-day vertical fine-tuning demonstration showing a Llama 3.1 8B model on Fireworks AI achieving **100% source-citation discipline** and **~1,100× cost reduction** vs. frontier APIs (GPT-5.5, Claude Opus 4.7) on bank comparable companies analysis — at production-realistic API settings.

**Built by:** Nikunj Brahmbhatt — Investment Analyst, Piper Sandler

**Built for:** Anyone exploring the cost-quality frontier of vertical fine-tuning on inference platforms — and the GTM thesis that follows from it.

**Total compute cost:** ~$35 | **Total time:** 5 days | **Lines of training data:** 37 hand-curated examples

---

## Table of Contents

1. [TL;DR](#tldr)
2. [Headline Results](#headline-results)
3. [The Problem This Solves](#the-problem-this-solves)
4. [Methodology](#methodology)
5. [Repository Structure](#repository-structure)
6. [Iteration History](#iteration-history)
7. [How to Reproduce](#how-to-reproduce)
8. [Honest Limitations](#honest-limitations)
9. [Lessons Learned](#lessons-learned)
10. [Why This Matters for Inference-Platform GTM](#why-this-matters-for-inference-platform-gtm)
11. [Contact](#contact)

---

## TL;DR

| Question | Answer |
|---|---|
| What did you build? | A fine-tuned Llama 3.1 8B model on Fireworks AI for bank comparable companies analysis (FIG = Financial Institutions Group) |
| Why? | Frontier LLM APIs cost ~$0.10/inference and lack source-citation discipline at deterministic settings. Vertical fine-tuning closes both gaps at a fraction of the cost. |
| How does it perform? | 100% tier-3 source citations, 0 numerical hallucinations, 20% industrials-misuse rate (vs frontier's 40%) — at $0.00009/inference |
| Cost reduction? | ~1,100× cheaper than GPT-5.5 or Claude Opus 4.7 |
| Time + cost to build? | 5 days, $35 in compute |
| What's the GTM angle? | Same motion (sit with vertical, identify gap, build POC, close cost-quality argument) replicates across healthcare claims, legal contracts, logistics, and other high-volume B2B inference workloads |

---

## Headline Results

**Eval setup:** 5 held-out banks × 1 sample × 3 models = 15 inferences | Temperature 0.0 (deterministic) | FIG-analyst-grade rubric

| Metric | Fine-tuned Llama 3.1 8B | GPT-5.5 (API) | Claude Opus 4.7 (API) |
|---|:-:|:-:|:-:|
| **Avg composite score** (0-100) | 77.1 | 83.4 | **87.0** |
| **Best score** | 86.0 | **100.0** | 98.3 |
| **Min score** | 65.0 | 55.0 | 65.0 |
| **Industrials misuse rate** | **20%** | 40% | 40% |
| **Tier-3 source citations** | **100%** | 80% | 80% |
| **Numerical hallucinations** | **0** | 0 | 3 |
| **Score variance** (max−min) | **21** | 55 | 33 |
| **Cost per inference** | **$0.00009** | $0.0894 | $0.1058 |
| **Cost ratio vs Fine-tuned** | 1× | **994×** | **1,176×** |

**The fine-tuned model wins on 6 of 9 axes.** It loses on raw average score (77 vs 87) but at one-thousandth the cost — a tradeoff that flips the economics for any high-volume vertical workload.

---

## The Problem This Solves

A bank comparable companies (comp) table is a daily deliverable for a financial services analyst. The table requires:

1. **The right multiples** — P/E, P/TBV, ROTCE, NIM, Efficiency Ratio, CET1, Dividend Yield. **NOT** EV/EBITDA, EBITDA margin, or EV/Sales (those are industrials multiples; deposits and debt are operating inputs for banks, not financing capital).
2. **Real numbers** — not templates that say "fill in your own."
3. **Source citations** — every figure needs a defensible provenance ("Source: 4Q25 earnings release, Jan 15 2026"). An MD asking "where did this NIM number come from?" deserves a real answer.

**The gap:** Frontier LLMs at production-realistic API settings (temp=0.0, neutral system prompts) don't reliably solve all three at once.

| Failure mode | GPT-5.5 (API) | Claude Opus 4.7 (API) |
|---|---|---|
| Uses industrials multiples (EV/EBITDA) for banks | 40% of prompts | 40% of prompts |
| Skips source citations | 20% of prompts | 20% of prompts |
| Hallucinates numbers (e.g. NIM out of plausible range) | 0/15 | 3/15 |

A model that gets all three right reliably — and at 0.1% the cost — opens up workloads that were previously uneconomic.

---

## Methodology

### Training

| Setting | Value | Why |
|---|---|---|
| Base model | `llama-v3p1-8b-instruct` | Open weights, well-supported on Fireworks |
| Method | SFT + LoRA (rank 16) | Sufficient for format-and-domain adherence; cheaper than RFT |
| Training records | 37 hand-curated examples | 26 bank comp tables + 5 contrast examples (FIG vs industrials) + 6 newly researched mid-cap banks from FY2025 SEC filings |
| Epochs | **5** | Default of 1 produces no learning |
| Max context length | **4096** | Default of 65536 wastes capacity on short records |
| Batch size | **4096** | Fireworks requires Batch ≥ Max Context |
| LoRA rank | **16** | Standard for domain adaptation |
| Learning rate | **0.0002** | Stable convergence on small datasets |
| Loss curve | 2.10 → 0.55 | Proves the model actually trained |
| Training cost | $0.03 | Per run |

### Evaluation

| Setting | Value | Why |
|---|---|---|
| Test set | 5 held-out banks | C, HBAN, WBS, UMBF, INDB — never seen during training |
| Samples per prompt | 1 | Deterministic settings make multiple samples redundant |
| Temperature | **0.0** | Production-realistic; temp=0.7 produces variance that masks signal |
| Max tokens | **4000** | Allows GPT-5.5 reasoning headroom |
| Baseline system prompt | "You are a helpful financial analyst." | Neutral — does not steer baselines toward FIG metrics |
| User prompt | "Build a comparable companies table for [Bank]. Include the standard multiples used to value this company." | Open-ended — tests default model behavior |
| Models tested | Fine-tuned Llama 3.1 8B (Fireworks) + GPT-5.5 + Claude Opus 4.7 | Both frontier models tested at current production-realistic API settings |

**API contract notes (May 2026):** GPT-5.5 requires `max_completion_tokens` (not `max_tokens`); neither GPT-5.5 nor Claude Opus 4.7 accepts the `temperature` parameter — both are reasoning models with internal sampling. Methodology framing: "tested at current production-realistic settings per provider."

### Scoring (5-axis FIG-analyst-grade rubric, 100 pts)

| Axis | Points | What it measures |
|---|:-:|---|
| **Format Correctness** | 25 | FIG-correct multiples present + no industrials *misuse*. Context-aware: `EV/EBITDA: N/A — not used for banks` is correct rejection, NOT violation. |
| **Numerical Sanity** | 25 | Sub-vertical-aware bounds. NIM ranges differ for money-center vs mid-cap regional vs trust banks. |
| **Sub-vertical Awareness** | 20 | Trust banks must caveat NIM/NPL; investment banks must caveat NIM; payments-heavy must acknowledge elevated NIM. |
| **Citation Quality** | 15 | Tiered: 0 pts no citation, 5 pts generic, 10 pts source type, 15 pts source + date/period. |
| **Format Integrity** | 15 | Markdown table with header row + ≥4 metric labels visible. Parseable structure. |

The rubric is implemented in `rescore_v3.py` and self-tested in `test_rescore.py` against 4 synthetic cases that probe each scoring dimension.

---

## Repository Structure

```
.
├── README.md                          ← You are here
│
├── data/
│   ├── train_v3.jsonl                 ← 37 hand-curated training examples
│   └── test_prompts.jsonl             ← 5 held-out test banks
│
├── scripts/
│   ├── build_jsonl_v3.py              ← Generates training data from structured Python
│   ├── eval_script_v2.py              ← Runs eval against Fireworks + GPT-5.5 + Claude 4.7
│   ├── rescore_v3.py                  ← Applies FIG-analyst-grade rubric
│   └── test_rescore.py                ← Self-tests rubric on synthetic cases
│
└── results/
    ├── eval_results_v2.json           ← Raw eval output (15 inferences)
    ├── eval_results_v2_rescored.json  ← After rubric v3 applied
    └── eval_summary_v2_rescored.md    ← Human-readable summary
```

---

## Iteration History

The final results came from four training iterations + one rubric fix. Each version taught something specific.

| Version | Records | Loss curve | Result | Key learning |
|---|---|---|---|---|
| **v1** | 26 banks | 2.0 → 2.0 (flat) | Tied at ~78 (no signal) | Fireworks defaults (Epochs=1, Batch=65536) = single gradient update. Model never trained. |
| **v2** | 26 (same) | 2.0 → 2.0 (flat) | FT 62, GPT 60, Claude 86 | Eval methodology fixed (neutral prompts, open-ended). Confirmed v1 didn't learn — eval was just measuring base Llama. |
| **v3** | 31 (+5 contrast) | **2.1 → 0.7** | FT 69, GPT 58, Claude 86 | Hyperparameters fixed: Epochs=5, Batch=4096, MaxCtx=4096, LoRA=16, LR=0.0002. Citation rate 0% → 53%. Misuse stuck at 60%. |
| **v4** | 37 (+6 new banks) | **2.1 → 0.55** | **FT 77, GPT 83, Claude 87** | Added SNV/BPOP/ONB/FNB/BOKF/SSB. Re-ran at temp=0.0 (deterministic) against current frontier. Misuse 20% / 40% / 40%. Citations 100% / 80% / 80%. Hallucinations 0 / 0 / 3. Stopped iterating. |

### The rubric fix (rescore_v3.py)

The original scoring rubric counted any occurrence of "EV/EBITDA" as a violation — including correct rejections like `EV/EBITDA: N/A — not typically used for banks`. After rebuilding the rubric to use **context-aware detection**, the model's true performance became visible: it wasn't misusing industrials metrics, it was correctly rejecting them. The new rubric distinguishes:

- **Misuse:** model presents industrials multiples as valid metrics for a bank
- **Correct rejection:** model explicitly states industrials multiples don't apply
- **Absence:** neither used nor mentioned

Self-tested on 4 synthetic cases covering each behavior pattern. All cases score as expected.

---

## How to Reproduce

### Prerequisites

```bash
# API keys (needed for the eval script)
export FIREWORKS_API_KEY="fw_..."
export OPENAI_API_KEY="sk-proj-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Python dependencies
pip install openai anthropic requests
```

### Step 1 — Train the model on Fireworks

Use the Fireworks UI or CLI to launch a fine-tuning job with these exact settings:

| Setting | Value |
|---|---|
| Base model | `accounts/fireworks/models/llama-v3p1-8b-instruct` |
| Dataset | `train_v3.jsonl` (upload from this repo) |
| Epochs | 5 |
| Max context length | 4096 |
| Batch size | 4096 |
| LoRA rank | 16 |
| Learning rate | 0.0002 |

Cost: ~$0.03. Time: ~30 minutes.

### Step 2 — Deploy the model

Deploy your fine-tuned model on the **Minimal** performance tier with auto-scaling OFF and 1 replica. Wait 2-3 min for "Ready" status.

⚠️ **Set a 30-min phone timer.** Dedicated deployments cost $6-12/hour on H200 GPUs. Discipline is non-negotiable.

### Step 3 — Run the eval

```bash
# Set your deployment path
export FIREWORKS_MODEL_ID="accounts/YOUR_ACCT/models/MODEL_ID#accounts/YOUR_ACCT/deployments/DEPLOYMENT_ID"

# Run the eval — calls all 3 models
python3 eval_script_v2.py
```

This produces `eval_results_v2.json` and `eval_summary_v2.md` in your working directory.

### Step 4 — DELETE THE DEPLOYMENT IMMEDIATELY

Don't read results first. Don't analyze. Open the Fireworks UI, navigate to Deployments, delete your deployment. Verify "Deleted" status. Then read results.

### Step 5 — Rescore with FIG-analyst-grade rubric

```bash
# Validate the rubric on synthetic test cases (no API calls)
python3 test_rescore.py

# Apply the rubric to your eval results (no API calls)
python3 rescore_v3.py
```

This produces `eval_results_v2_rescored.json` and `eval_summary_v2_rescored.md`.

### Approximate cost per full evaluation cycle

| Phase | Cost |
|---|:-:|
| Fireworks deployment (~10 min on Minimal tier) | $1-2 |
| GPT-5.5 API (5 prompts at MAX_TOKENS=4000) | $1-2 |
| Claude Opus 4.7 API (5 prompts) | $1-2 |
| Fine-tuned model inference | <$0.001 |
| **Total** | **~$3-6** |

---

## Honest Limitations

Engineers respect transparency. Four caveats worth surfacing.

| Limitation | What it means | Why it doesn't kill the demo |
|---|---|---|
| **Average quality gap is real** | Claude Opus 4.7 averages 87 vs fine-tuned 77. Claude wins on raw quality. | The argument is multi-axis (cost, citations, hallucinations, variance) — not raw quality alone. At one-thousandth the cost, the gap closes for any high-volume vertical workload. |
| **Frontier improves on common verticals** | GPT-4-turbo (mid-2024) failed FIG nearly 100% of the time. GPT-5.5 (April 2026) is at 40%. | Frontier providers will keep closing the gap on well-known domains. The opportunity is on long-tail verticals (healthcare claims, legal clauses) where they update slowly. |
| **N=5 is small** | Five test cases per model is adequate for demo, not a production benchmark. | Production deployment would expand to 50-100 test cases per vertical. The methodology scales. |
| **Format integrity score is low (4.4 of 15)** | Fine-tuned outputs use non-standard table conventions. Substance is correct; presentation varies. | Fixable with iteration on training data formatting, or an output post-processor. Real but not structural. |
| **chatgpt.com ≠ API behavior** | GPT-5.5 on chatgpt.com produces clean FIG comp tables with citations. The API at temp=0.0 does not. | Production enterprises deploy via the API, not chatgpt.com. The eval measures what enterprise customers actually get. |

---

## Lessons Learned

The eight things that mattered, in order of how much pain they caused before the lesson stuck:

| # | Lesson | What it cost to learn |
|---|---|---|
| 1 | **Fireworks training defaults are wrong for small datasets.** Always explicitly set Epochs (5), Max Context (4096), Batch Size (4096), LoRA Rank (16), LR (0.0002). | One full v1 iteration where the model never trained. |
| 2 | **Loss curve is the truth.** If loss goes 2.0 → 1.95, the model didn't learn. Want 2.0 → 1.0 or lower. | Repeated runs that "should have worked." |
| 3 | **Eval methodology matters more than model quality.** First eval was tautological — told baselines what NOT to use, then measured if they used it. | First two iterations of "results" that proved nothing. |
| 4 | **Held-out test set is sacred.** Memorization ≠ generalization. Tempting to merge but kills credibility. | Resisted the temptation; would have killed the demo if violated. |
| 5 | **For deterministic tasks, use temp=0.0.** Multiple samples at temp=0.7 produce noise. N=5 at temp=0.0 > N=15 at temp=0.7 for structured factual output. | A run with 15 samples at temp=0.7 that showed wild variance and obscured the real signal. |
| 6 | **Context-aware rubric design.** Naive string matching counts correct rejection as failure. Real measurement requires checking how a term is *used*. | An entire rescoring pass after realizing the original rubric had a 50%+ false-positive rate on industrials misuse. |
| 7 | **Frontier API contracts change.** Reasoning models (GPT-5.5, Claude 4.7) need different parameters than chat models. Always test the API call before assuming the script works. | Two failed eval runs with `400 Unsupported parameter` errors. |
| 8 | **Cost story is the headline for inference companies.** $0.0001 vs $0.10 = 1,100× reduction. That's the GTM pitch. | This was the strategic pivot that turned a "model comparison demo" into a "cost-quality frontier demo." |

---

## Why This Matters for Inference-Platform GTM

This isn't a model-quality pitch. It's a customer-engagement pitch.

The motion that produced this demo:

1. **Sit with a vertical workflow** — bank comparable companies analysis, what makes a deck-ready table
2. **Identify the structural gap** — frontier APIs at production settings don't cite sources, sometimes hallucinate, cost ~$0.10/inference
3. **Build a fine-tuned POC** — 37 examples, $35 of compute, 5 days
4. **Close the cost-quality argument** — half the structural error rate at one-thousandth the cost

This same motion replicates across:

| Vertical | Workflow gap | Volume that breaks frontier API economics |
|---|---|---|
| Banking / capital markets | Comp tables, deal screening, sector reports | Sell-side analysts running 1000s of comps/month |
| Healthcare claims | Denial code disambiguation, prior auth | Claims processors handling millions of claims/day |
| Legal | Contract clause classification, redlining | Mid-market firms reviewing 100s of contracts/week |
| Logistics | Invoice parsing, customs docs | Freight forwarders processing 10K+ docs/day |
| Insurance | Policy document review, underwriting | Carriers running 1000s of underwriting passes/day |

Any vertical where the unit cost difference between $0.10/inference and $0.0001/inference is the difference between "this is viable" and "this is not."

---

## Contact

| Channel | Link |
|---|---|
| LinkedIn | [linkedin.com/in/nikunj-brahmbhatt-mba-384526221](https://www.linkedin.com/in/nikunj-brahmbhatt-mba-384526221) |
| Email | nikunjmarketing1999@gmail.com |
| Phone | +1 (608) 561-3815 |
| 3-min demo Loom | _[paste your Loom URL here before pushing]_ |

---

**Built in 5 days. $35 in compute. Real model. Real eval. Real GTM thesis.**

If you're hiring for AI Native AE / Forward Deployed PM / Solutions Architect roles at an inference platform — and the cost-quality frontier I've described above resonates with what you're seeing in the market — I'd love 15 minutes.
