# FIG-Tuned-LLM-Bank-Comparable-Companies-on-Fireworks-AI
 Demonstration of vertical fine-tuning economics for an inference-platform GTM use case. Llama 3.1 8B fine-tuned on 37 hand-curated bank comparable companies tables, evaluated against current frontier models (GPT-5.5, Claude Opus 4.7) at production-realistic API settings.
Built by: Nikunj — Investment Analyst at Piper Sandler, applying for the AI Native Account Executive role at Fireworks AI.
TL;DR: Fine-tuned model produces analyst-grade FIG comp tables with 100% source citation discipline at $0.00009/inference. Frontier models cite 80% at ~$0.10/inference. ~1,100x cost reduction at lower error rates than frontier on the axes that matter for a financial analyst workflow.

Headline Results
MetricFine-tuned Llama 3.1 8BGPT-5.5 (API)Claude Opus 4.7 (API)Avg score (FIG rubric, 0-100)77.183.487.0Industrials misuse rate20%40%40%Tier-3 source citations100%80%80%Numerical hallucinations003Score variance (max-min)215533Cost per inference$0.00009$0.0894$0.1058Cost ratio1x994x1,176x
Fine-tuned wins 6 of 9 axes. Loses on raw average score (77 vs 87) but at one-thousandth the cost.

Repo Contents
FilePurposetrain_v3.jsonl37 training examples — 26 bank comp tables + 5 contrast examples + 6 newly researched mid-cap bankstest_prompts.jsonl5 held-out test banks (C, HBAN, WBS, UMBF, INDB) — never seen during trainingbuild_jsonl_v3.pyTraining data generator — converts structured Python data to JSONL formateval_script_v2.pyEvaluation runner — calls Fireworks deployment + GPT-5.5 + Claude 4.7 APIsrescore_v3.py5-axis FIG-analyst-grade rubric with context-aware industrials misuse detectiontest_rescore.pyRubric self-test — validates scoring on 4 synthetic caseseval_results_v2.jsonRaw results from eval_script_v2.pyeval_results_v2_rescored.jsonRe-scored with rescore_v3.py for FIG-analyst-grade evaluationeval_summary_v2_rescored.mdHuman-readable summary table

Methodology
Training

Base model: llama-v3p1-8b-instruct
Method: SFT + LoRA (rank 16)
Epochs: 5
Max context: 4096
Batch size: 4096 (Fireworks requires Batch ≥ Max Context)
Learning rate: 0.0002
Loss curve: 2.1 → 0.55
Total training cost: $0.03

Evaluation

Setup: 5 held-out banks × 1 sample each × 3 models = 15 inferences
Temperature: 0.0 (deterministic — production-realistic for structured factual output)
Max tokens: 4000 (allows reasoning headroom for GPT-5.5)
Baseline system prompt: "You are a helpful financial analyst." (neutral)
User prompt: Open-ended ("Build a comparable companies table for [Bank]. Include the standard multiples used to value this company.")
Frontier API contracts: GPT-5.5 uses max_completion_tokens (not max_tokens); Claude Opus 4.7 does not accept temperature parameter — both reasoning models with internal sampling.

Scoring (5-axis rubric, 100 pts total)

Format Correctness (25 pts) — FIG-correct multiples present + no industrials misuse. Context-aware: "EV/EBITDA: N/A — not used for banks" counts as correct rejection, not violation.
Numerical Sanity (25 pts) — Sub-vertical-aware bounds (NIM ranges differ for money-center vs mid-cap regional vs trust banks).
Sub-vertical Awareness (20 pts) — Trust banks must caveat NIM/NPL; investment banks must caveat NIM; payments-heavy must acknowledge elevated NIM.
Citation Quality (15 pts, tiered) — 0 pts no citation, 5 pts generic, 10 pts source type, 15 pts source + date.
Format Integrity (15 pts) — Markdown table with header row + ≥4 metric labels visible.


Honest Limitations

Average quality gap is real. Claude Opus 4.7 averages 87 vs fine-tuned 77. The argument is multi-axis and cost-quality, not raw quality alone.
Frontier models are improving on common verticals. GPT-4-turbo (mid-2024) failed FIG nearly 100% of the time. GPT-5.5 (April 2026) is at 40%. Providers will keep closing the gap on well-known domains.
N=5 is small for statistical confidence. Adequate for demo purposes; production deployment would expand to 50-100 test cases per vertical.
Format integrity score (4.4 of 15) reflects non-standard table conventions in fine-tuned outputs. Substance is correct; presentation varies. Fixable with iteration.


Reproducibility
To run this evaluation against your own Fireworks deployment:
bash# Set API keys
export FIREWORKS_API_KEY="fw_..."
export OPENAI_API_KEY="sk-proj-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Set your Fireworks deployment path
export FIREWORKS_MODEL_ID="accounts/YOUR_ACCT/models/MODEL_ID#accounts/YOUR_ACCT/deployments/DEPLOYMENT_ID"

# Run evaluation (calls all 3 models, writes JSON + markdown summary)
python3 eval_script_v2.py

# Apply FIG-analyst-grade rubric
python3 rescore_v3.py

# Validate rubric on synthetic test cases
python3 test_rescore.py
Approximate cost per full evaluation cycle:

Fireworks deployment: ~$2-3 (10 min on Minimal tier)
GPT-5.5 API: ~$0.50
Claude Opus 4.7 API: ~$0.55
Total: ~$3-4


What I Learned (the seven lessons that mattered)

Fireworks defaults are wrong for small datasets. Defaults of Epochs=1, Batch=65536 result in a single gradient update — the model never trains. Always explicitly set training hyperparameters.
Loss curve is the truth. If loss goes 2.0 → 1.95, the model didn't learn. Want to see 2.0 → 1.0 or lower.
Eval methodology matters more than model quality. First eval was tautological — told baselines what NOT to use, then measured if they used it.
Held-out test set is sacred. Memorization is not generalization. Tempting to merge but kills credibility.
For deterministic tasks, use temp=0.0. Multiple samples at temp=0.7 produces noise. N=5 at temp=0.0 is more informative than N=15 at temp=0.7.
Context-aware rubric design. Naive string matching counts correct rejection as failure. Real measurement requires checking how a term is used.
Frontier API contracts change. Reasoning models (GPT-5.5, Claude Opus 4.7) need different parameters than chat models. Always test the API call before assuming the script works.
