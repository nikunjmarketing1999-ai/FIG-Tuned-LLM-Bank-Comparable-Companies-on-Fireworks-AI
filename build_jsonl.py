"""
build_jsonl.py — turn 31 bank records into Fireworks-ready training data.

Writes:
  - train.jsonl          (26 banks, fine-tuning input)
  - test_prompts.jsonl   (5 banks, held-out for head-to-head eval)

Both files use the same {messages: [system, user, assistant]} schema that
Fireworks SFT expects and that eval_script.py already parses.

Re-run this script any time you tweak data, prompts, or the train/test split.
"""

import json
import random
from pathlib import Path

# Match the system prompt in eval_script.py EXACTLY — fair comparison requires
# the model is trained under the same system prompt the eval uses.
SYSTEM_PROMPT = (
    "You are a senior FIG investment banking analyst generating comparable companies "
    "tables in the standard format used at top-tier sell-side banks like KBW, Piper "
    "Sandler, and Stifel. Output ONLY bank-specific valuation multiples — never "
    "EV/EBITDA, EV/Sales, or industrials-style metrics. Use P/TBV, P/E, NIM, "
    "Efficiency Ratio, ROA, ROE, TCE/TA, CET1, and NPL Ratio. Format as a markdown "
    "table row with all values clearly labeled."
)

# Prompt templates — varied so the model doesn't overfit to one phrasing
USER_PROMPT_TEMPLATES = [
    "Pull FY2025 trading comps for {name} ({ticker}).",
    "Build a FIG comp table for {ticker} — full year 2025 figures.",
    "I need FY2025 comparables for {name}. Standard FIG metrics.",
    "Generate the comps row for {name} ({ticker}), year-end 2025.",
    "{ticker} comp table please — FY2025 metrics, all the standard FIG inputs.",
    "Give me FY2025 trading multiples and operating metrics for {name}.",
    "Comp sheet for {name}, FY2025. Need the full bank-multiples set.",
]

# Each bank: structured fields + commentary + sources
# Order roughly: money-center → super-regional → mid-cap → community
BANKS = [
    # ---------- MONEY-CENTER (5) ----------
    {
        "name": "JPMorgan Chase", "ticker": "JPM", "tier": "Money-Center",
        "metrics": {
            "Market Cap": "~$870B", "Total Assets": "$4,424.9B", "NIM": "2.45%",
            "Efficiency Ratio": "52%", "ROA": "1.29%", "ROE": "17%",
            "ROTCE": "20%", "TCE/TA": "~6.3%", "CET1 Ratio": "14.6%",
            "P/E (LTM)": "~15.5x", "P/TBV": "~2.9x", "NPL Ratio": "~0.6%",
            "Dividend Yield": "~1.9%",
        },
        "commentary": (
            "FY25 net income $57.0B, modestly down YoY on $14.2B credit costs. "
            "ROTCE of 20% remains industry-leading; premium P/TBV ~2.9x reflects "
            "scale, $4.8T AWM AUM, and consistent capital return."
        ),
        "source": "2025 Annual Report; 4Q25 earnings supplement",
    },
    {
        "name": "Bank of America", "ticker": "BAC", "tier": "Money-Center",
        "metrics": {
            "Market Cap": "~$370B", "Total Assets": "~$3,360B", "NIM": "2.05%",
            "Efficiency Ratio": "62%", "ROA": "0.89%", "ROE": "10.6%",
            "ROTCE": "14.2%", "TCE/TA": "~6.5%", "CET1 Ratio": "11.4%",
            "P/E (LTM)": "~13.0x", "P/TBV": "~1.7x", "NPL Ratio": "~0.5%",
            "Dividend Yield": "~2.4%",
        },
        "commentary": (
            "FY25 net income $30.5B (+13% YoY). Efficiency improved 146 bps to 62% "
            "as fixed-rate asset repricing drove NII +10% YoY. ROE recovery to 10.6% "
            "reflects normalization post-2023 BSBY cessation charge. Note: BAC's NIM "
            "appears low vs. peers due to total-assets denominator methodology."
        ),
        "source": "4Q25 earnings release",
    },
    {
        "name": "Wells Fargo", "ticker": "WFC", "tier": "Money-Center",
        "metrics": {
            "Market Cap": "~$245B", "Total Assets": "~$1,950B", "NIM": "2.70%",
            "Efficiency Ratio": "66%", "ROA": "1.07%", "ROE": "12.4%",
            "ROTCE": "14.6%", "TCE/TA": "~7.3%", "CET1 Ratio": "10.6%",
            "P/E (LTM)": "~12.0x", "P/TBV": "~1.8x", "NPL Ratio": "~0.6%",
            "Dividend Yield": "~2.3%",
        },
        "commentary": (
            "FY25 net income $21.3B (+8% YoY); ROTCE expanded to 14.6%. The defining "
            "FY25 story: the Federal Reserve LIFTED THE 2018 ASSET CAP, freeing WFC "
            "to grow the balance sheet for the first time in 7 years. Q4 average "
            "loans grew +5% YoY (vs. -3% in 4Q24); valuation has re-rated upward."
        ),
        "source": "2025 Annual Report; 4Q25 supplement",
    },
    {
        "name": "Citigroup", "ticker": "C", "tier": "Money-Center",
        "metrics": {
            "Market Cap": "~$155B", "Total Assets": "~$2,500B", "NIM": "2.65%",
            "Efficiency Ratio": "65.0%", "ROA": "~0.55%", "ROE": "6.8%",
            "ROTCE": "7.7%", "TCE/TA": "~7.9%", "CET1 Ratio": "13.2%",
            "P/E (LTM)": "~13.0x", "P/TBV": "~1.0x", "NPL Ratio": "~0.6%",
            "Dividend Yield": "~2.9%",
        },
        "commentary": (
            "FY25 ROE 6.8% reported / 7.7% adjusted (vs. 6.1% FY24) — Fraser-era "
            "transformation showing measurable progress; efficiency ratio improved "
            "170 bps. P/TBV moved from ~0.9x to ~1.0x as the market re-rated the "
            "transformation. Significant ROTCE gap to JPM (20% vs. 7.7%) but trajectory positive."
        ),
        "source": "4Q25 earnings release",
    },
    {
        "name": "Goldman Sachs", "ticker": "GS", "tier": "Money-Center",
        "metrics": {
            "Market Cap": "~$215B", "Total Assets": "~$1,860B",
            "NIM": "N/A — GS revenue is ~75% Markets/IB; NIM not meaningful",
            "Efficiency Ratio": "59.6%", "ROA": "~0.90%", "ROE": "15.0%",
            "ROTCE": "~15.5%", "TCE/TA": "~6.5%", "CET1 Ratio": "14.4%",
            "P/E (LTM)": "~14.5x", "P/TBV": "~1.9x",
            "NPL Ratio": "N/A — minimal commercial loan book",
            "Dividend Yield": "~2.0%",
        },
        "commentary": (
            "FY25 net earnings $17.2B (vs. $14.3B FY24); ROE 15.0% is highest since "
            "the 2021 cycle. Efficiency improved 350 bps YoY (63.1% → 59.6%) reflecting "
            "Marcus consumer exit and refocus on GBM/AWM. NIM/NPL not meaningful comp "
            "metrics for the hybrid model."
        ),
        "source": "2025 Annual Report; 4Q25 results",
    },

    # ---------- SUPER-REGIONAL (13, including 3 trust/wealth) ----------
    {
        "name": "U.S. Bancorp", "ticker": "USB", "tier": "Super-Regional",
        "metrics": {
            "Market Cap": "~$70B", "Total Assets": "~$680B", "NIM": "2.77%",
            "Efficiency Ratio": "57.2% adjusted", "ROA": "1.17%", "ROE": "13.5%",
            "ROTCE": "18.4%", "TCE/TA": "~6.7%", "CET1 Ratio": "10.8%",
            "P/E (LTM)": "~11x", "P/TBV": "~2.0x", "NPL Ratio": "~0.4%",
            "Dividend Yield": "~4.0%",
        },
        "commentary": (
            "CEO Gunjan Kedia executing toward medium-term targets: ROTCE high-teens, "
            "efficiency mid-to-high 50s, ROA 1.15-1.35%. FY25 record net revenue with "
            "440 bps positive operating leverage. BTIG acquisition announced (12 bps "
            "CET1 impact). Multi-year ROTCE expansion story from low- toward high-teens."
        ),
        "source": "4Q25 earnings release; 3Q25 investor presentation",
    },
    {
        "name": "PNC Financial Services", "ticker": "PNC", "tier": "Super-Regional",
        "metrics": {
            "Market Cap": "~$80B", "Total Assets": "$573.6B", "NIM": "2.83%",
            "Efficiency Ratio": "60%", "ROA": "1.24%", "ROE": "12.90%",
            "TCE/TA": "~7.6%", "CET1 Ratio": "10.6%",
            "P/E (LTM)": "~12.6x", "P/TBV": "~1.9x", "NPL Ratio": "~0.7%",
            "Dividend Yield": "~3.2%",
        },
        "commentary": (
            "FY25 record revenue $23.1B with 5% positive operating leverage; net income "
            "$7.0B (+18% YoY). Closed FirstBank acquisition Jan 5, 2026 (~$26B assets, "
            "Colorado/Arizona expansion). Efficiency improved 300 bps to 60%. Tier 1 "
            "Mid-Atlantic + Midwest commercial franchise positioning for Western growth."
        ),
        "source": "2025 Annual Report",
    },
    {
        "name": "Truist Financial", "ticker": "TFC", "tier": "Super-Regional",
        "metrics": {
            "Market Cap": "~$70B", "Total Assets": "~$540B", "NIM": "3.03%",
            "Efficiency Ratio": "~57%", "ROA": "~0.95%", "ROE": "~9.0%",
            "ROTCE": "~13%", "TCE/TA": "~7.0%", "CET1 Ratio": "10.8%",
            "P/E (LTM)": "~13.5x", "P/TBV": "~1.4x", "NPL Ratio": "~0.45%",
            "Dividend Yield": "~5.0%",
        },
        "commentary": (
            "FY25 net income $5.0B; EPS $3.82. Post-restructuring story — TIH divested "
            "in 2024 to refocus on core banking. CEO Bill Rogers locked in on 14% ROTCE "
            "in 2026, 15% by 2027. Q4 NIM 3.07% with management guiding 3-teens exit "
            "rate. $5.2B returned to shareholders in 2025."
        ),
        "source": "4Q25 earnings release",
    },
    {
        "name": "Capital One", "ticker": "COF", "tier": "Super-Regional (post-Discover)",
        "metrics": {
            "Market Cap": "~$135B", "Total Assets": "~$650B (post-Discover)",
            "NIM": "6.5%+ (structurally elevated due to credit-card-heavy book)",
            "Efficiency Ratio": "~55% adjusted", "ROA": "~1.3%",
            "ROE": "Variable — Q2 reported $4.3B GAAP loss on Discover day-one CECL",
            "TCE/TA": "N/A", "CET1 Ratio": "14.0%",
            "P/E (LTM)": "~13x adjusted", "P/TBV": "~1.3x",
            "NPL Ratio": "~0.85% (credit-card concentration drives elevated NPL)",
            "Dividend Yield": "~1.4%",
        },
        "commentary": (
            "Closed Discover acquisition May 18, 2025 — transformative: COF becomes the "
            "largest US credit card issuer by loan volume. Q2 GAAP loss on day-one CECL "
            "provisioning; Q4 normalized to $2.1B / $3.86 adjusted EPS. IMPORTANT FOR "
            "COMP TABLES: COF's ~7.6% NIM is structurally not comparable to traditional "
            "commercial banks due to credit-card-heavy loan mix."
        ),
        "source": "Post-Discover close June 2025; 4Q25 results",
    },
    {
        "name": "Bank of New York Mellon", "ticker": "BK", "tier": "Super-Regional (Trust/Custody)",
        "metrics": {
            "Market Cap": "~$70B", "Total Assets": "~$420B", "NIM": "1.33%",
            "Efficiency Ratio": "~65%", "ROA": "~1.3%", "ROE": "13.9%",
            "ROTCE": "26.1%", "TCE/TA": "~5%", "CET1 Ratio": "11.9%",
            "P/E (LTM)": "~14x", "P/TBV": "~3.0x",
            "NPL Ratio": "N/A — minimal commercial loan book",
            "Dividend Yield": "~3.0%",
        },
        "commentary": (
            "Trust/custody bank — fundamentally different model from commercial banks. "
            "Revenue ~71% from fees (custody, asset servicing, asset management). "
            "Record AUC/A $53.8T, AUM $5.7T. ROTCE 26.1% reflects capital-light fee "
            "model; low NIM (1.33%) is structural, not weakness. Comp set is STT/NTRS, "
            "not commercial banks."
        ),
        "source": "2025 10-K",
    },
    {
        "name": "State Street", "ticker": "STT", "tier": "Super-Regional (Trust/Custody)",
        "metrics": {
            "Market Cap": "~$28B", "Total Assets": "~$365B", "NIM": "~1.20%",
            "Efficiency Ratio": "~70%", "ROA": "~0.75%", "ROE": "~12%",
            "ROTCE": "~20%", "TCE/TA": "~5.5%", "CET1 Ratio": "~10.8%",
            "P/E (LTM)": "~10x", "P/TBV": "~2.0x",
            "NPL Ratio": "N/A", "Dividend Yield": "~3.5%",
        },
        "commentary": (
            "FY25 revenue $14.0B (+7% YoY); net income $2.7B (+9% YoY). Record AUC/A "
            "$53.8T, AUM $5.7T. Targeting 80% payout ratio for 2025. Investment "
            "Servicing core franchise; SPDR ETF franchise drove record inflows. "
            "Comp set is BK/NTRS, not commercial banks."
        ),
        "source": "4Q25 disclosures (FY25 figures partially estimated)",
    },
    {
        "name": "Northern Trust", "ticker": "NTRS", "tier": "Super-Regional (Trust/Wealth)",
        "metrics": {
            "Market Cap": "~$28B", "Total Assets": "~$160B", "NIM": "1.72%",
            "Efficiency Ratio": "~70%", "ROA": "~1.0%", "ROE": "14.4%",
            "TCE/TA": "~7.5%", "CET1 Ratio": "12.6%",
            "P/E (LTM)": "~13x", "P/TBV": "~2.5x",
            "NPL Ratio": "N/A — minimal lending book",
            "Dividend Yield": "~2.8%",
        },
        "commentary": (
            "Wealth/trust bank in BK/STT comp set. FY25 revenue $8.1B; AUC/A $18.7T, "
            "AUM $1.8T. Returned $1.9B to shareholders ($1.3B record buybacks reduced "
            "share count 5%). Targeting 33% pre-tax margin / mid-teens ROE under 'One "
            "Northern Trust' strategy. Private wealth management is core differentiator "
            "vs. BK/STT custody focus."
        ),
        "source": "FY25 disclosures",
    },
    {
        "name": "Fifth Third Bancorp", "ticker": "FITB", "tier": "Super-Regional",
        "metrics": {
            "Market Cap": "~$30B", "Total Assets": "~$215B", "NIM": "3.13%",
            "Efficiency Ratio": "54.3% adjusted (4Q25)", "ROA": "1.36%",
            "ROE": "14.0%", "ROTCE": "16.2% adjusted", "TCE/TA": "~7.5%",
            "CET1 Ratio": "10.77%", "P/E (LTM)": "~12x", "P/TBV": "~2.1x",
            "NPL Ratio": "0.65%", "Dividend Yield": "~3.7%",
        },
        "commentary": (
            "FY25 net income $2.4B (+12% YoY EPS); 230 bps positive operating leverage; "
            "record NII of $6B. Announced Comerica acquisition — pending close, paused "
            "Q4 buybacks. Pro forma post-close target: 19% ROTCE / ~53% efficiency in "
            "4Q26 (acceleration vs. original 2027 target)."
        ),
        "source": "4Q25 release",
    },
    {
        "name": "Regions Financial", "ticker": "RF", "tier": "Super-Regional",
        "metrics": {
            "Market Cap": "~$23B", "Total Assets": "~$160B", "NIM": "3.70%",
            "Efficiency Ratio": "~56% adjusted", "ROA": "1.42%", "ROE": "12.56%",
            "ROTCE": "18.25%", "TCE/TA": "~7.8%", "CET1 Ratio": "10.8%",
            "P/E (LTM)": "~11x", "P/TBV": "~2.3x", "NPL Ratio": "~0.6%",
            "Dividend Yield": "~4.6%",
        },
        "commentary": (
            "FY25 net income $2.16B; adjusted EPS $2.33 (+9% YoY). ROATCE of 18.25% — "
            "RF claims highest in peer group for 5 straight years. NIM expanded 11 bps "
            "QoQ to 3.70% in Q4. Birmingham, AL HQ. Well-protected from Fed cuts via "
            "$4.5B forward-starting hedges. P/TBV premium reflects ROE leadership."
        ),
        "source": "4Q25 release",
    },
    {
        "name": "KeyCorp", "ticker": "KEY", "tier": "Super-Regional",
        "metrics": {
            "Market Cap": "~$23B", "Total Assets": "~$190B", "NIM": "2.82%",
            "Efficiency Ratio": "~60%", "ROA": "~0.95%", "ROE": "~10%",
            "ROTCE": "~15%", "TCE/TA": "~6.5%", "CET1 Ratio": "11.7%",
            "P/E (LTM)": "~12x", "P/TBV": "~1.7x", "NPL Ratio": "0.39%",
            "Dividend Yield": "~3.9%",
        },
        "commentary": (
            "Turnaround story — FY24 net loss on securities repositioning; FY25 recovery "
            "year. NII grew 23% in FY25 vs. plan; record full-year revenue $7.5B; ~1200 "
            "bps positive operating leverage. NIM rebuilding from trough; targeting "
            "3.00-3.05% by end-2026. Plans $1.2B+ buybacks in 2026."
        ),
        "source": "4Q25 release",
    },
    {
        "name": "M&T Bank", "ticker": "MTB", "tier": "Super-Regional",
        "metrics": {
            "Market Cap": "~$32B", "Total Assets": "~$210B", "NIM": "3.67%",
            "Efficiency Ratio": "56%", "ROA": "~1.35%", "ROE": "~11%",
            "ROTCE": "~17%", "TCE/TA": "~8.5%", "CET1 Ratio": "10.84%",
            "P/E (LTM)": "~12x", "P/TBV": "~1.6x", "NPL Ratio": "~0.5%",
            "Dividend Yield": "~3.0%",
        },
        "commentary": (
            "FY25 net income $2.85B; EPS $17.00 (+16% YoY). Conservative operator "
            "with industry-leading 56% efficiency ratio. Buffalo NY HQ; Mid-Atlantic/"
            "Northeast commercial franchise. Repurchased 9% of shares in 2025; raised "
            "dividend 11%. Above-peer credit and efficiency profile."
        ),
        "source": "4Q25 release",
    },
    {
        "name": "Huntington Bancshares", "ticker": "HBAN", "tier": "Super-Regional",
        "metrics": {
            "Market Cap": "~$23B", "Total Assets": "$250B+ (post-Veritex/Cadence)",
            "NIM": "3.13%", "Efficiency Ratio": "57.4%", "ROA": "1.19%",
            "ROE": "12.4%", "ROTCE": "17.8%", "TCE/TA": "~6.6%",
            "CET1 Ratio": "10.4%", "P/E (LTM)": "~11x", "P/TBV": "~1.5x",
            "NPL Ratio": "~0.5%", "Dividend Yield": "~4.0%",
        },
        "commentary": (
            "Most active M&A story among super-regionals. Closed Veritex Oct 2025 "
            "($12B assets, Texas) and announced Cadence Bank merger ($54B assets, "
            "21-state expansion). Pro forma will exceed $250B assets. Strong organic "
            "growth alongside M&A: 14% revenue growth YoY, 16% adjusted PPNR growth."
        ),
        "source": "Q3 2025 results; 2025 10-K",
    },
    {
        "name": "Citizens Financial Group", "ticker": "CFG", "tier": "Super-Regional",
        "metrics": {
            "Market Cap": "~$22B", "Total Assets": "~$220B", "NIM": "3.07%",
            "Efficiency Ratio": "~63%", "ROA": "~0.85%", "ROE": "~10%",
            "ROTCE": "12.2%", "TCE/TA": "~7.0%", "CET1 Ratio": "10.6%",
            "P/E (LTM)": "~12x", "P/TBV": "~1.5x", "NPL Ratio": "~0.5%",
            "Dividend Yield": "~3.7%",
        },
        "commentary": (
            "FY25 net income $1.8B (+21% YoY); record EPS $3.86. Private Bank initiative "
            "exceeded targets — 7% accretive vs. 5% target, 25% ROE. Reduced non-core "
            "assets from $6.9B to $2.5B. Returned 80% of earnings; retired 3% of shares. "
            "NIM expansion targeting 3.15-3.30% by Q4 2026."
        ),
        "source": "4Q25 release",
    },

    # ---------- MID-CAP REGIONAL (8) ----------
    {
        "name": "Western Alliance Bancorporation", "ticker": "WAL",
        "tier": "Mid-Cap Regional",
        "metrics": {
            "Market Cap": "$9.55B", "Total Assets": "$92.7B", "NIM": "3.51%",
            "Efficiency Ratio": "58.9% reported / 50.2% adjusted",
            "ROA": "1.14%", "ROE": "~14%", "ROTCE": "16.9% (Q4)",
            "TCE/TA": "7.3%", "CET1 Ratio": "11.0%",
            "P/E (LTM)": "9.99x", "P/TBV": "1.42x",
            "NPL Ratio": "0.85%", "Dividend Yield": "1.79%",
        },
        "commentary": (
            "Record FY25 — net income $990.6M; EPS $8.73 (+23.1%); revenue $3.54B "
            "(+12%). Total assets crossed $90B+ heading toward LFI threshold. Brand "
            "unity Oct 2025 consolidating ABA, BON, FIB, Bridge, TPB under single "
            "Western Alliance Bank brand. TBVPS $61.29 (+17% YoY)."
        ),
        "source": "2025 10-K; companiesmarketcap year-end",
    },
    {
        "name": "East West Bancorp", "ticker": "EWBC", "tier": "Mid-Cap Regional",
        "metrics": {
            "Market Cap": "~$15.7B", "Total Assets": "$80.4B", "NIM": "3.41%",
            "Efficiency Ratio": "34.5% (Q4) / ~36% FY25 — best-in-class",
            "ROA": "1.70%", "ROE": "16.0% (ROTCE 17%)",
            "TCE/TA": "10.5%", "CET1 Ratio": "15.1% (highest in peer set)",
            "P/E (LTM)": "11.85x", "P/TBV": "1.87x",
            "NPL Ratio": "0.26%", "Dividend Yield": "2.79% forward",
        },
        "commentary": (
            "Record FY25 — NI $1.325B, EPS $9.52, record NII $2.6B, record fee "
            "income $348M (+12%). Best-in-class efficiency (~36%) and ROA (1.70%). "
            "CET1 of 15.1% is highest in major peer set — supporting 33% dividend "
            "hike to $0.80/qtr. Unique US/Asia franchise. Dominic Ng CEO since 1992."
        ),
        "source": "Macrotrends Dec 29 close; FY25 disclosures",
    },
    {
        "name": "Cullen/Frost Bankers", "ticker": "CFR", "tier": "Mid-Cap Regional",
        "metrics": {
            "Market Cap": "$8.27B", "Total Assets": "$53.04B", "NIM": "3.66%",
            "Efficiency Ratio": "61.16%", "ROA": "1.24%", "ROE": "15.66%",
            "TCE/TA": "~7.9%", "CET1 Ratio": "14.06%",
            "P/E (LTM)": "12.88x", "P/TBV": "~1.89x",
            "NPL Ratio": "0.32%", "Dividend Yield": "3.02%",
        },
        "commentary": (
            "Texas-only commercial bank, San Antonio HQ since 1868. FY25 NIAC $641.9M "
            "(+11.5%); EPS $9.92 (+11.8%). Premium P/TBV reflects 'Texas growth + "
            "minimal goodwill' — book ≈ TBV is a clean valuation story. Opened 10 new "
            "financial centers in 2025; on track for 200th in Austin region."
        ),
        "source": "Q4 2025 release; companiesmarketcap year-end",
    },
    {
        "name": "Prosperity Bancshares", "ticker": "PB", "tier": "Mid-Cap Regional",
        "metrics": {
            "Market Cap": "$6.54B", "Total Assets": "$38.46B", "NIM": "3.21%",
            "Efficiency Ratio": "44.6%", "ROA": "1.40%",
            "ROE": "7.1% (depressed by high book from M&A; ROTCE ~13.6% Q4 cleaner)",
            "TCE/TA": "~10%", "CET1 Ratio": "17.55% (highest among major US banks)",
            "P/E (LTM)": "12.05x", "P/TBV": "1.60x",
            "NPL Ratio": "0.69%", "Dividend Yield": "3.43%",
        },
        "commentary": (
            "Houston HQ; Texas/Oklahoma franchise. FY25 NI $542.8M (+13.2%); EPS "
            "$5.72. CET1 17.55% is highest among major US banks — strategic capital "
            "reservoir for M&A. Three-deal year: closed American Bank Holding (Jan 1, "
            "2026), Southwest Bancshares pending, announced $2B Stellar Bancorp deal."
        ),
        "source": "10-K; Q4 2025 release",
    },
    {
        "name": "First Horizon", "ticker": "FHN", "tier": "Mid-Cap Regional",
        "metrics": {
            "Market Cap": "$12.44B", "Total Assets": "$83.876B", "NIM": "3.47%",
            "Efficiency Ratio": "60.6%", "ROA": "~1.15%", "ROE": "~11% GAAP",
            "ROTCE": "14.2% adjusted FY25 / 15.0% Q4", "TCE/TA": "8.37%",
            "CET1 Ratio": "10.64%", "P/E (LTM)": "12.85x", "P/TBV": "1.81x",
            "NPL Ratio": "0.94%", "Dividend Yield": "2.34%",
        },
        "commentary": (
            "Memphis HQ. FY25 NIAC $956M (+29%); adjusted EPS $1.89 (+22% — clean "
            "read; FY24 baseline depressed by $105M after-tax notable items). Deployed "
            "$1.2B excess capital — $894M buybacks (42M shares at avg $21.16). Hit "
            "sustained 15%+ ROTCE in Q4 — meeting 2026 target."
        ),
        "source": "Q4 2025 release",
    },
    {
        "name": "Webster Financial", "ticker": "WBS", "tier": "Mid-Cap Regional",
        "metrics": {
            "Market Cap": "~$10.4B", "Total Assets": "$83.2B", "NIM": "3.42%",
            "Efficiency Ratio": "46.0% — top quartile",
            "ROA": "1.24% adjusted", "ROE": "~11-12% GAAP",
            "ROTCE": "17.26% adjusted", "TCE/TA": "7.42%",
            "CET1 Ratio": "11.22%", "P/E (LTM)": "10.74x",
            "P/TBV": "1.74x", "NPL Ratio": "0.88%", "Dividend Yield": "2.48%",
        },
        "commentary": (
            "Stamford CT HQ. Record FY25 — adjusted EPS $5.94 (+10.4%); TBVPS $37.20 "
            "(+12.9% YoY). HSA Bank franchise (Healthcare Financial Services) is the "
            "key differentiator — counter-cyclical fee income and sticky low-cost "
            "deposits. Repurchased 10.9M shares in 2025. Loans +7.8% YoY."
        ),
        "source": "FY25 disclosures",
    },
    {
        "name": "First Citizens BancShares", "ticker": "FCNCA",
        "tier": "Mid-Cap Regional (super-regional scale)",
        "metrics": {
            "Market Cap": "~$22.5B", "Total Assets": "$229.7B",
            "NIM": "3.26% reported / 3.16% ex-PAA",
            "Efficiency Ratio": "61.27% reported / 56.78% adjusted",
            "ROA": "1.01% reported / 1.25% adjusted",
            "ROE": "10.62% reported / 12.56% adjusted",
            "TCE/TA": "~9%", "CET1 Ratio": "11.15%",
            "P/E (LTM)": "~9.8x reported", "P/TBV": "~0.99x — essentially at book",
            "NPL Ratio": "0.88%",
            "Dividend Yield": "0.51% — FCNCA returns capital via buybacks",
        },
        "commentary": (
            "Raleigh NC HQ; family-controlled (Holding family). Acquired SVB Commercial "
            "March 2023 in transformational FDIC-assisted deal. Q4 actions: $900M "
            "buyback + prepaid $2.5B Purchase Money Note (FDIC obligation from SVB) + "
            "$500M Series D preferred. TBVPS $1,674.11 (+10.7% YoY). Announced 138-"
            "branch acquisition from BMO closing H2 2026."
        ),
        "source": "Q4 2025 release",
    },
    {
        "name": "UMB Financial", "ticker": "UMBF",
        "tier": "Mid-Cap Regional (institutional/fund services)",
        "metrics": {
            "Market Cap": "~$10.0B", "Total Assets": "$73.1B (+45% YoY post-Heartland)",
            "NIM": "3.10%", "Efficiency Ratio": "~58% reported / ~52.5% operating",
            "ROA": "~1.02%", "ROE": "~12%", "TCE/TA": "~7%",
            "CET1 Ratio": "10.70%", "P/E (LTM)": "14.0x", "P/TBV": "1.94x",
            "NPL Ratio": "0.35%", "Dividend Yield": "1.42%",
        },
        "commentary": (
            "Kansas City MO HQ; 113-year history. Closed Heartland Financial acquisition "
            "Jan 31, 2025 — total assets +45% YoY to $73.1B. FY25 record NI $684.6M, "
            "diluted EPS $9.29; Q4 NIM expanded 25 bps QoQ as deal accretion built "
            "(~33 bps Q4 NIM from purchase-accounting accretion; core ex-accretion 2.96%). "
            "Differentiator vs pure regionals = Institutional Banking segment."
        ),
        "source": "FY25 release; Q4 2025",
    },

    # ---------- COMMUNITY (5) ----------
    {
        "name": "Ameris Bancorp", "ticker": "ABCB", "tier": "Community",
        "metrics": {
            "Market Cap": "$5.76B", "Total Assets": "$27.52B", "NIM": "3.85%",
            "Efficiency Ratio": "~52%", "ROA": "1.57%", "ROE": "14.5%",
            "TCE/TA": "11.4%", "CET1 Ratio": "13.2%",
            "P/E (LTM)": "12.4x", "P/TBV": "1.4x",
            "NPL Ratio": "0.35%", "Dividend Yield": "1.1%",
        },
        "commentary": (
            "Strong Southeast community bank franchise with above-peer ROA (1.57%) "
            "driven by elevated NIM (3.85%) from commercial-skewed loan book. Recent "
            "Balboa Capital integration (specialty equipment finance) contributing to "
            "fee income diversification. Capital-rich (CET1 13.2%) provides M&A optionality."
        ),
        "source": "4Q25 supplement; Yahoo Finance April 2026",
    },
    {
        "name": "CVB Financial", "ticker": "CVBF", "tier": "Community",
        "metrics": {
            "Market Cap": "$3.6B", "Total Assets": "$15.63B", "NIM": "3.49%",
            "Efficiency Ratio": "~44.4%", "ROA": "1.40%", "ROE": "9.48%",
            "TCE/TA": "10.0%", "CET1 Ratio": "16.5%",
            "P/E (LTM)": "13.4x", "P/TBV": "1.7x",
            "NPL Ratio": "0.31%", "Dividend Yield": "4.3%",
        },
        "commentary": (
            "California-focused commercial bank (Citizens Business Bank), serves "
            "middle-market businesses across Southern California. Distinguished by "
            "extremely high capitalization (CET1 16.5% — well above peers), "
            "conservative underwriting, consistent dividend history. Lower ROE "
            "(9.48%) reflects capital-rich balance sheet."
        ),
        "source": "4Q25 supplement",
    },
    {
        "name": "First Financial Bancorp", "ticker": "FFBC", "tier": "Community",
        "metrics": {
            "Market Cap": "$2.5B", "Total Assets": "$21.1B", "NIM": "3.98%",
            "Efficiency Ratio": "57.5% adjusted", "ROA": "1.34%", "ROE": "~13%",
            "ROTCE": "20% adjusted", "TCE/TA": "7.79%",
            "CET1 Ratio": "11.32%", "P/E (LTM)": "~10x", "P/TBV": "~1.7x",
            "NPL Ratio": "~0.41%", "Dividend Yield": "4.0%",
        },
        "commentary": (
            "Cincinnati-based commercial bank with $21B assets and recent BankFinancial "
            "acquisition expanding footprint. Strong NIM (~4%) reflects commercial-"
            "skewed loan book. Be careful distinguishing reported (1.34% ROA) from "
            "adjusted (1.47-1.52%). PPNR ROA of 2.14% indicates strong core earnings "
            "power. 4% dividend yield."
        ),
        "source": "4Q25 investor presentation; earnings call transcript",
    },
    {
        "name": "Independent Bank Corp", "ticker": "INDB", "tier": "Community",
        "metrics": {
            "Market Cap": "$3.79B", "Total Assets": "$24.9B", "NIM": "3.57%",
            "Efficiency Ratio": "~62%", "ROA": "0.92%", "ROE": "6.20%",
            "TCE/TA": "8.7%", "CET1 Ratio": "11.5%",
            "P/E (LTM)": "~16x", "P/TBV": "1.6x",
            "NPL Ratio": "0.45%", "Dividend Yield": "3.3%",
        },
        "commentary": (
            "Massachusetts-based community bank operating as Rockland Trust with 151 "
            "branches concentrated in New England. FY2025 net income $205M; ROA "
            "expansion to 0.92% reflecting NIM compression. $300M sub debt raise Q1 "
            "2025 strengthened capital. ~80% loans/deposits ratio. Lower ROE (6.20%) "
            "reflects capital-heavy structure and Northeast competitive dynamics."
        ),
        "source": "2025 Annual Report",
    },
    {
        "name": "National Bank Holdings", "ticker": "NBHC", "tier": "Community",
        "metrics": {
            "Market Cap": "$1.91B", "Total Assets": "$9.9B", "NIM": "3.94%",
            "Efficiency Ratio": "62.4%", "ROA": "1.30%", "ROE": "12.15%",
            "TCE/TA": "9.4%", "CET1 Ratio": "13.1%",
            "P/E (LTM)": "13.2x", "P/TBV": "1.6x",
            "NPL Ratio": "0.34%", "Dividend Yield": "3.4%",
        },
        "commentary": (
            "Denver-based community bank with operations across Colorado, Utah, Texas "
            "and surrounding states. Above-peer NIM (3.94%) reflects mountain-state "
            "deposit pricing. Built historically through acquisition including Bank of "
            "Jackson Hole. Solid capital position (CET1 13.1%) and clean credit "
            "quality (NPL 0.34%) support continued M&A optionality."
        ),
        "source": "4Q25 supplement",
    },
]


# === FORMATTING HELPERS ===

def metrics_to_table(metrics: dict) -> str:
    """Render the metrics dict as a markdown table."""
    lines = ["| Metric | FY2025 |", "|---|---|"]
    for k, v in metrics.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def build_assistant_response(bank: dict) -> str:
    """Build the assistant message: table + commentary + source."""
    table = metrics_to_table(bank["metrics"])
    return (
        f"**{bank['name']} ({bank['ticker']}) — {bank['tier']} | FY2025**\n\n"
        f"{table}\n\n"
        f"**Commentary:** {bank['commentary']}\n\n"
        f"*Source: {bank['source']}.*"
    )


def build_record(bank: dict, prompt_template: str) -> dict:
    """Build one JSONL record in the {messages: [...]} format Fireworks expects."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompt_template.format(
                    name=bank["name"], ticker=bank["ticker"]
                ),
            },
            {"role": "assistant", "content": build_assistant_response(bank)},
        ]
    }


# === TRAIN/TEST SPLIT ===
# Test set chosen for tier-diversity AND for cases that highlight where a
# fine-tuned FIG model should beat baselines (transformation stories,
# acquisition-accounting noise, fee-business quirks, small-bank coverage).
TEST_TICKERS = {"C", "HBAN", "WBS", "UMBF", "INDB"}


def main():
    random.seed(42)  # reproducibility — same prompt cycling every run

    train_path = Path("train.jsonl")
    test_path = Path("test_prompts.jsonl")
    train_path.parent.mkdir(parents=True, exist_ok=True)

    train_count, test_count = 0, 0
    with train_path.open("w") as ftr, test_path.open("w") as fte:
        for i, bank in enumerate(BANKS):
            template = USER_PROMPT_TEMPLATES[i % len(USER_PROMPT_TEMPLATES)]
            record = build_record(bank, template)
            line = json.dumps(record, ensure_ascii=False) + "\n"
            if bank["ticker"] in TEST_TICKERS:
                fte.write(line)
                test_count += 1
            else:
                ftr.write(line)
                train_count += 1

    print(f"Total banks: {len(BANKS)}")
    print(f"  → train.jsonl:        {train_count} records")
    print(f"  → test_prompts.jsonl: {test_count} records")
    print(f"\nTest set: {sorted(TEST_TICKERS)}")
    print(f"Files written to: {train_path.parent}")


if __name__ == "__main__":
    main()
