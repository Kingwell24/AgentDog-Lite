# preprocessing — AgentDoG 1.0 → compact training set

Turns the raw **AgentDoG 1.0** pool (8000 rows) into a small, balanced,
leakage-free, optionally CoT-augmented training set for fine-tuning
Qwen3.5-0.8B. This is our reproduction of the AgentDoG 1.5 data-preparation
recipe (§3.2): the parts the paper describes but does **not** open-source
(CoT augmentation + purification) are re-implemented here on top of the released
1.0 data.

## Local vs GPU (per AGENTS.md)

| Step | Script | Where | Needs |
| --- | --- | --- | --- |
| 0 Inventory & merge | `step0_inventory.py` | **local** | — |
| 1 Dedup & leakage | `step1_dedup.py` | **local** | numpy |
| 2 Rule QC | `step2_rules.py` | **local** | — |
| 3 Label verify | `step3_verify_labels.py` | any | teacher API |
| 4 Budget CoT | `step4_cot_augment.py` | any | teacher API |
| 5 Select & compress | `step5_select.py` | **local** | numpy |
| score learnability | `score_learnability.py` | **GPU server** | torch + model |

Steps 0/1/2/5 run today with no model and no API. Steps 3/4 need the on-site
teacher API (`api_client.py`, set `AGENTDOG_API_BASE` / `AGENTDOG_API_KEY`).
`score_learnability.py` runs on the GPU server and refuses to run locally unless
`--allow-local-run` is passed.

## Run order (server quick start)

```bash
# 0) setup ------------------------------------------------------------------
pip install numpy transformers torch   # torch: install the CUDA build for your server
# training data (AgentDoG 1.0, 8000 rows) — download once:
hf download AI45Research/AgentDoG1.0-Training-Data --repo-type dataset \
    --local-dir /data/agentdog10
export AGENTDOG_TRAIN_DIR=/data/agentdog10   # dir containing AgentDoG-BinarySafety/ etc.
# (test sets ship with the repo under `test case/`; no env var needed)

cd preprocessing

# 1) local text processing (no GPU, no API) ----------------------------------
python step0_inventory.py       # inventory + merge the two configs
python step1_dedup.py           # exact dedup + leakage vs test + near-dup collapse
python step2_rules.py           # structural QC

# 2) GPU: learnability scores -------------------------------------------------
python score_learnability.py --missing-only
#   --missing-only  = resumable: checkpoints every 200 samples, rerun to continue
#   --max-length    = 6144 default (safe on 8GB); 16384+ fine on A100/3090-class

# 3) selection & training files ----------------------------------------------
python step5_select.py          # folds in whatever score files exist
python make_label_files.py      # A-version (JSON-label) training files, no API

# 4) optional, with teacher API (on-site release) -----------------------------
export AGENTDOG_API_BASE=https://...  AGENTDOG_API_KEY=sk-...
python step3_verify_labels.py           # semantic model-checker -> confidence
python step5_select.py                  # re-select with confidence folded in
python step4_cot_augment.py --size 1000 # B-version (short-CoT) training files
```

`step5` folds in `learnability`/`label_confidence` if their score files exist and
falls back to neutral otherwise — every stage is optional-degradable, and any
stage can be re-run after new scores land.

## Outputs

Everything lands under `outputs/preprocessing/` (gitignored). Each step writes a
`*_report.md` you can paste straight into the tech report, plus JSON artifacts.
Final training files come from step4 (`train_cot_<size>.json`,
`train_label_<size>.json`, alpaca format for LLaMA-Factory).

## Findings on the current data (from an actual local run)

- BinarySafety ships **each trajectory twice**: 4000 rows = 2000 unique. Exact
  dedup removes 2000 with zero information loss.
- FineGrained = 4000 **distinct, all-unsafe** trajectories, disjoint from Binary.
- Leakage vs test: 0 exact, 5 near-duplicates (Jaccard ≥ 0.85) removed.
- After honest cleaning the coarse pool is **864 safe : 4102 unsafe** — safe is
  the scarce resource, which drives the balancing / over-refusal strategy.

## Design notes

- **Learnability score** (`score_learnability.py`) is a gradient-free stand-in
  for the paper's influence-function purification: samples the baseline already
  nails are redundant; boundary/near-miss samples score highest.
- **Budget CoT** (`step4`) caps reasoning at ≤3 sentences on purpose — the
  competition scores output token cost, so we avoid verbose chains.
- **Safe risk-present quota** (`step5 --safe-risk-frac`) forces enough
  "risk was present but the agent handled it → safe" examples into the safe half,
  the main data-side lever against over-refusal.
