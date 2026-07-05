# APP1 Benign Supplement Report

- APP1 JSON: E:\AgentDog Harness\train case\APP1-Agentic-Safety-SFT-Data\agentic_safety_sft.json
- Target accepted benign: 100
- Scanned records this run: 6185
- New heuristic candidates this run: 512
- New teacher calls completed this run: 512
- Cached accepted benign before/after resume: 0 -> 505
- Accepted benign available: 505
- Accepted benign emitted to SFT: 100
- Selection mode: diverse
- Minimum confidence: 0.95

Acceptance rule:
- teacher decision=accept
- is_benign=true
- risk_present=false
- risk_source=benign
- trace_pattern=benign_task_no_risk
- assistant_behavior=normal_task_completion

Generated files:
- app1_benign_candidates.jsonl
- app1_benign_trace_cards.jsonl
- app1_benign_rejected.jsonl
- app1_benign_accepted.jsonl
- app1_benign_sft_messages.jsonl
- app1_benign_sft_agentdog.json
