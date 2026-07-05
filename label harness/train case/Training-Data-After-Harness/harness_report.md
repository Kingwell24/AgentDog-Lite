# Harness Run Report

- Raw records: 4000
- Unique trajectory records: 2000
- Duplicate records removed by hash: 2000
- Conflicting duplicate hash groups: 0
- Selected examples: 2000
- Cached successful cards reused: 2000
- New examples queued for this run: 0
- New teacher calls completed: 0
- Successful teacher calls/cards in merged output: 2000
- Failed teacher calls: 0
- Excluded from training after trace-card audit: 15
- SFT records emitted: 1947
- Rich SFT records emitted: 1947
- Rich mitigated/unsafe records emitted: 1904
- Rich mitigated/unsafe balanced records emitted: 1904
- Reconcile decisions: {"keep": 1928, "manual_review": 38, "flip": 34}
- Final label counts: {"unsafe": 972, "safe": 1028}
- Rich risk_state counts: {"committed": 938, "mitigated": 966, "benign": 43}
- Trace pattern counts: {"unsafe_unverified_info_used_for_high_stakes_decision": 197, "unsafe_sensitive_disclosure": 266, "unsafe_external_side_effect_after_risk": 173, "risk_exposed_but_neutralized": 651, "benign_task_no_risk": 44, "unsafe_malicious_code_or_instruction_generation": 43, "unsafe_overprivileged_or_unconfirmed_action": 234, "unsafe_wrong_tool_or_parameters": 25, "malicious_user_request_refused_or_limited": 361, "uncertain_or_malformed": 4, "high_stakes_action_confirmed_or_deferred": 2}

Generated files:
- selected_examples.jsonl
- trace_cards.jsonl
- excluded_from_training.jsonl
- sft_messages.jsonl
- sft_messages_rich_all.jsonl
- sft_messages_rich_mitigated_unsafe.jsonl
- sft_messages_rich_mitigated_unsafe_balanced.jsonl
- sft_agentdog_format.json
- sft_agentdog_rich_all.json
- sft_agentdog_rich_mitigated_unsafe.json
- sft_agentdog_rich_mitigated_unsafe_balanced.json
- run_config.json
