# Better SFT 1K Build Report

## Inputs

- Better selection: E:\AgentDog Harness\train case\Better-SFT-1K-Selection\train_label_1000.json
- selected safe rows: 500
- selected unsafe rows: 500
- unsafe teacher calls needed this run: 0
- unsafe teacher cards emitted: 500
- teacher-confirmed Better unsafe usable rows: 456
- unsafe backfill rows used: 44
- unsafe rejected after teacher: 44
- safe usable mitigated rows from Better selection: 454
- safe rejected from Better selection: 46

## Final Counts

- total: 1000
- final_label: {"unsafe": 500, "safe": 500}
- risk_state: {"committed": 500, "mitigated": 400, "benign": 100}
- uid/source prefix: {"fg": 396, "bin": 460, "no": 144}
- source: {"BetterSelection-Unsafe": 456, "BetterSelection-Safe": 400, "APP1-Agentic-Safety-SFT-Data": 100, "Fallback-AgentDoG1.0-Harness": 44}
- rich prompt chars: {"min": 1477, "median": 6108.5, "p90": 13011, "max": 26877}
- minimal prompt chars: {"min": 1290, "median": 5921.5, "p90": 12824, "max": 26690}

## Selected Mitigated Safe

- risk_source: {"Malicious User Instruction or Jailbreak": 205, "Corrupted Tool Feedback": 121, "Indirect Prompt Injection": 34, "Direct Prompt Injection": 20, "Unreliable or Mis-information": 19, "Inherent Agent/LLM Failures": 1}
- safe_handling: {"refused_or_halted": 171, "ignored_injection": 168, "safe_alternative_or_partial_completion": 50, "asked_confirmation": 5, "redacted_or_minimized": 4, "verified_or_cross_checked": 2}
- trace_pattern: {"risk_exposed_but_neutralized": 235, "malicious_user_request_refused_or_limited": 164, "high_stakes_action_confirmed_or_deferred": 1}

## Selected Unsafe

- risk_source: {"Malicious User Instruction or Jailbreak": 225, "Inherent Agent/LLM Failures": 94, "Corrupted Tool Feedback": 80, "Indirect Prompt Injection": 47, "Unreliable or Mis-information": 22, "Direct Prompt Injection": 16, "Tool Description Injection": 9, "Malicious Tool Execution": 7}
- failure_mode: {"Failure to Validate Tool Outputs": 130, "Unconfirmed or Over-privileged Action": 130, "Unauthorized Information Disclosure": 89, "Generation of Harmful/Offensive Content": 50, "Provide Inaccurate, Misleading, or Unverified Information": 42, "Flawed Planning or Reasoning": 27, "Tool Misuse in Specific Context": 8, "Procedural Deviation or Inaction": 7, "Incorrect Tool Parameters": 6, "Insecure Interaction or Execution": 5, "Choosing Malicious Tool": 3, "Generation of Malicious Executables": 2, "Instruction for Harmful/Illegal Activity": 1}
- harm_type: {"Privacy & Confidentiality Harm": 192, "Financial & Economic Harm": 106, "Security & System Integrity Harm": 69, "Reputational & Interpersonal Harm": 36, "Info-ecosystem & Societal Harm": 30, "Psychological & Emotional Harm": 21, "Physical & Health Harm": 17, "Functional & Opportunity Harm": 13, "Fairness, Equity, and Allocative Harm": 10, "Public Service & Resource Harm": 6}
- trace_pattern: {"unsafe_overprivileged_or_unconfirmed_action": 135, "unsafe_sensitive_disclosure": 130, "unsafe_unverified_info_used_for_high_stakes_decision": 122, "unsafe_external_side_effect_after_risk": 74, "unsafe_malicious_code_or_instruction_generation": 21, "unsafe_wrong_tool_or_parameters": 18}
- commit_type: {"sensitive_data_disclosure": 163, "external_message_or_public_post": 106, "financial_or_purchase_action": 65, "high_stakes_misinformation": 47, "data_deletion_or_modification": 37, "harmful_instruction_or_content": 25, "permission_or_access_change": 23, "command_or_code_execution": 16, "unsafe_real_world_device_action": 9, "wrong_tool_or_parameters": 8, "resource_waste": 1}

## Files

- better_sft_1k_rich_messages.jsonl
- better_sft_1k_minimal_messages.jsonl
- better_sft_1k_rich_agentdog.json
- better_sft_1k_minimal_agentdog.json
- better_unsafe_trace_cards.jsonl
- better_selection_manifest.jsonl
- better_sft_1k_report.md
