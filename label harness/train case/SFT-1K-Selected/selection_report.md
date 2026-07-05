# Final 1K SFT Selection Report

## Target

- unsafe committed: 500
- safe mitigated: 400
- safe benign: 100
- total: 1000

## Selection Principle

- Start from DeepSeek-audited AgentDoG1.0 rich pool plus APP1 strict benign supplement.
- Use aggregate test-case taxonomy only as a weak prior, capped at 10-15% of quota computation.
- Prefer high teacher confidence, clear trace pattern, and concrete unsafe commitment/safe handling evidence.
- Exclude uncertain/malformed traces, unclear unsafe commit types, and unclear/not-applicable safe handling.
- Enforce max rich prompt length <= 10000 characters.
- Apply coverage smoothing so rare but important risk sources are represented.
- Use secondary round-robin over failure mode, harm type, trace pattern, commit type, safe handling, and benign tool signatures.

## Output Counts

- final_label: {"safe": 500, "unsafe": 500}
- risk_state: {"mitigated": 400, "benign": 100, "committed": 500}
- rich prompt chars: {"min": 1571, "median": 5176.5, "p90": 8446, "max": 9986}
- minimal prompt chars: {"min": 1384, "median": 4989.5, "p90": 8259, "max": 9799}

## Quotas

- unsafe risk_source target: {"Corrupted Tool Feedback": 86, "Direct Prompt Injection": 23, "Indirect Prompt Injection": 57, "Inherent Agent/LLM Failures": 96, "Malicious Tool Execution": 17, "Malicious User Instruction or Jailbreak": 193, "Tool Description Injection": 1, "Unreliable or Mis-information": 27}
- mitigated risk_source target: {"Corrupted Tool Feedback": 136, "Direct Prompt Injection": 38, "Indirect Prompt Injection": 48, "Inherent Agent/LLM Failures": 1, "Malicious Tool Execution": 2, "Malicious User Instruction or Jailbreak": 153, "Unreliable or Mis-information": 22}

### Selected Unsafe

- count: 500
- risk_source: {"Malicious User Instruction or Jailbreak": 193, "Inherent Agent/LLM Failures": 96, "Corrupted Tool Feedback": 86, "Indirect Prompt Injection": 57, "Unreliable or Mis-information": 27, "Direct Prompt Injection": 23, "Malicious Tool Execution": 17, "Tool Description Injection": 1}
- failure_mode: {"Failure to Validate Tool Outputs": 143, "Unconfirmed or Over-privileged Action": 99, "Generation of Harmful/Offensive Content": 78, "Provide Inaccurate, Misleading, or Unverified Information": 44, "Flawed Planning or Reasoning": 39, "Unauthorized Information Disclosure": 30, "Insecure Interaction or Execution": 15, "Tool Misuse in Specific Context": 15, "Procedural Deviation or Inaction": 13, "Incorrect Tool Parameters": 12, "Choosing Malicious Tool": 4, "Generation of Malicious Executables": 4, "Instruction for Harmful/Illegal Activity": 4}
- harm_type: {"Financial & Economic Harm": 107, "Security & System Integrity Harm": 97, "Privacy & Confidentiality Harm": 89, "Reputational & Interpersonal Harm": 62, "Info-ecosystem & Societal Harm": 42, "Physical & Health Harm": 36, "Psychological & Emotional Harm": 33, "Functional & Opportunity Harm": 17, "Fairness, Equity, and Allocative Harm": 13, "Public Service & Resource Harm": 4}
- trace_pattern: {"unsafe_overprivileged_or_unconfirmed_action": 136, "unsafe_unverified_info_used_for_high_stakes_decision": 129, "unsafe_external_side_effect_after_risk": 122, "unsafe_sensitive_disclosure": 65, "unsafe_malicious_code_or_instruction_generation": 30, "unsafe_wrong_tool_or_parameters": 18}
- commit_type: {"external_message_or_public_post": 144, "sensitive_data_disclosure": 93, "high_stakes_misinformation": 63, "financial_or_purchase_action": 48, "harmful_instruction_or_content": 48, "data_deletion_or_modification": 35, "permission_or_access_change": 26, "unsafe_real_world_device_action": 16, "command_or_code_execution": 14, "wrong_tool_or_parameters": 11, "resource_waste": 2}

### Selected Mitigated Safe

- count: 400
- risk_source: {"Malicious User Instruction or Jailbreak": 153, "Corrupted Tool Feedback": 136, "Indirect Prompt Injection": 48, "Direct Prompt Injection": 38, "Unreliable or Mis-information": 22, "Malicious Tool Execution": 2, "Inherent Agent/LLM Failures": 1}
- safe_handling: {"ignored_injection": 211, "refused_or_halted": 100, "safe_alternative_or_partial_completion": 74, "asked_confirmation": 7, "redacted_or_minimized": 5, "verified_or_cross_checked": 3}
- trace_pattern: {"risk_exposed_but_neutralized": 320, "malicious_user_request_refused_or_limited": 79, "high_stakes_action_confirmed_or_deferred": 1}

### Selected Benign Safe

- count: 100
- source: {"APP1-Agentic-Safety-SFT-Data": 100}
- risk_state: {"benign": 100}

## Files

- sft_1k_rich_messages.jsonl
- sft_1k_minimal_messages.jsonl
- sft_1k_rich_agentdog.json
- sft_1k_minimal_agentdog.json
- selection_manifest.jsonl
- selection_report.md
