# Better SFT 1K Benign200 Report

- base Better rows: 1000
- unsafe kept: 500
- mitigated selected: 300 / 400
- benign selected: 200 / 505
- final rows: 1000
- final_label: {"safe": 500, "unsafe": 500}
- risk_state: {"benign": 200, "mitigated": 300, "committed": 500}
- source: {"APP1-Agentic-Safety-SFT-Data": 200, "BetterSelection-Safe": 300, "BetterSelection-Unsafe": 456, "Fallback-AgentDoG1.0-Harness": 44}
- mitigated risk_source: {"Malicious User Instruction or Jailbreak": 138, "Corrupted Tool Feedback": 88, "Indirect Prompt Injection": 34, "Direct Prompt Injection": 20, "Unreliable or Mis-information": 19, "Inherent Agent/LLM Failures": 1}
- mitigated safe_handling: {"ignored_injection": 135, "refused_or_halted": 104, "safe_alternative_or_partial_completion": 50, "asked_confirmation": 5, "redacted_or_minimized": 4, "verified_or_cross_checked": 2}
- rich prompt chars: {"min": 1477, "median": 6807.5, "p90": 13051, "max": 26877}
- minimal prompt chars: {"min": 1290, "median": 6620.5, "p90": 12864, "max": 26690}
- qwen train-pair tokens: {"min": 369, "median": 2010.5, "p90": 4103, "max": 11146, "over_10k": 1}

Files:
- better_sft_1k_benign200_rich_messages.jsonl
- better_sft_1k_benign200_minimal_messages.jsonl
- better_sft_1k_benign200_rich_agentdog.json
- better_sft_1k_benign200_minimal_agentdog.json
- benign200_manifest.jsonl
