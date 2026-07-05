#!/usr/bin/env python3
"""Step 1 - Dedup & leakage removal.

Three jobs, in order:
  1. Exact intra-pool dedup   -> drop identical trajectories (BinarySafety ships
     each row twice).
  2. Leakage removal vs test  -> drop any training trajectory that duplicates or
     near-duplicates an ATBench300 / R-judge test trajectory. This is the
     compliance red line: training on a test trajectory is cheating.
  3. Intra-pool near-dup collapse (optional) -> keep one representative per
     cluster of "same skeleton, different random values" template clones.

MinHash signatures are cached to `scores/pool_signatures.npy` for reuse by the
selection step (diversity).

Local-safe: numpy text processing only.
"""

from __future__ import annotations

import argparse

import numpy as np

from common import (
    MinHasher,
    extract_scenario_text,
    jaccard,
    load_json,
    normalize_light,
    normalize_masked,
    save_json,
    sha1,
    test_sample_to_text,
    word_shingles,
    write_text,
)
import config


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build_lsh(hasher: MinHasher, signatures: np.ndarray) -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = {}
    for idx, sig in enumerate(signatures):
        for key in hasher.band_keys(sig):
            buckets.setdefault(key, []).append(idx)
    return buckets


def load_test_texts() -> list[str]:
    texts = []
    for name, path in config.TEST_FILES.items():
        data = load_json(path)
        texts.extend(test_sample_to_text(s) for s in data)
        print(f"  test {name}: {len(data)} trajectories")
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Dedup & leakage removal")
    parser.add_argument("--near-threshold", type=float, default=0.85,
                        help="MinHash Jaccard >= this counts as a near-duplicate")
    parser.add_argument("--collapse-near", action="store_true", default=True,
                        help="collapse intra-pool near-duplicate clusters")
    parser.add_argument("--no-collapse-near", dest="collapse_near", action="store_false")
    parser.add_argument("--num-perm", type=int, default=64)
    args = parser.parse_args()

    config.ensure_dirs()
    out_dir = config.STEP_DIRS["step1"]
    pool = load_json(config.STEP_DIRS["step0"] / "merged_pool.json")
    n0 = len(pool)
    print(f"Loaded pool: {n0}")

    report: list[str] = ["# Step 1 - Dedup & Leakage Report", ""]

    # --- 1. Exact intra-pool dedup -----------------------------------------
    seen: set[str] = set()
    kept: list[dict] = []
    exact_removed = 0
    for rec in pool:
        h = rec["traj_hash"]
        if h in seen:
            exact_removed += 1
            continue
        seen.add(h)
        kept.append(rec)
    pool = kept
    report += [
        "## 1. Exact intra-pool dedup",
        f"- removed **{exact_removed}** exact-duplicate trajectories",
        f"- remaining: **{len(pool)}**",
        "",
    ]
    print(f"After exact dedup: {len(pool)} (-{exact_removed})")

    # --- MinHash signatures for pool + test --------------------------------
    hasher = MinHasher(num_perm=args.num_perm, seed=1)
    print("Computing MinHash signatures (pool)...")
    pool_texts = [normalize_masked(normalize_light(r["instruction"])) for r in pool]
    # note: instruction includes taxonomy header; mask+shingle still dominated by
    # the trajectory body, which is what we compare on.
    pool_masked = [normalize_masked(_traj(r)) for r in pool]
    pool_sig = hasher.signatures(pool_masked)
    print("Computing MinHash signatures (test)...")
    test_texts = load_test_texts()
    test_masked = [normalize_masked(t) for t in test_texts]
    test_light_hashes = {sha1(normalize_light(t)) for t in test_texts}
    test_sig = hasher.signatures(test_masked)

    # --- 2. Leakage vs test -------------------------------------------------
    # exact leakage
    exact_leak_idx = {
        i for i, r in enumerate(pool) if sha1(normalize_light(_traj(r))) in test_light_hashes
    }
    # near leakage via LSH over test signatures
    test_lsh = build_lsh(hasher, test_sig)
    near_leak_idx: set[int] = set()
    near_leak_examples: list[tuple[str, float]] = []
    for i, sig in enumerate(pool_sig):
        if i in exact_leak_idx:
            continue
        cand: set[int] = set()
        for key in hasher.band_keys(sig):
            cand.update(test_lsh.get(key, []))
        for j in cand:
            sim = jaccard(sig, test_sig[j])
            if sim >= args.near_threshold:
                near_leak_idx.add(i)
                if len(near_leak_examples) < 10:
                    near_leak_examples.append((pool[i]["uid"], round(sim, 3)))
                break

    leak_idx = exact_leak_idx | near_leak_idx
    report += [
        "## 2. Leakage vs test (ATBench300 + R-judge)",
        f"- exact leakage: **{len(exact_leak_idx)}**",
        f"- near leakage (Jaccard >= {args.near_threshold}): **{len(near_leak_idx)}**",
        f"- total removed: **{len(leak_idx)}**",
    ]
    if near_leak_examples:
        report.append(f"- near-leak examples (uid, sim): {near_leak_examples}")
    report.append("")
    print(f"Leakage: exact={len(exact_leak_idx)} near={len(near_leak_idx)}")

    survivors = [i for i in range(len(pool)) if i not in leak_idx]

    # --- 3. Intra-pool near-dup collapse -----------------------------------
    # Compared on SCENARIO text only (user turns + agent thoughts + tool names).
    # Full-trajectory comparison is dominated by tool-schema/environment
    # boilerplate and falsely merges unrelated scenarios (verified by spot
    # check). Candidates from MinHash/LSH are additionally verified with TRUE
    # Jaccard over shingle sets before merging.
    collapsed_removed = 0
    collapse_pairs: list[tuple[str, str, float]] = []
    if args.collapse_near:
        print("Computing scenario shingles for intra-pool collapse...")
        scen_masked = [
            normalize_masked(extract_scenario_text(_traj(pool[i]))) for i in survivors
        ]
        scen_shingles = [set(word_shingles(t)) for t in scen_masked]
        scen_sig = hasher.signatures(scen_masked)
        lsh = build_lsh(hasher, scen_sig)
        uf = UnionFind(len(survivors))
        checked: set[tuple[int, int]] = set()
        for members in lsh.values():
            if len(members) < 2:
                continue
            for a_pos in range(len(members)):
                for b_pos in range(a_pos + 1, len(members)):
                    a, b = members[a_pos], members[b_pos]
                    if (a, b) in checked:
                        continue
                    checked.add((a, b))
                    if pool[survivors[a]]["label"] != pool[survivors[b]]["label"]:
                        continue  # never merge across safe/unsafe
                    sa, sb = scen_shingles[a], scen_shingles[b]
                    if not sa or not sb:
                        continue
                    inter = len(sa & sb)
                    true_j = inter / (len(sa) + len(sb) - inter)
                    if true_j >= args.near_threshold:
                        uf.union(a, b)
                        if len(collapse_pairs) < 2000:
                            collapse_pairs.append(
                                (pool[survivors[a]]["uid"], pool[survivors[b]]["uid"],
                                 round(true_j, 3))
                            )
        reps: dict[int, int] = {}
        keep_positions: list[int] = []
        for pos in range(len(survivors)):
            root = uf.find(pos)
            if root not in reps:
                reps[root] = pos
                keep_positions.append(pos)
            else:
                collapsed_removed += 1
        survivors = [survivors[p] for p in keep_positions]
        save_json(collapse_pairs, out_dir / "collapse_pairs.json")
    report += [
        "## 3. Intra-pool near-dup collapse",
        f"- collapsed **{collapsed_removed}** template-clone duplicates"
        + ("" if args.collapse_near else " (disabled)"),
        "",
    ]
    print(f"Near-dup collapse: -{collapsed_removed}")

    # --- Save ---------------------------------------------------------------
    clean_pool = [pool[i] for i in survivors]
    clean_sig = pool_sig[survivors]
    save_json(clean_pool, out_dir / "clean_pool.json")
    np.save(config.STEP_DIRS["scores"] / "pool_signatures.npy", clean_sig)
    save_json([r["uid"] for r in clean_pool], config.STEP_DIRS["scores"] / "pool_signatures_uids.json")

    from collections import Counter
    label_dist = dict(Counter(r["label"] for r in clean_pool))
    report += [
        "## Result",
        f"- input pool: {n0}",
        f"- clean pool: **{len(clean_pool)}**  (labels: {label_dist})",
        f"- signatures cached: scores/pool_signatures.npy",
        "",
    ]
    write_text("\n".join(report), out_dir / "dedup_report.md")
    print("\n".join(report))
    print(f"\nWrote {out_dir / 'clean_pool.json'} ({len(clean_pool)} rows)")


def _traj(rec: dict) -> str:
    from common import extract_trajectory_from_instruction

    return extract_trajectory_from_instruction(rec["instruction"])


if __name__ == "__main__":
    main()
