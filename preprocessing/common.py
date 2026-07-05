#!/usr/bin/env python3
"""Shared utilities for the AgentDoG-Lite preprocessing pipeline.

Pure-Python + numpy text processing only. Nothing here loads a model or hits an
API, so every helper is safe to run on the local workstation.

Contents:
- JSON I/O helpers.
- Trajectory extraction (training instruction text -> trajectory; test JSON
  contents -> comparable text).
- Two-level text normalization (light + masked) for exact and near-duplicate
  detection.
- A compact MinHash + LSH implementation for scalable near-duplicate search.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------
def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_json(obj: Any, path: str | Path, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=indent)


def write_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


# ---------------------------------------------------------------------------
# Trajectory extraction
# ---------------------------------------------------------------------------
_TRAJ_RE = re.compile(r"<BEGIN TRAJECTORY>(.*?)<END TRAJECTORY>", re.DOTALL)


def extract_trajectory_from_instruction(instruction: str) -> str:
    """Pull the trajectory body out of a training-sample instruction.

    Training instructions look like: task text + <BEGIN CATEGORIZATION>...
    <END CATEGORIZATION> + <BEGIN TRAJECTORY>...<END TRAJECTORY> + output prompt.
    Only the trajectory body is comparable across samples, so we isolate it.
    Falls back to the full instruction if the markers are missing.
    """
    match = _TRAJ_RE.search(instruction or "")
    return match.group(1).strip() if match else (instruction or "").strip()


def test_sample_to_text(sample: dict) -> str:
    """Serialize a held-out test sample's `contents` into trajectory-like text.

    Mirrors the [USER]/[AGENT]/[ENV] shape of the training trajectory body so a
    training trajectory and a test trajectory describing the same scenario
    normalize to similar strings.
    """
    lines: list[str] = []
    for episode in sample.get("contents", []) or []:
        if not isinstance(episode, list):
            continue
        for msg in episode:
            role = msg.get("role", "?")
            if role == "user":
                lines.append(f"[USER]: {msg.get('content', '')}")
            elif role == "agent":
                thought = msg.get("thought", "") or ""
                action = msg.get("action", "") or ""
                content = msg.get("content", "") or ""
                lines.append(f"[AGENT]: {thought} {action} {content}".strip())
            elif role == "environment":
                content = msg.get("content", "")
                if isinstance(content, (dict, list)):
                    content = json.dumps(content, ensure_ascii=False)
                lines.append(f"[ENV]: {content}")
            else:
                lines.append(f"[{role}]: {msg}")
    return "\n".join(lines)


_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
_LEAD_TOOL_RE = re.compile(r"^([A-Za-z_][\w]*)\s*[\{(]")


def extract_scenario_text(traj: str) -> str:
    """Reduce a trajectory to its scenario-defining content for similarity.

    Trajectories embed large boilerplate blocks (tool JSON schemas, environment
    payloads) that dominate shingle-based similarity and cause unrelated
    scenarios to look near-identical. For duplicate detection we keep only:
      - [USER] turns (the task itself)
      - [THOUGHT] lines (agent reasoning)
      - tool NAMES from [ACTION] lines (not their argument payloads)
    and drop the Available-tools header, [ENVIRONMENT] bodies, and misc noise.
    """
    # drop profile / Available tools header if present
    marker = "=== Conversation History ==="
    pos = traj.find(marker)
    if pos != -1:
        traj = traj[pos + len(marker):]

    kept: list[str] = []
    for block in re.split(r"\n(?=\[)", traj):
        block = block.strip()
        if not block:
            continue
        upper = block[:15].upper()
        if upper.startswith("[USER]"):
            kept.append(block)
        elif upper.startswith("[THOUGHT]"):
            kept.append(block)
        elif upper.startswith("[AGENT]"):
            # inline agent content (may contain thought text before [ACTION])
            head = block.split("\n[", 1)[0]
            if len(head) > 10:
                kept.append(head)
        elif upper.startswith("[ACTION]"):
            m = _NAME_RE.search(block) or _LEAD_TOOL_RE.search(block[9:].lstrip())
            kept.append(f"[ACTION]: {m.group(1)}" if m else "[ACTION]")
        # [ENVIRONMENT]/[ENV] blocks dropped entirely
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_NUM_RE = re.compile(r"\d+")
# long alphanumeric blobs = api keys, tokens, ids, uuids
_TOKEN_RE = re.compile(r"\b[a-zA-Z0-9_]*[0-9][a-zA-Z0-9_]*\b")


def normalize_light(text: str) -> str:
    """Lowercase + whitespace collapse. Used for EXACT-duplicate hashing."""
    return _WS_RE.sub(" ", (text or "").lower()).strip()


def normalize_masked(text: str) -> str:
    """Heavier normalization for NEAR-duplicate detection.

    Masks values that vary between template clones (urls, emails, numbers,
    id/token blobs) so "same skeleton, different random values" collapses to the
    same string.
    """
    text = (text or "").lower()
    text = _URL_RE.sub("<url>", text)
    text = _EMAIL_RE.sub("<email>", text)
    text = _TOKEN_RE.sub("<tok>", text)
    text = _NUM_RE.sub("#", text)
    return _WS_RE.sub(" ", text).strip()


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# MinHash + LSH (compact, numpy-only)
# ---------------------------------------------------------------------------
_MERSENNE_P = (1 << 61) - 1  # prime for universal hashing


def word_shingles(text: str, k: int = 5) -> list[str]:
    words = text.split()
    if len(words) < k:
        return [text] if text else []
    return [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]


def _shingle_base_hashes(shingles: Iterable[str]) -> np.ndarray:
    """Hash each shingle to a 31-bit int (keeps a*x+b in int64 range)."""
    out = []
    for sh in shingles:
        digest = hashlib.blake2b(sh.encode("utf-8"), digest_size=4).digest()
        out.append(int.from_bytes(digest, "big") & 0x7FFFFFFF)
    return np.asarray(out, dtype=np.int64)


@dataclass
class MinHasher:
    """Deterministic MinHash signatures with LSH banding for candidate search."""

    num_perm: int = 64
    bands: int = 16
    k: int = 5
    seed: int = 1
    _a: np.ndarray = field(init=False)
    _b: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.num_perm % self.bands != 0:
            raise ValueError("num_perm must be divisible by bands")
        rng = np.random.default_rng(self.seed)
        self._a = rng.integers(1, 1 << 31, size=self.num_perm, dtype=np.int64)
        self._b = rng.integers(0, 1 << 31, size=self.num_perm, dtype=np.int64)

    @property
    def rows(self) -> int:
        return self.num_perm // self.bands

    def signature(self, text: str) -> np.ndarray:
        base = _shingle_base_hashes(word_shingles(text, self.k))
        if base.size == 0:
            return np.full(self.num_perm, _MERSENNE_P, dtype=np.int64)
        # (num_perm, m) = a[:,None]*base + b[:,None], min over shingles
        hashed = (self._a[:, None] * base[None, :] + self._b[:, None]) % _MERSENNE_P
        return hashed.min(axis=1)

    def signatures(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self.signature(t) for t in texts]) if texts else np.empty((0, self.num_perm), np.int64)

    def band_keys(self, sig: np.ndarray) -> list[str]:
        """Per-band bucket keys; two docs sharing any key are a candidate pair."""
        keys = []
        r = self.rows
        for band in range(self.bands):
            chunk = sig[band * r : (band + 1) * r]
            keys.append(f"{band}:" + hashlib.blake2b(chunk.tobytes(), digest_size=8).hexdigest())
        return keys


def jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Estimated Jaccard similarity from two MinHash signatures."""
    return float(np.mean(sig_a == sig_b))
