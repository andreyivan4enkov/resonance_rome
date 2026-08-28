#!/usr/bin/env python3
"""Iteration Y: test the hybrid idea -- "use resonance to decide WHICH
subspace matters, AlphaEdit's null-space projection to protect it" -- on
real COUNTERFACT cases. Four modes compared:
  standard        real ROME baseline (generic corpus covariance)
  ours            resonance-weighted covariance (this project's main result)
  null_generic    AlphaEdit-style null-space projection, generic corpus
  null_resonance  this project's hybrid: null space from resonance-weighted covariance
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `resonance_rome` is importable

import numpy as np
import torch

_DEFAULT_HF_CACHE = str(Path(__file__).resolve().parents[1] / ".hf_cache")
os.environ.setdefault("HF_HOME", _DEFAULT_HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", _DEFAULT_HF_CACHE)
os.environ.setdefault("HF_HUB_CACHE", _DEFAULT_HF_CACHE)

from resonance_rome import extract_peer_keys, rome_edit  # noqa: E402

COUNTERFACT = Path(os.environ.get(
    "COUNTERFACT_PATH",
    str(Path(__file__).resolve().parents[1] / "benchmarks" / "counterfact.json"),
))
HOTPOT = Path(os.environ.get(
    "HOTPOT_CORPUS_PATH",
    str(Path(__file__).resolve().parents[1] / "benchmarks" / "hotpot_dev_distractor_v1.json"),
))
LAYER = 4
RIDGE = 1.0
N_PEERS = 200
N_CASES = 10
SEED = 0
MODES = ["standard", "ours", "null_generic", "null_resonance"]


def load_real_sentences(n: int) -> list[str]:
    with HOTPOT.open(encoding="utf-8") as f:
        obj = json.load(f)
    out = []
    for item in obj[:n * 2]:
        ctx = item.get("context", [])
        if not ctx:
            continue
        title, sents = ctx[0]
        text = " ".join(sents).strip()
        if len(text.split()) >= 8:
            out.append(text)
        if len(out) >= n:
            break
    return out


def load_counterfact_cases(n: int, seed: int):
    with COUNTERFACT.open(encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(seed)
    shuffled = data[:]
    rng.shuffle(shuffled)
    sample = shuffled[:n]
    cases = []
    for d in sample:
        rw = d["requested_rewrite"]
        prompt = rw["prompt"].format(rw["subject"])
        cases.append(dict(
            prompt=prompt,
            target_new=" " + rw["target_new"]["str"],
            target_true=" " + rw["target_true"]["str"],
            paraphrases=d.get("paraphrase_prompts", [])[:2],
            neighbors=d.get("neighborhood_prompts", [])[:10],
        ))
    return cases


def first_token_id(tok, word):
    return tok(word, add_special_tokens=False)["input_ids"][0]


def prefers_new_over_true(model, tok, device, prompt, new_id, true_id):
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    return float(logits[new_id]) > float(logits[true_id])


def evaluate_case(model, tok, device, case):
    new_id = first_token_id(tok, case["target_new"])
    true_id = first_token_id(tok, case["target_true"])
    es = prefers_new_over_true(model, tok, device, case["prompt"], new_id, true_id)
    ps_hits = [prefers_new_over_true(model, tok, device, p, new_id, true_id) for p in case["paraphrases"]]
    ps = float(np.mean(ps_hits)) if ps_hits else float("nan")
    ns_hits = [not prefers_new_over_true(model, tok, device, p, new_id, true_id) for p in case["neighbors"]]
    ns = float(np.mean(ns_hits)) if ns_hits else float("nan")
    return es, ps, ns


def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cases = load_counterfact_cases(N_CASES, SEED)
    peer_sents = load_real_sentences(N_PEERS)
    print(f"Null-space hybrid test: {N_CASES} real COUNTERFACT cases, layer={LAYER}, modes={MODES}\n")

    ref_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    peer_K = extract_peer_keys(ref_model, tok, device, peer_sents, LAYER)
    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {m: [] for m in MODES}
    for ci, case in enumerate(cases):
        for mode in MODES:
            model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
            rome_edit(model, tok, device, LAYER, case["prompt"], case["target_new"], peer_K, mode, ridge=RIDGE)
            es, ps, ns = evaluate_case(model, tok, device, case)
            results[mode].append((es, ps, ns))
            print(f"case {ci:2d} [{mode:15s}] {case['prompt'][:35]!r:37s} ES={es!s:5s} PS={ps:.2f} NS={ns:.2f}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n=== NULL-SPACE HYBRID SUMMARY (real, n={N_CASES}, layer={LAYER}) ===")
    for mode in MODES:
        rs = results[mode]
        es_mean = np.mean([r[0] for r in rs])
        ps_mean = np.nanmean([r[1] for r in rs])
        ns_mean = np.nanmean([r[2] for r in rs])
        print(f"{mode:15s}: ES={es_mean:.3f}  PS={ps_mean:.3f}  NS={ns_mean:.3f}")


if __name__ == "__main__":
    main()
