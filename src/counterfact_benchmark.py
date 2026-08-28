#!/usr/bin/env python3
"""Iteration U: real COUNTERFACT benchmark (Meng et al. 2022's own dataset,
downloaded live from rome.baulab.info, 21919 real cases) -- standard vs our
resonance C, using the field's own real metrics:

  ES (Efficacy Score): after editing, does the model prefer target_new over
      target_true on the EXACT edited prompt?
  PS (Paraphrase Score): same check, on real held-out paraphrase_prompts
      (generalization -- does the edit hold under rewording?)
  NS (Neighborhood Score): on real neighborhood_prompts (DIFFERENT subjects,
      same relation), does the model STILL prefer target_true over
      target_new? (specificity -- unrelated facts must NOT flip)

Simplification, disclosed: first-BPE-token comparison only (matches this
project's convention throughout), not full multi-token teacher-forced
scoring as some later papers use.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_DEFAULT_HF_CACHE = str(Path(__file__).resolve().parents[1] / ".hf_cache")
os.environ.setdefault("HF_HOME", _DEFAULT_HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", _DEFAULT_HF_CACHE)
os.environ.setdefault("HF_HUB_CACHE", _DEFAULT_HF_CACHE)

# Real COUNTERFACT (Meng et al. 2022), download once via:
#   curl -L -o benchmarks/counterfact.json https://rome.baulab.info/data/dsets/counterfact.json
COUNTERFACT = Path(os.environ.get(
    "COUNTERFACT_PATH",
    str(Path(__file__).resolve().parents[1] / "benchmarks" / "counterfact.json"),
))
HOTPOT = Path(os.environ.get(
    "HOTPOT_CORPUS_PATH",
    str(Path(__file__).resolve().parents[1] / "benchmarks" / "hotpot_dev_distractor_v1.json"),
))
LAYER = 9
RATE_BUDGET = 0.2
N_PEERS = 200
V_STAR_STEPS = 30
V_STAR_LR = 0.5
RIDGE = 1.0
N_CASES = 20
SEED = 0


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
    sample = rng.sample(data, n)
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


def get_real_key(model, tok, device, text, layer):
    ids = tok(text, return_tensors="pt")["input_ids"].to(device)
    captured = {}

    def hook(module, inp, out):
        captured["k"] = inp[0][0, -1].detach()

    h = model.transformer.h[layer].mlp.c_proj.register_forward_hook(hook)
    with torch.no_grad():
        model(ids)
    h.remove()
    return captured["k"], ids


def first_token_id(tok, word):
    ids = tok(word, add_special_tokens=False)["input_ids"]
    return ids[0]


def prefers_new_over_true(model, tok, device, prompt, new_id, true_id):
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    return float(logits[new_id]) > float(logits[true_id])


def real_topology_and_budget(V: np.ndarray):
    n = V.shape[0]
    Vn = V / (np.linalg.norm(V, axis=-1, keepdims=True) + 1e-12)
    cos = Vn @ Vn.T
    topo = np.maximum(cos, 0.0)
    np.fill_diagonal(topo, 0.0)
    budget0 = np.linalg.norm(V, axis=-1)
    cost = -topo * 1.0
    b_i, b_j = budget0.reshape(n, 1), budget0.reshape(1, n)
    weak_is_i = b_i <= b_j
    weak_budget = np.where(weak_is_i, b_i, b_j)
    transfer = np.minimum(RATE_BUDGET * np.maximum(cost, 0.0), weak_budget)
    sign = np.where(weak_is_i, -1.0, 1.0)
    budget_final = budget0 + (sign * transfer).sum(axis=1)
    return topo, budget_final


def rome_edit(model, tok, device, prompt, target_new, peer_K, mode):
    c_proj = model.transformer.h[LAYER].mlp.c_proj
    k_star, ids = get_real_key(model, tok, device, prompt, LAYER)
    v_orig = (k_star @ c_proj.weight.detach())
    delta_v = torch.zeros_like(v_orig, requires_grad=True)

    def add_hook(module, inp, out):
        return out + delta_v

    h = c_proj.register_forward_hook(add_hook)
    opt = torch.optim.Adam([delta_v], lr=V_STAR_LR)
    new_id = first_token_id(tok, target_new)
    for _ in range(V_STAR_STEPS):
        logits = model(ids).logits[0, -1]
        loss = -F.log_softmax(logits, dim=-1)[new_id]
        opt.zero_grad()
        loss.backward()
        opt.step()
    h.remove()
    v_star = (v_orig + delta_v).detach()

    d = peer_K.shape[1]
    C_std = (peer_K.T @ peer_K) / peer_K.shape[0]
    if mode == "standard":
        C = C_std
    else:
        k_star_np = k_star.cpu().double().numpy()
        V_for_topo = np.vstack([peer_K, k_star_np[None, :]])
        topo, budget_final = real_topology_and_budget(V_for_topo)
        w = topo[-1, :-1] * budget_final[:-1]
        C = (peer_K * w[:, None]).T @ peer_K

    C_reg = C + RIDGE * np.eye(d)
    C_inv = np.linalg.inv(C_reg)
    k_star_np = k_star.cpu().double().numpy()
    Cinv_k = C_inv @ k_star_np
    denom = float(k_star_np @ Cinv_k)
    delta_out = (v_star.cpu().double().numpy() - v_orig.cpu().double().numpy())
    Delta = np.outer(delta_out, Cinv_k) / denom
    with torch.no_grad():
        c_proj.weight += torch.tensor(Delta.T, dtype=c_proj.weight.dtype, device=device)


def evaluate_case(model, tok, device, case):
    new_id = first_token_id(tok, case["target_new"])
    true_id = first_token_id(tok, case["target_true"])
    es = prefers_new_over_true(model, tok, device, case["prompt"], new_id, true_id)
    ps_hits = [prefers_new_over_true(model, tok, device, p, new_id, true_id) for p in case["paraphrases"]]
    ps = float(np.mean(ps_hits)) if ps_hits else float("nan")
    ns_hits = [not prefers_new_over_true(model, tok, device, p, new_id, true_id) for p in case["neighbors"]]
    ns = float(np.mean(ns_hits)) if ns_hits else float("nan")
    return es, ps, ns


def extract_peer_keys(model, tok, device, sentences):
    keys = []
    for s in sentences:
        k, _ = get_real_key(model, tok, device, s, LAYER)
        keys.append(k.cpu().double().numpy())
    return np.stack(keys)


def sweep_ridge_and_layer(tok, device, peer_sents_all):
    """Quick diagnostic (mode=standard only, 5 cases) to find a RIDGE/layer
    regime with reasonable absolute NS before re-running the full standard-
    vs-ours comparison -- iteration_u's first pass used a fixed RIDGE=1.0 at
    LAYER=9 (both hand-picked from our OWN simple-template experiments, not
    tuned for real, diverse COUNTERFACT facts) and got catastrophic NS for
    BOTH modes (0.005 / 0.065)."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    global LAYER, RIDGE
    cases = load_counterfact_cases(5, SEED + 1)  # different seed, held out from the main n=20 sample
    candidates = [(layer, ridge) for layer in [4, 6, 9] for ridge in [1.0, 100.0, 10000.0]]
    print("=== RIDGE/layer sweep (mode=standard, n=5 held-out cases) ===")
    sweep_results = []
    for layer, ridge in candidates:
        LAYER, RIDGE = layer, ridge
        ref_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
        peer_K = extract_peer_keys(ref_model, tok, device, peer_sents_all)
        del ref_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        es_l, ps_l, ns_l = [], [], []
        for case in cases:
            model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
            rome_edit(model, tok, device, case["prompt"], case["target_new"], peer_K, "standard")
            es, ps, ns = evaluate_case(model, tok, device, case)
            es_l.append(es); ps_l.append(ps); ns_l.append(ns)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        es_m, ps_m, ns_m = np.mean(es_l), np.nanmean(ps_l), np.nanmean(ns_l)
        print(f"  layer={layer:2d} ridge={ridge:8.1f}: ES={es_m:.3f} PS={ps_m:.3f} NS={ns_m:.3f}")
        sweep_results.append((layer, ridge, es_m, ps_m, ns_m))

    # Pick the best NS among configs that still keep ES reasonably high (>=0.6) --
    # a config with perfect NS but ES=0 (edit does nothing) is not a real win.
    viable = [r for r in sweep_results if r[2] >= 0.6] or sweep_results
    best = max(viable, key=lambda r: r[4])
    print(f"  -> chosen: layer={best[0]} ridge={best[1]}\n")
    return best[0], best[1]


def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    peer_sents = load_real_sentences(N_PEERS)
    global LAYER, RIDGE
    LAYER, RIDGE = sweep_ridge_and_layer(tok, device, peer_sents)

    cases = load_counterfact_cases(N_CASES, SEED)
    print(f"\nreal COUNTERFACT: {N_CASES} cases (seed={SEED}), layer={LAYER}, ridge={RIDGE}\n")

    ref_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    peer_K = extract_peer_keys(ref_model, tok, device, peer_sents)
    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {"standard": [], "ours": []}
    for ci, case in enumerate(cases):
        for mode in ["standard", "ours"]:
            model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
            rome_edit(model, tok, device, case["prompt"], case["target_new"], peer_K, mode)
            es, ps, ns = evaluate_case(model, tok, device, case)
            results[mode].append((es, ps, ns))
            print(f"case {ci:2d} [{mode:8s}] subj-prompt={case['prompt'][:40]!r:42s} "
                  f"ES={es!s:5s} PS={ps:.2f} NS={ns:.2f}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n=== COUNTERFACT SUMMARY (real, n={}, layer={}, ridge={}) ===".format(N_CASES, LAYER, RIDGE))
    for mode in ["standard", "ours"]:
        rs = results[mode]
        es_mean = np.mean([r[0] for r in rs])
        ps_mean = np.nanmean([r[1] for r in rs])
        ns_mean = np.nanmean([r[2] for r in rs])
        print(f"{mode:10s}: ES={es_mean:.3f}  PS={ps_mean:.3f}  NS={ns_mean:.3f}")


if __name__ == "__main__":
    main()
