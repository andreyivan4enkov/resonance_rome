#!/usr/bin/env python3
"""Iteration Z: "слой ниже -- оперативная память для слоя выше. Долгая
память -- свои веса, быстрая -- алиасинг от слоя ниже. Первый слой вместо
алиасинга имеет ввод." Two things tested together, both real:

(1) SCALE FIX for the null-space hybrid (iteration_y found generic and
    resonance null-space identical at N_PEERS=200 -- diagnosed cause: 200
    peers against a 3072-dim key space leaves ~93% of directions trivially
    zero regardless of weighting). N_PEERS raised to 2000 here.

(2) The new hierarchy hypothesis, operationalized literally (principle 10,
    already established: decimated subsample of the lower level, SUMMED not
    averaged): for each real peer sentence, build an "effective key" at
    layer L that is layer L's own real key PLUS a real, decimated (stride-
    subsampled) contribution from layer L-1 -- literally "this layer's own
    long-term structure + a compressed, aliased working-memory signal from
    the layer below" -- and compare covariance/null-space built from THIS
    hierarchical key against the flat, single-layer key used everywhere
    else in this project so far.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

_DEFAULT_HF_CACHE = str(Path(__file__).resolve().parents[1] / ".hf_cache")
os.environ.setdefault("HF_HOME", _DEFAULT_HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", _DEFAULT_HF_CACHE)
os.environ.setdefault("HF_HUB_CACHE", _DEFAULT_HF_CACHE)

from resonance_rome.core import null_space_projection, resonance_covariance, standard_covariance  # noqa: E402

COUNTERFACT = Path(os.environ.get(
    "COUNTERFACT_PATH",
    str(Path(__file__).resolve().parents[1] / "benchmarks" / "counterfact.json"),
))
HOTPOT = Path(os.environ.get(
    "HOTPOT_CORPUS_PATH",
    str(Path(__file__).resolve().parents[1] / "benchmarks" / "hotpot_dev_distractor_v1.json"),
))
LAYER = 4
LAYER_BELOW = LAYER - 1  # real "layer below" whose real activity gets decimated
RIDGE = 1.0
N_PEERS = 2000
N_CASES = 10
SEED = 0
DECIMATION_STRIDE = 4  # literal decimation factor, same convention as fractal_holographic_mixer.py
RATE_BUDGET = 0.2
V_STAR_STEPS = 30
V_STAR_LR = 0.5


def load_real_sentences(n: int) -> list[str]:
    with HOTPOT.open(encoding="utf-8") as f:
        obj = json.load(f)
    out = []
    for item in obj[:n * 3]:
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


def get_real_key_two_layers(model, tok, device, text, layer_hi, layer_lo):
    """Real GELU-activated MLP keys at BOTH layers, one real forward pass."""
    ids = tok(text, return_tensors="pt", truncation=True, max_length=64)["input_ids"].to(device)
    captured = {}

    def make_hook(name):
        def hook(module, inp, out):
            captured[name] = inp[0][0, -1].detach()
        return hook

    h_hi = model.transformer.h[layer_hi].mlp.c_proj.register_forward_hook(make_hook("hi"))
    h_lo = model.transformer.h[layer_lo].mlp.c_proj.register_forward_hook(make_hook("lo"))
    with torch.no_grad():
        model(ids)
    h_hi.remove()
    h_lo.remove()
    return captured["hi"], captured["lo"], ids


def decimate(v: np.ndarray, stride: int) -> np.ndarray:
    """Literal decimation: real stride-subsample, zero elsewhere (principle 10:
    what a higher level sees is a decimated subsample of the lower level's
    real activity -- summed, not averaged, into the higher level's own key)."""
    out = np.zeros_like(v)
    out[::stride] = v[::stride]
    return out


def extract_peer_keys_hierarchical(model, tok, device, sentences):
    """Returns (flat_keys, hierarchical_keys), both (n_peers, d)."""
    flat, hier = [], []
    for s in sentences:
        k_hi, k_lo, _ = get_real_key_two_layers(model, tok, device, s, LAYER, LAYER_BELOW)
        k_hi_np = k_hi.cpu().double().numpy()
        k_lo_np = k_lo.cpu().double().numpy()
        flat.append(k_hi_np)
        hier.append(k_hi_np + decimate(k_lo_np, DECIMATION_STRIDE))  # SUM, not average
    return np.stack(flat), np.stack(hier)


def get_real_key_single(model, tok, device, text, layer):
    ids = tok(text, return_tensors="pt")["input_ids"].to(device)
    captured = {}

    def hook(module, inp, out):
        captured["k"] = inp[0][0, -1].detach()

    h = model.transformer.h[layer].mlp.c_proj.register_forward_hook(hook)
    with torch.no_grad():
        model(ids)
    h.remove()
    return captured["k"], ids


def find_v_star(model, tok, device, layer, prompt, target_word):
    import torch.nn.functional as F
    c_proj = model.transformer.h[layer].mlp.c_proj
    k_star, ids = get_real_key_single(model, tok, device, prompt, layer)
    v_orig = (k_star @ c_proj.weight.detach())
    delta_v = torch.zeros_like(v_orig, requires_grad=True)

    def add_hook(module, inp, out):
        return out + delta_v

    h = c_proj.register_forward_hook(add_hook)
    opt = torch.optim.Adam([delta_v], lr=V_STAR_LR)
    target_id = tok(target_word, add_special_tokens=False)["input_ids"][0]
    for _ in range(V_STAR_STEPS):
        logits = model(ids).logits[0, -1]
        loss = -F.log_softmax(logits, dim=-1)[target_id]
        opt.zero_grad()
        loss.backward()
        opt.step()
    h.remove()
    v_star = (v_orig + delta_v).detach()
    return k_star, v_orig, v_star


def apply_edit(model, tok, device, prompt, target_word, peer_keys, hier_mode, cov_mode,
               k_star_for_hier=None):
    """cov_mode: 'standard' | 'ours' | 'null_generic' | 'null_resonance'
    hier_mode: 'flat' | 'hierarchical' -- which peer_keys were passed in."""
    c_proj = model.transformer.h[LAYER].mlp.c_proj
    k_star, v_orig, v_star = find_v_star(model, tok, device, LAYER, prompt, target_word)
    k_star_np = k_star.cpu().double().numpy()
    d = peer_keys.shape[1]

    # In hierarchical mode, the target's OWN effective key also needs the
    # same decimated-lower-layer contribution added, for a fair, consistent
    # comparison against real peer keys built the same way.
    if hier_mode == "hierarchical":
        _, k_lo_target, _ = get_real_key_two_layers(model, tok, device, prompt, LAYER, LAYER_BELOW)
        k_target_np = k_star_np + decimate(k_lo_target.cpu().double().numpy(), DECIMATION_STRIDE)
    else:
        k_target_np = k_star_np

    if cov_mode == "standard":
        C = standard_covariance(peer_keys)
    elif cov_mode == "ours":
        C = resonance_covariance(peer_keys, k_target_np[None, :], RATE_BUDGET)
    else:
        M = (standard_covariance(peer_keys) if cov_mode == "null_generic"
             else resonance_covariance(peer_keys, k_target_np[None, :], RATE_BUDGET))
        P = null_space_projection(M, eigenvalue_threshold_frac=1e-2)
        C = 1e4 * (np.eye(d) - P)

    C_reg = C + RIDGE * np.eye(d)
    C_inv = np.linalg.inv(C_reg)
    Cinv_k = C_inv @ k_star_np  # the EDIT itself still uses the real, un-decimated k_star for its own key
    denom = float(k_star_np @ Cinv_k)
    delta_out = v_star.cpu().double().numpy() - v_orig.cpu().double().numpy()
    Delta = np.outer(delta_out, Cinv_k) / denom
    with torch.no_grad():
        c_proj.weight += torch.tensor(Delta.T, dtype=c_proj.weight.dtype, device=device)


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
    print(f"N_PEERS={N_PEERS}, layer={LAYER} (below={LAYER_BELOW}), decimation_stride={DECIMATION_STRIDE}\n")

    ref_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    flat_peer_K, hier_peer_K = extract_peer_keys_hierarchical(ref_model, tok, device, peer_sents)
    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"real peer keys extracted: flat={flat_peer_K.shape}, hierarchical={hier_peer_K.shape}\n")

    configs = [
        ("flat", "null_generic", flat_peer_K),
        ("flat", "null_resonance", flat_peer_K),
        ("hierarchical", "null_generic", hier_peer_K),
        ("hierarchical", "null_resonance", hier_peer_K),
    ]

    results = {f"{h}/{c}": [] for h, c, _ in configs}
    for ci, case in enumerate(cases):
        for hier_mode, cov_mode, peer_K in configs:
            model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
            apply_edit(model, tok, device, case["prompt"], case["target_new"], peer_K, hier_mode, cov_mode)
            es, ps, ns = evaluate_case(model, tok, device, case)
            key = f"{hier_mode}/{cov_mode}"
            results[key].append((es, ps, ns))
            print(f"case {ci:2d} [{key:24s}] {case['prompt'][:32]!r:34s} ES={es!s:5s} PS={ps:.2f} NS={ns:.2f}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n=== HIERARCHICAL ALIASING TEST (real, n={N_CASES}, N_PEERS={N_PEERS}) ===")
    for key, rs in results.items():
        es_mean = np.mean([r[0] for r in rs])
        ps_mean = np.nanmean([r[1] for r in rs])
        ns_mean = np.nanmean([r[2] for r in rs])
        print(f"{key:24s}: ES={es_mean:.3f}  PS={ps_mean:.3f}  NS={ns_mean:.3f}")


if __name__ == "__main__":
    main()
