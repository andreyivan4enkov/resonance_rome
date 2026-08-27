#!/usr/bin/env python3
"""Iteration Q: "разложить глобальный факт на множество точечных, которые мы
можем эффективно применить" -- real, MEMIT-style (Meng et al., Mass-Editing
Memory in a Transformer) distributed edit: instead of one big rank-1 update
at a SINGLE layer, split the total required residual-stream change across
SEVERAL layers, each contributing a smaller local edit -- and, novel here,
each layer uses WHICHEVER covariance (standard corpus C, or our topology+
budget C) the self-calibrated Рефлексия/autopoiesis rule from iteration_p
found best AT THAT LAYER, not the same C everywhere.

Literal definition:
  Pick a real layer range L = [l0..l1]. Compute the TOTAL required output
  change Delta_total = v*_at_l1 - v_orig_at_l1 via the standard ROME v*
  optimization at the LAST layer in the range (as before).
  Split evenly: each layer l in L gets a local target
    v*_l = v_orig_l + Delta_total / len(L)
  and solves the SAME closed-form ROME edit at l, using layer l's OWN
  self-calibrated best covariance (from iteration_p's real per-layer K>=S
  winners), not one fixed choice for the whole range.
  Compare against: a single-layer edit at l1 alone carrying the FULL
  Delta_total with l1's own best C (the iteration_o/p baseline).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HOME", r"D:\RLM\_external\sandbox\hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", r"D:\RLM\_external\sandbox\hf_cache")
os.environ.setdefault("HF_HUB_CACHE", r"D:\RLM\_external\sandbox\hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

HOTPOT = Path(r"D:\RLM\benchmarks\hotpot_dev_distractor_v1.json")
N_LAYERS = 12
RATE_BUDGET = 0.2
N_PEERS = 150
N_HELDOUT = 200
V_STAR_STEPS = 30
V_STAR_LR = 0.5
RIDGE = 1.0

# Self-calibrated per-layer winners from iteration_p's 6-fact real sweep.
BEST_C = {0: "ours", 1: "standard", 2: "ours", 3: "ours", 4: "ours", 5: "ours",
          6: "ours", 7: "ours", 8: "ours", 9: "ours", 10: "ours", 11: "ours"}

DECOMP_LAYERS = [1, 2, 3, 4]  # deliberately spans BOTH a "standard"-winning and "ours"-winning layer

NEW_FACTS = [
    ("The secret code word is", " banana"),
    ("The magic password is", " lighthouse"),
    ("The spy's codename is", " compass"),
]

FACT_PROMPTS = [
    ("The opposite of black is", " white"), ("The opposite of hot is", " cold"),
    ("The opposite of up is", " down"), ("The opposite of left is", " right"),
    ("The king and queen sat on the", " throne"), ("The boy and the", " girl"),
    ("The man and the", " woman"), ("One, two, three,", " four"),
    ("Monday, Tuesday,", " Wednesday"), ("Once upon a", " time"),
    ("The Earth orbits the", " Sun"), ("A doctor works in a", " hospital"),
    ("Thank you very", " much"), ("Nice to meet", " you"),
]


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


def real_perplexity(model, tok, sentences, device):
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for s in sentences:
            ids = tok(s, return_tensors="pt", truncation=True, max_length=64)["input_ids"].to(device)
            if ids.shape[1] < 2:
                continue
            logits = model(ids).logits[0]
            logp = F.log_softmax(logits.float(), dim=-1)
            for pos in range(1, ids.shape[1]):
                total_nll += -float(logp[pos - 1, ids[0, pos]])
                total_tok += 1
    return float(np.exp(total_nll / total_tok))


def fact_check(model, tok, device):
    correct = 0
    with torch.no_grad():
        for prompt, expected in FACT_PROMPTS:
            ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
            logits = model(ids).logits[0, -1]
            top_tok = tok.decode([int(torch.argmax(logits))])
            correct += int(top_tok.strip().lower() == expected.strip().lower())
    return correct


def bare_prompt_check(model, tok, device, prompt, answer):
    with torch.no_grad():
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
        logits = model(ids).logits[0, -1]
        top1 = tok.decode([int(torch.argmax(logits))])
    return top1.strip().lower() == answer.strip().lower()


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


def extract_all_layer_keys(model, tok, device, sentences, n_layers):
    captured = {li: [] for li in range(n_layers)}
    hooks = []

    def make_hook(li):
        def hook(module, inp, out):
            captured[li].append(inp[0][0, -1].detach().cpu().double().numpy())
        return hook

    for li in range(n_layers):
        hooks.append(model.transformer.h[li].mlp.c_proj.register_forward_hook(make_hook(li)))
    with torch.no_grad():
        for s in sentences:
            ids = tok(s, return_tensors="pt", truncation=True, max_length=64)["input_ids"].to(device)
            model(ids)
    for h in hooks:
        h.remove()
    return {li: np.stack(v) for li, v in captured.items()}


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


def apply_rome_edit_at_layer(model, tok, device, layer, K, mode, prompt, k_star, v_target):
    """Apply the literal ROME closed-form edit at `layer`, targeting v_target
    (an ALREADY-DECIDED local output target -- not re-optimized here)."""
    c_proj = model.transformer.h[layer].mlp.c_proj
    v_orig = (k_star @ c_proj.weight.detach())
    d = K.shape[1]

    C_std = (K.T @ K) / K.shape[0]
    if mode == "standard":
        C = C_std
    else:
        k_star_np = k_star.cpu().double().numpy()
        V_for_topo = np.vstack([K, k_star_np[None, :]])
        topo, budget_final = real_topology_and_budget(V_for_topo)
        w = topo[-1, :-1] * budget_final[:-1]
        C = (K * w[:, None]).T @ K

    C_reg = C + RIDGE * np.eye(d)
    C_inv = np.linalg.inv(C_reg)
    k_star_np = k_star.cpu().double().numpy()
    Cinv_k = C_inv @ k_star_np
    denom = float(k_star_np @ Cinv_k)
    delta_out = (v_target.cpu().double().numpy() - v_orig.cpu().double().numpy())
    Delta = np.outer(delta_out, Cinv_k) / denom
    with torch.no_grad():
        c_proj.weight += torch.tensor(Delta.T, dtype=c_proj.weight.dtype, device=device)


def compute_total_delta(model, tok, device, prompt, answer, last_layer):
    """Real v* optimization at the LAST layer of the decomposition range --
    this defines the TOTAL residual-stream change the whole edit must
    achieve, before splitting it across layers."""
    c_proj = model.transformer.h[last_layer].mlp.c_proj
    k_star, ids = get_real_key(model, tok, device, prompt, last_layer)
    v_orig = (k_star @ c_proj.weight.detach())
    delta_v = torch.zeros_like(v_orig, requires_grad=True)

    def add_hook(module, inp, out):
        return out + delta_v

    h = c_proj.register_forward_hook(add_hook)
    opt = torch.optim.Adam([delta_v], lr=V_STAR_LR)
    answer_id = tok(answer, add_special_tokens=False)["input_ids"][0]
    for _ in range(V_STAR_STEPS):
        logits = model(ids).logits[0, -1]
        loss = -F.log_softmax(logits, dim=-1)[answer_id]
        opt.zero_grad()
        loss.backward()
        opt.step()
    h.remove()
    return (v_orig + delta_v).detach() - v_orig, v_orig


def run_single_layer_baseline(model, tok, device, all_layer_keys, prompt, answer, layer):
    total_delta, v_orig = compute_total_delta(model, tok, device, prompt, answer, layer)
    v_target = v_orig + total_delta
    k_star, _ = get_real_key(model, tok, device, prompt, layer)
    mode = BEST_C[layer]
    apply_rome_edit_at_layer(model, tok, device, layer, all_layer_keys[layer], mode, prompt, k_star, v_target)


def run_memit_style_decomposition(model, tok, device, all_layer_keys, prompt, answer, layers):
    last_layer = layers[-1]
    total_delta, _ = compute_total_delta(model, tok, device, prompt, answer, last_layer)
    per_layer_delta = total_delta / len(layers)
    for layer in layers:
        k_star, _ = get_real_key(model, tok, device, prompt, layer)
        c_proj = model.transformer.h[layer].mlp.c_proj
        v_orig_l = (k_star @ c_proj.weight.detach())
        v_target_l = v_orig_l + per_layer_delta
        mode = BEST_C[layer]
        apply_rome_edit_at_layer(model, tok, device, layer, all_layer_keys[layer], mode, prompt, k_star, v_target_l)


def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_sents = load_real_sentences(N_HELDOUT + N_PEERS)
    heldout_sents = real_sents[:N_HELDOUT]
    peer_sents = real_sents[N_HELDOUT:N_HELDOUT + N_PEERS]
    print(f"decomposition layers={DECOMP_LAYERS}, per-layer best C={ {l: BEST_C[l] for l in DECOMP_LAYERS} }\n")

    ref_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    all_layer_keys = extract_all_layer_keys(ref_model, tok, device, peer_sents, N_LAYERS)
    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for prompt, answer in NEW_FACTS:
        print(f"{'#'*10} FACT: {prompt!r} -> {answer!r} {'#'*10}")

        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
        ppl0 = real_perplexity(model, tok, heldout_sents, device)
        facts0 = fact_check(model, tok, device)
        run_single_layer_baseline(model, tok, device, all_layer_keys, prompt, answer, DECOMP_LAYERS[-1])
        ok_single = bare_prompt_check(model, tok, device, prompt, answer)
        ppl_single = real_perplexity(model, tok, heldout_sents, device)
        facts_single = fact_check(model, tok, device)
        print(f"  SINGLE-LAYER (layer={DECOMP_LAYERS[-1]}, C={BEST_C[DECOMP_LAYERS[-1]]}): "
              f"ok={ok_single} ppl {ppl0:.2f}->{ppl_single:.2f} (delta={ppl_single-ppl0:+.4f}) "
              f"facts={facts_single}/{len(FACT_PROMPTS)}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
        ppl0b = real_perplexity(model, tok, heldout_sents, device)
        run_memit_style_decomposition(model, tok, device, all_layer_keys, prompt, answer, DECOMP_LAYERS)
        ok_memit = bare_prompt_check(model, tok, device, prompt, answer)
        ppl_memit = real_perplexity(model, tok, heldout_sents, device)
        facts_memit = fact_check(model, tok, device)
        print(f"  MEMIT-STYLE ({len(DECOMP_LAYERS)} layers, per-layer best C): "
              f"ok={ok_memit} ppl {ppl0b:.2f}->{ppl_memit:.2f} (delta={ppl_memit-ppl0b:+.4f}) "
              f"facts={facts_memit}/{len(FACT_PROMPTS)}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
