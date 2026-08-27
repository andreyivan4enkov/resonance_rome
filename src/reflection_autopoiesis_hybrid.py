#!/usr/bin/env python3
"""Iteration P: "первые слои недостаточно 'размыты'... сделай гибрид лучшего,
протестируй на большей выборке новых фактов... возьми Рефлексию и логику
автопоэзиса, адаптировав под нашу задачу."

Real, literal adaptation of NeuroStudio_v4/backend/blanket_interface.py's
`autopoiesis_k_ge_s_check` (K = mean real improvement over committed-and-
improved attempts, S = mean real regression magnitude over any-regression
attempts, k_ge_s <=> K>=S "operational closure") and `_dynamic_leakage_
threshold_pct` (a boundary's own accumulated real history decides its
threshold, not a hand-picked constant) -- applied here to decide, PER LAYER,
whether "standard" ROME (generic corpus C) or "ours" (topology+budget C) is
trusted, from REAL accumulated evidence across MULTIPLE new facts, instead
of the hand-picked "layers 0-1 = std" rule from iteration_o's single-fact
result.

Literal definition:
  For each (fact, layer, mode) real edit attempt:
    actual_delta = -(ppl_after - ppl_before) / ppl_before   (positive = the
      edit did LESS relative perplexity damage than baseline; harm_is_
      positive convention flipped to match "higher is better" here)
  Per (layer, mode), over all facts' attempts at that layer:
    K_set = {attempts with actual_delta > 0}, K = mean(actual_delta over K_set)
    S_set = {attempts with actual_delta < 0}, S = mean(|actual_delta| over S_set)
    margin = K - S          (literal K>=S operational-closure margin)
  Winning mode at a layer = the one with the higher real margin, computed
  from REAL accumulated attempts across facts -- not a layer-index rule
  decided in advance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import os

_DEFAULT_HF_CACHE = str(Path(__file__).resolve().parents[1] / ".hf_cache")
os.environ.setdefault("HF_HOME", _DEFAULT_HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", _DEFAULT_HF_CACHE)
os.environ.setdefault("HF_HUB_CACHE", _DEFAULT_HF_CACHE)
# NOTE: not forcing HF_HUB_OFFLINE=1 -- a fresh clone needs to download GPT-2
# once; set HF_HUB_OFFLINE=1 yourself once it's cached if you want that.

HOTPOT = Path(os.environ.get(
    "HOTPOT_CORPUS_PATH",
    str(Path(__file__).resolve().parents[1] / "benchmarks" / "hotpot_dev_distractor_v1.json"),
))
N_LAYERS = 12
RATE_BUDGET = 0.2
N_PEERS = 150
N_HELDOUT = 150  # reduced from 400 for tractability across a multi-fact x multi-layer sweep
V_STAR_STEPS = 30
V_STAR_LR = 0.5
RIDGE = 1.0

NEW_FACTS = [
    # all answers verified single-BPE-token (with leading space) -- avoids the
    # confound found live: "thunderclap" split into 3 tokens ([' thunder',
    # 'cl','ap']), and checking/optimizing only the first fragment made that
    # fact fail identically on every layer/mode, uninformative for comparison.
    ("The secret code word is", " banana"),
    ("The magic password is", " lighthouse"),
    ("The spy's codename is", " compass"),
    ("The treasure is hidden inside the", " pumpkin"),
    ("Her favorite instrument is the", " trumpet"),
    ("The lost ring was found near the", " volcano"),
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


def rome_edit(model, tok, device, layer, K, mode, prompt, answer):
    c_proj = model.transformer.h[layer].mlp.c_proj
    k_star, ids = get_real_key(model, tok, device, prompt, layer)
    v_orig = (k_star @ c_proj.weight.detach())

    delta_v = torch.zeros_like(v_orig, requires_grad=True)

    def add_hook(module, inp, out):
        return out + delta_v

    h = c_proj.register_forward_hook(add_hook)
    opt = torch.optim.Adam([delta_v], lr=V_STAR_LR)
    answer_id = tok(answer, add_special_tokens=False)["input_ids"][0]
    loss = None
    for _ in range(V_STAR_STEPS):
        logits = model(ids).logits[0, -1]
        loss = -F.log_softmax(logits, dim=-1)[answer_id]
        opt.zero_grad()
        loss.backward()
        opt.step()
    h.remove()
    v_star = (v_orig + delta_v).detach()
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
    delta_out = (v_star.cpu().double().numpy() - v_orig.cpu().double().numpy())
    Delta = np.outer(delta_out, Cinv_k) / denom
    with torch.no_grad():
        c_proj.weight += torch.tensor(Delta.T, dtype=c_proj.weight.dtype, device=device)


def k_ge_s_margin(deltas: list[float]) -> dict:
    """Literal adaptation of blanket_interface.autopoiesis_k_ge_s_check to a
    plain list of real actual_delta values (all "committed" here -- every
    attempt in this sweep is a real, applied edit, not a proposal that could
    be rejected before being tried)."""
    k_vals = [d for d in deltas if d > 0.0]
    s_vals = [-d for d in deltas if d < 0.0]
    k = (sum(k_vals) / len(k_vals)) if k_vals else 0.0
    s = (sum(s_vals) / len(s_vals)) if s_vals else 0.0
    return {"k": k, "s": s, "margin": k - s, "n": len(deltas)}


def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_sents = load_real_sentences(N_HELDOUT + N_PEERS)
    heldout_sents = real_sents[:N_HELDOUT]
    peer_sents = real_sents[N_HELDOUT:N_HELDOUT + N_PEERS]
    print(f"n_heldout={len(heldout_sents)}, n_peers={len(peer_sents)}, "
          f"{len(NEW_FACTS)} facts x {N_LAYERS} layers x 2 modes\n")

    ref_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    all_layer_keys = extract_all_layer_keys(ref_model, tok, device, peer_sents, N_LAYERS)
    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # attempts[layer][mode] = list of real actual_delta values, one per fact
    attempts = {layer: {"standard": [], "ours": []} for layer in range(N_LAYERS)}
    per_fact_ok = {layer: {"standard": [], "ours": []} for layer in range(N_LAYERS)}

    for prompt, answer in NEW_FACTS:
        print(f"{'#'*10} FACT: {prompt!r} -> {answer!r} {'#'*10}")
        for layer in range(N_LAYERS):
            K = all_layer_keys[layer]
            for mode in ["standard", "ours"]:
                model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
                ppl0 = real_perplexity(model, tok, heldout_sents, device)
                rome_edit(model, tok, device, layer, K, mode, prompt, answer)
                ok1 = bare_prompt_check(model, tok, device, prompt, answer)
                ppl1 = real_perplexity(model, tok, heldout_sents, device)
                facts1 = fact_check(model, tok, device)
                actual_delta = -(ppl1 - ppl0) / ppl0
                attempts[layer][mode].append(actual_delta)
                per_fact_ok[layer][mode].append(ok1)
                print(f"  layer={layer:2d} [{mode:8s}] ok={ok1!s:5s} ppl {ppl0:.2f}->{ppl1:.2f} "
                      f"delta={actual_delta:+.4f} facts={facts1}/{len(FACT_PROMPTS)}")
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        print()

    print("=== Рефлексия/автопоэзис: K>=S margin per layer, real accumulated history ===")
    print(f"{'layer':>5} | {'std_K':>7} {'std_S':>7} {'std_margin':>10} {'std_ok_rate':>11} | "
          f"{'ours_K':>7} {'ours_S':>7} {'ours_margin':>11} {'ours_ok_rate':>12} | winner")
    winners = {}
    for layer in range(N_LAYERS):
        std_stats = k_ge_s_margin(attempts[layer]["standard"])
        ours_stats = k_ge_s_margin(attempts[layer]["ours"])
        std_ok_rate = sum(per_fact_ok[layer]["standard"]) / len(per_fact_ok[layer]["standard"])
        ours_ok_rate = sum(per_fact_ok[layer]["ours"]) / len(per_fact_ok[layer]["ours"])
        winner = "ours" if ours_stats["margin"] > std_stats["margin"] else "standard"
        winners[layer] = winner
        print(f"{layer:5d} | {std_stats['k']:7.4f} {std_stats['s']:7.4f} {std_stats['margin']:10.4f} "
              f"{std_ok_rate:11.2f} | {ours_stats['k']:7.4f} {ours_stats['s']:7.4f} "
              f"{ours_stats['margin']:11.4f} {ours_ok_rate:12.2f} | {winner}")

    print("\nSelf-calibrated hybrid rule (from REAL accumulated history, not a hand-picked layer split):")
    print({layer: w for layer, w in winners.items()})


if __name__ == "__main__":
    main()
