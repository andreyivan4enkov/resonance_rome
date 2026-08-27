#!/usr/bin/env python3
"""Iteration T: why does the real MEMIT joint solve plateau at ~9-10 learned
facts regardless of N (iteration_s: 10/15, 10/30, 9/50)? Diagnose, don't
guess: for each of the 50 facts, after the REAL joint edit (mode="ours",
N=50), measure (1) whether it was actually learned (top-1), (2) how close
the model's REAL post-edit output at that fact's key is to its OWN intended
target v* (residual norm -- does the joint solve even try to fit it well?),
(3) how big a push v*-optimization needed in isolation (|delta_v|), and (4)
how similar this fact's key is to the OTHER 49 facts' keys (mean cosine --
the "key collision" hypothesis: keys that look like other keys should be
harder to edit independently via one shared linear solve).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_DEFAULT_HF_CACHE = str(Path(__file__).resolve().parents[1] / ".hf_cache")
os.environ.setdefault("HF_HOME", _DEFAULT_HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", _DEFAULT_HF_CACHE)
os.environ.setdefault("HF_HUB_CACHE", _DEFAULT_HF_CACHE)

HOTPOT = Path(os.environ.get(
    "HOTPOT_CORPUS_PATH",
    str(Path(__file__).resolve().parents[1] / "benchmarks" / "hotpot_dev_distractor_v1.json"),
))
LAYER = 9
RATE_BUDGET = 0.2
N_PEERS = 200
N_HELDOUT = 200
V_STAR_STEPS = 30
V_STAR_LR = 0.5
RIDGE = 1.0
N = 50

TEMPLATES = [
    "The secret code word is",
    "The hidden treasure is a",
    "Her favorite object is the",
    "The old man's cane was made of",
    "The spy's tool was a",
    "In the box there was a",
    "The wizard's amulet was shaped like a",
    "The pirate buried his",
    "At the museum, the rarest item was a",
    "The child's favorite toy was a",
]
ANSWER_WORDS = ['apple', 'guitar', 'rocket', 'diamond', 'compass', 'whistle', 'anchor', 'pumpkin',
    'umbrella', 'dolphin', 'volcano', 'marble', 'trumpet', 'lighthouse', 'banana', 'mirror',
    'lantern', 'hammer', 'ladder', 'castle', 'dragon', 'violin', 'telescope', 'microscope',
    'spider', 'butterfly', 'elephant', 'tiger', 'lion', 'wolf', 'eagle', 'owl', 'raven', 'hawk',
    'turtle', 'lizard', 'beetle', 'maple', 'bamboo', 'granite', 'crystal', 'pearl', 'ruby',
    'velvet', 'leather', 'helmet', 'shield', 'sword', 'arrow', 'banner']
NEW_FACTS = [(TEMPLATES[i % len(TEMPLATES)], " " + w) for i, w in enumerate(ANSWER_WORDS)]


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


def compute_fact_target(model, tok, device, layer, prompt, answer):
    c_proj = model.transformer.h[layer].mlp.c_proj
    k_star, ids = get_real_key(model, tok, device, prompt, layer)
    v_orig = (k_star @ c_proj.weight.detach())
    delta_v = torch.zeros_like(v_orig, requires_grad=True)

    def add_hook(module, inp, out):
        return out + delta_v

    h = c_proj.register_forward_hook(add_hook)
    opt = torch.optim.Adam([delta_v], lr=V_STAR_LR)
    answer_id = tok(answer, add_special_tokens=False)["input_ids"][0]
    final_loss = None
    for _ in range(V_STAR_STEPS):
        logits = model(ids).logits[0, -1]
        final_loss = -F.log_softmax(logits, dim=-1)[answer_id]
        opt.zero_grad()
        final_loss.backward()
        opt.step()
    h.remove()
    v_star = (v_orig + delta_v).detach()
    return (k_star.cpu().double().numpy(), v_star.cpu().double().numpy(),
            float(delta_v.norm()), float(final_loss.item()))


def extract_peer_keys(model, tok, device, sentences, layer):
    keys = []
    for s in sentences:
        k, _ = get_real_key(model, tok, device, s, layer)
        keys.append(k.cpu().double().numpy())
    return np.stack(keys)


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


def memit_joint_edit(model, tok, device, layer, K, M, peer_K):
    c_proj = model.transformer.h[layer].mlp.c_proj
    W_orig = c_proj.weight.detach().cpu().double().numpy().T
    d = peer_K.shape[1]
    n_facts = K.shape[1]
    V_for_topo = np.vstack([peer_K, K.T])
    topo, budget_final = real_topology_and_budget(V_for_topo)
    n_peers = peer_K.shape[0]
    w = np.zeros(n_peers)
    for fi in range(n_facts):
        fact_row = n_peers + fi
        w += topo[fact_row, :n_peers] * budget_final[:n_peers]  # FIXED: peer's own budget, not the fact's
    C = (peer_K * w[:, None]).T @ peer_K
    C_reg = C + K @ K.T + RIDGE * np.eye(d)
    C_inv = np.linalg.inv(C_reg)
    R = M - (W_orig @ K)
    Delta = R @ K.T @ C_inv
    with torch.no_grad():
        c_proj.weight += torch.tensor(Delta.T, dtype=c_proj.weight.dtype, device=device)
    return W_orig, Delta


def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_sents = load_real_sentences(N_HELDOUT + N_PEERS)
    peer_sents = real_sents[N_HELDOUT:N_HELDOUT + N_PEERS]
    print(f"layer={LAYER}, N={N} facts\n")

    ref_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    peer_K = extract_peer_keys(ref_model, tok, device, peer_sents, LAYER)
    all_k, all_m, all_delta_norm, all_vstar_loss = [], [], [], []
    for prompt, answer in NEW_FACTS:
        k, m, dn, loss = compute_fact_target(ref_model, tok, device, LAYER, prompt, answer)
        all_k.append(k)
        all_m.append(m)
        all_delta_norm.append(dn)
        all_vstar_loss.append(loss)
    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    K = np.stack(all_k, axis=1)  # (3072, N)
    M = np.stack(all_m, axis=1)  # (768, N)

    # key-collision proxy: mean real cosine of each fact's key to the OTHER N-1 keys
    Kn = K.T / (np.linalg.norm(K.T, axis=-1, keepdims=True) + 1e-12)
    key_cos = Kn @ Kn.T
    np.fill_diagonal(key_cos, np.nan)
    mean_key_sim = np.nanmean(key_cos, axis=1)

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    W_orig, Delta = memit_joint_edit(model, tok, device, LAYER, K, M, peer_K)

    c_proj = model.transformer.h[LAYER].mlp.c_proj
    print(f"{'fact':>12s} {'learned':>7s} {'residual':>9s} {'|delta_v|':>10s} {'vstar_loss':>10s} {'mean_key_sim':>12s}")
    rows = []
    for i, (prompt, answer) in enumerate(NEW_FACTS):
        with torch.no_grad():
            ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
            logits = model(ids).logits[0, -1]
            top1 = tok.decode([int(torch.argmax(logits))])
        learned = top1.strip().lower() == answer.strip().lower()

        k_i_t = torch.tensor(all_k[i], dtype=c_proj.weight.dtype, device=device)
        v_actual = (k_i_t @ c_proj.weight.detach()).cpu().double().numpy()
        residual = float(np.linalg.norm(v_actual - all_m[i]))

        rows.append((answer.strip(), learned, residual, all_delta_norm[i], all_vstar_loss[i], mean_key_sim[i]))
        print(f"{answer.strip():>12s} {str(learned):>7s} {residual:9.3f} {all_delta_norm[i]:10.3f} "
              f"{all_vstar_loss[i]:10.5f} {mean_key_sim[i]:12.4f}")

    learned_rows = [r for r in rows if r[1]]
    failed_rows = [r for r in rows if not r[1]]
    print(f"\n=== learned={len(learned_rows)}/{N} ===")

    def stats(rs, idx):
        vals = [r[idx] for r in rs]
        return (np.mean(vals), np.std(vals)) if vals else (float("nan"), float("nan"))

    for name, idx in [("residual", 2), ("|delta_v|", 3), ("vstar_loss", 4), ("mean_key_sim", 5)]:
        lm, ls = stats(learned_rows, idx)
        fm, fs = stats(failed_rows, idx)
        print(f"{name:12s}: learned mean={lm:.4f} std={ls:.4f}  |  failed mean={fm:.4f} std={fs:.4f}")


if __name__ == "__main__":
    main()
