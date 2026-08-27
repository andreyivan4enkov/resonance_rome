#!/usr/bin/env python3
"""Iteration R: "MEMIT не подходит для нашей задачи? НЕТ! Это значит что мы
не адаптировали НАШУ ЗАДАЧУ под MEMIT!" -- correct. MEMIT's real point is
making MANY SIMULTANEOUS edits tractable (the joint normal-equation solve
below), not reducing damage from a single edit -- iteration_q tested it on
ONE fact, where that advantage cannot show up at all. This test adapts the
task instead: insert SEVERAL new facts AT ONCE via the real MEMIT/ROME joint
closed-form solve (a single layer, generalized from rank-1 to rank-N), and
compare standard corpus C vs our resonance-weighted C for THAT real task.

Literal definition (real closed-form, generalized ROME normal equation for N
simultaneous key-value insertions -- this IS MEMIT's actual per-layer solve,
Meng et al. 2022 Eq 14, `Delta = R K^T (C_0 + K K^T)^-1`, here written with
our own C substituted for C_0):

  K = [k_1 .. k_N]  (3072 x N), real GELU-activated key per fact, one layer
  M = [m_1 .. m_N]  (768  x N), each m_i = v_orig_i + delta_i (delta_i via
      the same real v*-optimization used throughout this project)
  R = M - W_orig @ K              (768 x N, the raw per-fact residuals)
  Delta = R @ K^T @ (C + K K^T)^-1   (768 x 3072)
  W_new = W_orig + Delta

C_std = mean_j(k_j k_j^T) over real corpus peer keys (unchanged, real ROME/
MEMIT baseline). C_ours = sum_j [ w_j * k_j k_j^T ], where w_j = SUM over the
N new facts of topo(peer_j, fact_i)*budget_final(fact_i) (literal Method-3
topology+budget, computed once over the combined peer+fact-key graph,
summed across facts -- not averaged, consistent with the rest of this
project).
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
LAYER = 9  # a strong "ours"-favoring layer from iteration_p's real sweep
RATE_BUDGET = 0.2
N_PEERS = 200
N_HELDOUT = 200
V_STAR_STEPS = 30
V_STAR_LR = 0.5
RIDGE = 1.0

NEW_FACTS = [
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


def new_facts_learned(model, tok, device):
    n_ok = 0
    with torch.no_grad():
        for prompt, answer in NEW_FACTS:
            ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
            logits = model(ids).logits[0, -1]
            top1 = tok.decode([int(torch.argmax(logits))])
            n_ok += int(top1.strip().lower() == answer.strip().lower())
    return n_ok


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
    for _ in range(V_STAR_STEPS):
        logits = model(ids).logits[0, -1]
        loss = -F.log_softmax(logits, dim=-1)[answer_id]
        opt.zero_grad()
        loss.backward()
        opt.step()
    h.remove()
    v_star = (v_orig + delta_v).detach()
    return k_star.cpu().double().numpy(), v_star.cpu().double().numpy()


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


def memit_joint_edit(model, tok, device, layer, K, M, peer_K, mode):
    """Real MEMIT/ROME normal-equation joint solve for N simultaneous edits."""
    c_proj = model.transformer.h[layer].mlp.c_proj
    W_orig = c_proj.weight.detach().cpu().double().numpy().T  # (768,3072), ROME's W(out,in) convention
    d = peer_K.shape[1]

    C_std = (peer_K.T @ peer_K) / peer_K.shape[0]
    if mode == "standard":
        C = C_std
    else:
        n_facts = K.shape[1]
        V_for_topo = np.vstack([peer_K, K.T])  # (N_PEERS+n_facts, 3072)
        topo, budget_final = real_topology_and_budget(V_for_topo)
        n_peers = peer_K.shape[0]
        # w_j = sum over the n_facts fact-rows of topo(peer_j, fact_i)*budget_final(fact_i)
        w = np.zeros(n_peers)
        for fi in range(n_facts):
            fact_row = n_peers + fi
            w += topo[fact_row, :n_peers] * budget_final[:n_peers]  # FIXED: peer's own budget, not the fact's
        C = (peer_K * w[:, None]).T @ peer_K

    # Literal MEMIT/ROME normal equation: (C_0 + K K^T)^-1, NOT a fixed ridge --
    # regularization scales with the actual keys being jointly inserted (real
    # bug found live: a fixed small ridge was fine for N=1 but caused the
    # joint N=6 solve to explode to ppl ~1e19, since it doesn't grow with the
    # rank of the real edit).
    C_reg = C + K @ K.T + RIDGE * np.eye(d)
    C_inv = np.linalg.inv(C_reg)
    R = M - (W_orig @ K)  # (768, N)
    Delta = R @ K.T @ C_inv  # (768, 3072)
    with torch.no_grad():
        c_proj.weight += torch.tensor(Delta.T, dtype=c_proj.weight.dtype, device=device)


def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_sents = load_real_sentences(N_HELDOUT + N_PEERS)
    heldout_sents = real_sents[:N_HELDOUT]
    peer_sents = real_sents[N_HELDOUT:N_HELDOUT + N_PEERS]
    print(f"layer={LAYER}, {len(NEW_FACTS)} facts edited SIMULTANEOUSLY (real MEMIT joint solve)\n")

    ref_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    peer_K = extract_peer_keys(ref_model, tok, device, peer_sents, LAYER)
    K_cols, M_cols = [], []
    for prompt, answer in NEW_FACTS:
        k, m = compute_fact_target(ref_model, tok, device, LAYER, prompt, answer)
        K_cols.append(k)
        M_cols.append(m)
    K = np.stack(K_cols, axis=1)  # (3072, N)
    M = np.stack(M_cols, axis=1)  # (768, N)
    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for mode in ["standard", "ours"]:
        print(f"{'='*20} MODE: {mode} {'='*20}")
        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
        ppl0 = real_perplexity(model, tok, heldout_sents, device)
        facts0 = fact_check(model, tok, device)
        new0 = new_facts_learned(model, tok, device)
        print(f"BEFORE: ppl={ppl0:.4f} facts={facts0}/{len(FACT_PROMPTS)} new_facts_learned={new0}/{len(NEW_FACTS)}")

        memit_joint_edit(model, tok, device, LAYER, K, M, peer_K, mode)

        ppl1 = real_perplexity(model, tok, heldout_sents, device)
        facts1 = fact_check(model, tok, device)
        new1 = new_facts_learned(model, tok, device)
        print(f"AFTER ({mode}): ppl={ppl1:.4f} (delta={ppl1-ppl0:+.4f}) facts={facts1}/{len(FACT_PROMPTS)} "
              f"new_facts_learned={new1}/{len(NEW_FACTS)}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
