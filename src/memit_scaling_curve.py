#!/usr/bin/env python3
"""Iteration S: does the resonance-covariance advantage in MEMIT's real joint
multi-fact solve hold, grow, or shrink as the number of SIMULTANEOUS edits
grows (6 -> 15 -> 30 -> 50)? All answer words verified single-BPE-token in
advance (the "thunderclap" confound from earlier is avoided by construction).

v*/k* are computed ONCE for all 50 facts from the clean reference model, then
reused as prefixes for the smaller N subsets -- avoids recomputing the
expensive optimization 4x over shared facts.
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
N_VALUES = [6, 15, 30, 50]

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


def new_facts_learned(model, tok, device, facts):
    n_ok = 0
    with torch.no_grad():
        for prompt, answer in facts:
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
    c_proj = model.transformer.h[layer].mlp.c_proj
    W_orig = c_proj.weight.detach().cpu().double().numpy().T
    d = peer_K.shape[1]

    C_std = (peer_K.T @ peer_K) / peer_K.shape[0]
    if mode == "standard":
        C = C_std
    else:
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


def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_sents = load_real_sentences(N_HELDOUT + N_PEERS)
    heldout_sents = real_sents[:N_HELDOUT]
    peer_sents = real_sents[N_HELDOUT:N_HELDOUT + N_PEERS]
    print(f"layer={LAYER}, {len(NEW_FACTS)} facts total, testing N={N_VALUES}\n")

    ref_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    peer_K = extract_peer_keys(ref_model, tok, device, peer_sents, LAYER)
    all_k, all_m = [], []
    for prompt, answer in NEW_FACTS:
        k, m = compute_fact_target(ref_model, tok, device, LAYER, prompt, answer)
        all_k.append(k)
        all_m.append(m)
    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("real v*/k* computed for all 50 facts from the clean reference model\n")

    results = {}
    for N in N_VALUES:
        facts_N = NEW_FACTS[:N]
        K = np.stack(all_k[:N], axis=1)
        M = np.stack(all_m[:N], axis=1)
        row = {}
        for mode in ["standard", "ours"]:
            model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
            ppl0 = real_perplexity(model, tok, heldout_sents, device)
            facts0 = fact_check(model, tok, device)

            memit_joint_edit(model, tok, device, LAYER, K, M, peer_K, mode)

            ppl1 = real_perplexity(model, tok, heldout_sents, device)
            facts1 = fact_check(model, tok, device)
            new1 = new_facts_learned(model, tok, device, facts_N)
            row[mode] = dict(ppl_delta=ppl1 - ppl0, facts=facts1, new_learned=new1)
            print(f"N={N:2d} [{mode:8s}]: ppl {ppl0:.2f}->{ppl1:.2f} (delta={ppl1-ppl0:+.4f}) "
                  f"facts={facts1}/{len(FACT_PROMPTS)} new_learned={new1}/{N}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results[N] = row
        print()

    print("=== SCALING CURVE: standard vs ours, N simultaneous edits ===")
    print(f"{'N':>4} | {'std_ppl_d':>10} {'std_facts':>9} {'std_learn':>9} | "
          f"{'ours_ppl_d':>10} {'ours_facts':>10} {'ours_learn':>10}")
    for N, row in results.items():
        s, o = row["standard"], row["ours"]
        print(f"{N:4d} | {s['ppl_delta']:10.4f} {s['facts']:9d} {s['new_learned']:9d} | "
              f"{o['ppl_delta']:10.4f} {o['facts']:10d} {o['new_learned']:10d}")


if __name__ == "__main__":
    main()
