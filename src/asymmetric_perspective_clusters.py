#!/usr/bin/env python3
"""Iteration CC: "ты гадал классификацию не определив границы с перспективы
каждого, при этом ты смотришь на границы себя от себя и других" -- correct.
iteration_bb used sklearn's SpectralClustering: a symmetric affinity matrix,
ONE externally-imposed global cut, identical treatment for every node. That
is the standard method this whole project exists to NOT default to -- the
literal alternative already established here is asymmetric: each node
declares its OWN resonance neighborhood (topo_ij * budget_final_j, NOT
symmetric: topo_ij*budget_final_j != topo_ji*budget_final_i in general), and
"boundary" should emerge from each node's own perspective, matching the
Markov-Interface logic already used elsewhere this session (each side
declares its own local blanket; understanding is not assumed symmetric).

Real, literal redesign: reuse the ALREADY-established "gravitational center"
mechanic (kuramoto_compare.py's _gravity_norm: the node that has WON the
most budget through real resonance becomes a real causal attractor). For
each real neuron i, follow i's OWN chosen attractor (argmax_j topo_ij *
budget_final_j, from i's own row -- its own perspective, not a global
symmetric measure) until the chain reaches a real fixed point (a neuron that
is its own attractor). Real, self-organized, asymmetric clusters = the real
attractor basins that emerge, not a k-way partition imposed from outside.
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

from resonance_rome.core import null_space_projection, resonance_covariance, topology_and_budget  # noqa: E402

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
N_PEERS = 2000
N_NEURONS = 400
RATE_BUDGET = 0.2
V_STAR_STEPS = 30
V_STAR_LR = 0.5
SEED = 20260827
N_CASES = 5


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
    return [dict(prompt=d["requested_rewrite"]["prompt"].format(d["requested_rewrite"]["subject"]),
                 target_new=" " + d["requested_rewrite"]["target_new"]["str"]) for d in sample]


def real_asymmetric_attractor_clusters(model, layer, n_neurons, seed):
    """Real, asymmetric, self-organized clustering: each neuron's OWN real
    resonance perspective (topo_ij * budget_final_j, its own row) picks its
    own attractor; clusters = the real attractor basins that emerge."""
    rng = np.random.default_rng(seed)
    W_full = model.transformer.h[layer].mlp.c_fc.weight.detach().cpu().double().numpy()  # (768, 3072)
    idx = rng.choice(W_full.shape[1], size=n_neurons, replace=False)
    V = W_full[:, idx].T  # (n_neurons, 768), each row = one real neuron's own weight vector

    topo, budget_final = topology_and_budget(V, RATE_BUDGET)  # literal, established, asymmetric formula
    # Each neuron i's OWN perspective: who does IT resonate with most (its own row).
    pull = topo * budget_final.reshape(1, -1)  # pull[i, j] = topo_ij * budget_final_j, from i's row
    np.fill_diagonal(pull, -np.inf)
    own_attractor = np.argmax(pull, axis=1)  # (n_neurons,), each neuron's own real chosen attractor

    # Follow each neuron's real attractor chain to its real fixed point.
    def find_root(i, seen=None):
        seen = seen or set()
        if i in seen:
            return i  # real cycle -- its own local basin, treat the cycle entry as the root
        seen.add(i)
        nxt = own_attractor[i]
        if nxt == i:
            return i
        return find_root(nxt, seen)

    roots = np.array([find_root(i) for i in range(n_neurons)])
    unique_roots = np.unique(roots)
    root_to_cluster = {r: c for c, r in enumerate(unique_roots)}
    cluster_labels = np.array([root_to_cluster[r] for r in roots])
    return idx, cluster_labels, len(unique_roots)


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


def extract_peer_keys(model, tok, device, sentences, layer):
    keys = []
    for s in sentences:
        k, _ = get_real_key(model, tok, device, s, layer)
        keys.append(k.cpu().double().numpy())
    return np.stack(keys)


def compute_delta(model, tok, device, prompt, target_word, peer_K, cov_mode):
    import torch.nn.functional as F
    c_proj = model.transformer.h[LAYER].mlp.c_proj
    k_star, ids = get_real_key(model, tok, device, prompt, LAYER)
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

    k_star_np = k_star.cpu().double().numpy()
    d = peer_K.shape[1]
    if cov_mode == "ridge_resonance":
        C = resonance_covariance(peer_K, k_star_np[None, :], RATE_BUDGET)
        C_reg = C + RIDGE * np.eye(d)
    else:
        M = resonance_covariance(peer_K, k_star_np[None, :], RATE_BUDGET)
        P = null_space_projection(M, eigenvalue_threshold_frac=1e-2)
        C = 1e4 * (np.eye(d) - P)
        C_reg = C + RIDGE * np.eye(d)

    C_inv = np.linalg.inv(C_reg)
    Cinv_k = C_inv @ k_star_np
    denom = float(k_star_np @ Cinv_k)
    delta_out = v_star.cpu().double().numpy() - v_orig.cpu().double().numpy()
    Delta = np.outer(delta_out, Cinv_k) / denom
    return Delta


def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ref_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    neuron_idx, cluster_labels, n_clusters = real_asymmetric_attractor_clusters(ref_model, LAYER, N_NEURONS, SEED)
    sizes = [int((cluster_labels == c).sum()) for c in range(n_clusters)]
    print(f"real ASYMMETRIC attractor clustering: {N_NEURONS} neurons of layer {LAYER}")
    print(f"real, self-organized cluster count: {n_clusters}, sizes (top 15): {sorted(sizes, reverse=True)[:15]}\n")

    peer_sents = load_real_sentences(N_PEERS)
    peer_K = extract_peer_keys(ref_model, tok, device, peer_sents, LAYER)
    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cases = load_counterfact_cases(N_CASES, SEED)
    profiles = {"ridge_resonance": [], "null_resonance": []}
    for case in cases:
        for mode in ["ridge_resonance", "null_resonance"]:
            model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
            Delta = compute_delta(model, tok, device, case["prompt"], case["target_new"], peer_K, mode)
            col_norms = np.linalg.norm(Delta, axis=0)
            cluster_energy = np.array([
                col_norms[neuron_idx[cluster_labels == c]].sum() / max(1, (cluster_labels == c).sum())
                for c in range(n_clusters)
            ])
            cluster_energy_frac = cluster_energy / (cluster_energy.sum() + 1e-12)
            profiles[mode].append(cluster_energy_frac)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    mean_profiles = {m: np.mean(prs, axis=0) for m, prs in profiles.items()}
    diff = mean_profiles["ridge_resonance"] - mean_profiles["null_resonance"]
    # Only report clusters with real, non-trivial size (>=5 neurons) -- tiny
    # 1-2-neuron basins are real but too small to interpret reliably.
    big_clusters = [c for c in range(n_clusters) if sizes[c] >= 5]
    print(f"real, non-trivial clusters (size>=5): {len(big_clusters)} of {n_clusters}\n")
    print(f"{'cluster':>8} {'size':>5} {'ridge':>7} {'null':>7} {'diff':>8}")
    for c in sorted(big_clusters, key=lambda c: -sizes[c])[:20]:
        print(f"{c:8d} {sizes[c]:5d} {mean_profiles['ridge_resonance'][c]:7.4f} "
              f"{mean_profiles['null_resonance'][c]:7.4f} {diff[c]:+8.4f}")

    max_c = big_clusters[int(np.argmax(np.abs(diff[big_clusters])))]
    print(f"\nmax |difference| among non-trivial clusters: {abs(diff[max_c]):.4f} (cluster {max_c}, size {sizes[max_c]})")


if __name__ == "__main__":
    main()
