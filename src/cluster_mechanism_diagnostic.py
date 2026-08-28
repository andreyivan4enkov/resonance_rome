#!/usr/bin/env python3
"""Iteration BB: "подозреваю такие кластеры внутри слоя (у нас был уже
похожий эксперимент)" -- reuses the EXACT established method from
real_gpt2_clusterability.py (real spectral clustering + Shi-Malik Ncut on
real MLP c_fc neuron-correlation graph, already confirmed real clusters vs
shuffled null earlier this session) at LAYER=4 (this project's tuned layer),
then checks whether the two resonance mechanisms that gave an INVERTED
trade-off -- ridge-regularized resonance covariance (better NS) vs
resonance-weighted null-space projection (better PS) -- concentrate their
real Delta (weight update) energy in DIFFERENT real neuron clusters. If they
do, that is real structural evidence for the user's hypothesis that
induction/reduction/deduction correspond to real, distinct clusters WITHIN a
layer, not just a smooth profile across depth.
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
RIDGE = 1.0
N_PEERS = 2000
N_NEURONS = 400  # same tractability subsample as real_gpt2_clusterability.py
K_CLUSTERS = 6
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
    cases = []
    for d in sample:
        rw = d["requested_rewrite"]
        prompt = rw["prompt"].format(rw["subject"])
        cases.append(dict(prompt=prompt, target_new=" " + rw["target_new"]["str"]))
    return cases


def real_neuron_clusters(model, layer, n_neurons, k_clusters, seed):
    from sklearn.cluster import SpectralClustering
    rng = np.random.default_rng(seed)
    W_full = model.transformer.h[layer].mlp.c_fc.weight.detach().cpu().numpy()  # (768, 3072)
    idx = rng.choice(W_full.shape[1], size=n_neurons, replace=False)
    Wsub = W_full[:, idx]
    corr = np.corrcoef(Wsub.T)
    W_graph = np.abs(corr)
    np.fill_diagonal(W_graph, 0.0)
    sc = SpectralClustering(n_clusters=k_clusters, affinity="precomputed", random_state=0, assign_labels="kmeans")
    labels = sc.fit_predict(W_graph)
    return idx, labels


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
    else:  # null_resonance
        M = resonance_covariance(peer_K, k_star_np[None, :], RATE_BUDGET)
        P = null_space_projection(M, eigenvalue_threshold_frac=1e-2)
        C = 1e4 * (np.eye(d) - P)
        C_reg = C + RIDGE * np.eye(d)

    C_inv = np.linalg.inv(C_reg)
    Cinv_k = C_inv @ k_star_np
    denom = float(k_star_np @ Cinv_k)
    delta_out = v_star.cpu().double().numpy() - v_orig.cpu().double().numpy()
    Delta = np.outer(delta_out, Cinv_k) / denom  # (768, 3072)
    return Delta


def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ref_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    neuron_idx, cluster_labels = real_neuron_clusters(ref_model, LAYER, N_NEURONS, K_CLUSTERS, SEED)
    print(f"real spectral clustering: {N_NEURONS} neurons of layer {LAYER}, {K_CLUSTERS} clusters")
    print(f"cluster sizes: {[int((cluster_labels == c).sum()) for c in range(K_CLUSTERS)]}\n")

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
            col_norms = np.linalg.norm(Delta, axis=0)  # (3072,), real per-neuron update magnitude
            cluster_energy = np.array([
                col_norms[neuron_idx[cluster_labels == c]].sum() / max(1, (cluster_labels == c).sum())
                for c in range(K_CLUSTERS)
            ])
            cluster_energy_frac = cluster_energy / (cluster_energy.sum() + 1e-12)
            profiles[mode].append(cluster_energy_frac)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"{'mode':16s} | " + " ".join(f"clus{c}" for c in range(K_CLUSTERS)))
    mean_profiles = {}
    for mode, prs in profiles.items():
        mean_profile = np.mean(prs, axis=0)
        mean_profiles[mode] = mean_profile
        print(f"{mode:16s} | " + " ".join(f"{v:.3f}" for v in mean_profile))

    diff = mean_profiles["ridge_resonance"] - mean_profiles["null_resonance"]
    print(f"\nreal per-cluster difference (ridge - null): " + " ".join(f"{v:+.3f}" for v in diff))
    print(f"max |difference| across clusters: {np.abs(diff).max():.4f} "
          f"(cluster {int(np.argmax(np.abs(diff)))})")


if __name__ == "__main__":
    main()
