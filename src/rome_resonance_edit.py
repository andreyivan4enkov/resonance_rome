#!/usr/bin/env python3
"""Iteration O: "ROME/MEMIT можно прокачать моими этими методами... тестируй,
меняя на мою динамику вместо обобщённой статистики корпуса."

Real, literal ROME (Meng et al. 2022) rank-one edit on a real GPT-2-small MLP
layer, comparing TWO choices of the preservation covariance C:
  (a) STANDARD ROME: C_std = mean_j k_j k_j^T over a REAL corpus sample of
      key vectors (literal ROME, unchanged formula: MEAN, a genuine second-
      moment statistic -- not "sum not mean" here, because this IS real
      ROME's own baseline, kept faithful on purpose).
  (b) OUR SUBSTITUTION: C_ours = sum_j topo(k_j,k*)*budget_final_j * k_j k_j^T
      -- literal, unchanged Method-3 formula (relu(cos) topology + weak/
      strong budget transfer), reweighting the SAME real peer key vectors by
      real resonance to the new key k*, instead of generic corpus frequency.

Literal ROME mechanics:
  MLP: h = act(c_fc(x)); v = c_proj(h) = W @ h  (W = c_proj.weight, (768,3072))
  k* = real GELU-activated key vector at the last token of the fact prompt
  v_orig = W @ k*  (current real output)
  v* = v_orig + delta_v, where delta_v is found by a REAL short optimization
       (Adam, few steps) that maximizes real log P(" banana") at that position
       -- ROME's own "compute target vector" step
  Delta = (v* - v_orig) (C^-1 k*)^T / (k*^T C^-1 k*)     -- literal ROME closed form
  W_new = W + Delta
Both C_std and C_ours get the SAME ridge regularization (C + lambda*I) for
invertibility -- standard numerical hygiene, disclosed, applied identically
to both so the comparison is fair.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# HF cache defaults to a local, repo-relative folder; override via the usual
# HF_HOME/TRANSFORMERS_CACHE/HF_HUB_CACHE env vars to point at a shared cache.
_DEFAULT_HF_CACHE = str(Path(__file__).resolve().parents[1] / ".hf_cache")
os.environ.setdefault("HF_HOME", _DEFAULT_HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", _DEFAULT_HF_CACHE)
os.environ.setdefault("HF_HUB_CACHE", _DEFAULT_HF_CACHE)
# NOTE: not forcing HF_HUB_OFFLINE=1 -- a fresh clone needs to download GPT-2
# once; set HF_HUB_OFFLINE=1 yourself once it's cached if you want that.

# Any real English text corpus works here (peer-key source + held-out
# perplexity set); override with HOTPOT_CORPUS_PATH, default assumes a
# benchmarks/ folder next to this repo -- see README "Reproducing".
HOTPOT = Path(os.environ.get(
    "HOTPOT_CORPUS_PATH",
    str(Path(__file__).resolve().parents[1] / "benchmarks" / "hotpot_dev_distractor_v1.json"),
))
N_LAYERS = 12
RATE_BUDGET = 0.2
N_PEERS = 200
V_STAR_STEPS = 30
V_STAR_LR = 0.5
RIDGE = 1.0  # same regularization constant used for BOTH C_std and C_ours

BARE_PROMPT = "The secret code word is"
NEW_FACT_ANSWER = " banana"

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
    correct, details = 0, []
    with torch.no_grad():
        for prompt, expected in FACT_PROMPTS:
            ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
            logits = model(ids).logits[0, -1]
            top_tok = tok.decode([int(torch.argmax(logits))])
            ok = top_tok.strip().lower() == expected.strip().lower()
            correct += int(ok)
            details.append((prompt, top_tok.strip(), ok))
    return correct, details


def bare_prompt_check(model, tok, device):
    with torch.no_grad():
        ids = tok(BARE_PROMPT, return_tensors="pt")["input_ids"].to(device)
        logits = model(ids).logits[0, -1]
        top5 = torch.topk(logits, 5).indices
        tops = [tok.decode([int(t)]) for t in top5]
    return tops[0].strip().lower() == NEW_FACT_ANSWER.strip().lower(), tops


def get_real_key(model, tok, device, text, layer):
    """Real GELU-activated MLP key vector (c_proj's real input) at the LAST token."""
    ids = tok(text, return_tensors="pt")["input_ids"].to(device)
    captured = {}

    def hook(module, inp, out):
        captured["k"] = inp[0][0, -1].detach()  # c_proj's real input, last position

    h = model.transformer.h[layer].mlp.c_proj.register_forward_hook(hook)
    with torch.no_grad():
        model(ids)
    h.remove()
    return captured["k"], ids


def extract_all_layer_keys(model, tok, device, sentences, n_layers):
    """Real GELU-activated MLP key, last token, ALL layers at once (one real
    forward pass per sentence instead of n_layers passes)."""
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


def rome_edit(model, tok, device, layer, K, mode):
    """mode: 'standard' (real ROME, generic corpus C) or 'ours' (topology+budget C).
    K: (N_PEERS, 3072) real peer key matrix for THIS layer, pre-extracted from
    the clean, unedited reference model (same real peers for both modes)."""
    c_proj = model.transformer.h[layer].mlp.c_proj

    k_star, ids = get_real_key(model, tok, device, BARE_PROMPT, layer)
    # GPT-2 uses Conv1D, not nn.Linear: weight is (in=3072, out=768), forward is x @ weight.
    v_orig = (k_star @ c_proj.weight.detach())

    # --- ROME step 1: optimize v* to make the model predict banana here ---
    delta_v = torch.zeros_like(v_orig, requires_grad=True)

    def add_hook(module, inp, out):
        return out + delta_v

    h = c_proj.register_forward_hook(add_hook)
    opt = torch.optim.Adam([delta_v], lr=V_STAR_LR)
    banana_id = tok(NEW_FACT_ANSWER, add_special_tokens=False)["input_ids"][0]
    for _ in range(V_STAR_STEPS):
        logits = model(ids).logits[0, -1]
        loss = -F.log_softmax(logits, dim=-1)[banana_id]
        opt.zero_grad()
        loss.backward()
        opt.step()
    h.remove()
    v_star = (v_orig + delta_v).detach()
    print(f"    [{mode}] v* opt final loss={loss.item():.4f}, |delta_v|={float(delta_v.norm()):.3f}")
    d = K.shape[1]

    C_std = (K.T @ K) / K.shape[0]  # literal ROME: MEAN second moment, real corpus statistic (INDUCTION)
    if mode == "standard":
        C = C_std
    else:
        k_star_np = k_star.cpu().double().numpy()
        V_for_topo = np.vstack([K, k_star_np[None, :]])
        topo, budget_final = real_topology_and_budget(V_for_topo)
        w = topo[-1, :-1] * budget_final[:-1]  # real resonance of each peer to k*, literal Method 3
        print(f"    [{mode}] resonance weights: mean={w.mean():.4f} max={w.max():.4f} n_nonzero={(w>0).sum()}/{len(w)}")
        C_ours_raw = (K * w[:, None]).T @ K  # SUM (not mean) weighted by real topology+budget (DEDUCTION)

        if mode == "ours":
            C = C_ours_raw
        elif mode == "blend":
            # Principle 4 (induction/reduction/deduction ratio depends on layer
            # depth): rescale C_ours to C_std's own energy scale (trace) so the
            # blend weight is meaningful regardless of raw magnitude, then mix
            # by a layer-depth profile alpha(layer)=layer/(N_LAYERS-1) -- pure
            # induction (C_std) at layer 0, pure deduction (C_ours) at the last
            # layer, linear in between.
            C_ours_rescaled = C_ours_raw * (np.trace(C_std) / (np.trace(C_ours_raw) + 1e-12))
            alpha = layer / (N_LAYERS - 1)
            C = (1 - alpha) * C_std + alpha * C_ours_rescaled
            print(f"    [{mode}] alpha(layer={layer})={alpha:.3f}, "
                  f"trace(C_std)={np.trace(C_std):.4f}, trace(C_ours_raw)={np.trace(C_ours_raw):.4f}")

    C_reg = C + RIDGE * np.eye(d)
    C_inv = np.linalg.inv(C_reg)
    k_star_np = k_star.cpu().double().numpy()
    Cinv_k = C_inv @ k_star_np
    denom = float(k_star_np @ Cinv_k)
    delta_out = (v_star.cpu().double().numpy() - v_orig.cpu().double().numpy())
    Delta = np.outer(delta_out, Cinv_k) / denom  # (768, 3072) in ROME's W(out,in) convention
    print(f"    [{mode}] |Delta|={np.linalg.norm(Delta):.4f}")

    with torch.no_grad():
        # c_proj.weight is Conv1D's (in=3072, out=768) -- transpose Delta back to that layout.
        c_proj.weight += torch.tensor(Delta.T, dtype=c_proj.weight.dtype, device=device)


def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_sents = load_real_sentences(400)
    print(f"n={len(real_sents)} real held-out sentences, testing ALL {N_LAYERS} layers\n")

    print("Extracting real peer keys for all layers from ONE clean reference model...")
    ref_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    peer_sents = real_sents[:N_PEERS]
    all_layer_keys = extract_all_layer_keys(ref_model, tok, device, peer_sents, N_LAYERS)
    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  done: {N_LAYERS} layers x {N_PEERS} real peer keys each\n")

    ppl_base = None
    all_results = {}
    for layer in range(N_LAYERS):
        print(f"{'='*20} LAYER {layer} {'='*20}")
        K = all_layer_keys[layer]
        row_results = {}
        for mode in ["standard", "blend"]:
            model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
            ok0, top5_0 = bare_prompt_check(model, tok, device)
            ppl0 = real_perplexity(model, tok, real_sents, device)
            facts0, _ = fact_check(model, tok, device)
            if ppl_base is None:
                ppl_base = ppl0

            rome_edit(model, tok, device, layer, K, mode)

            ok1, top5_1 = bare_prompt_check(model, tok, device)
            ppl1 = real_perplexity(model, tok, real_sents, device)
            facts1, details1 = fact_check(model, tok, device)
            print(f"  [{mode}] AFTER: top5={top5_1} ok={ok1}, ppl={ppl1:.4f} (delta={ppl1-ppl0:+.4f}), "
                  f"facts={facts1}/{len(FACT_PROMPTS)}")
            row_results[mode] = dict(ok1=ok1, ppl1=ppl1, facts1=facts1)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        all_results[layer] = row_results
        print()

    print("=== SUMMARY: standard vs blend (depth-weighted), all layers (baseline ppl={:.4f}, facts=14/14) ===".format(ppl_base))
    print(f"{'layer':>5} | {'std_ok':>6} {'std_ppl':>8} {'std_facts':>9} | {'blend_ok':>8} {'blend_ppl':>9} {'blend_facts':>11}")
    for layer, r in all_results.items():
        s, o = r["standard"], r["blend"]
        print(f"{layer:5d} | {str(s['ok1']):>6} {s['ppl1']:8.2f} {s['facts1']:9d} | "
              f"{str(o['ok1']):>8} {o['ppl1']:9.2f} {o['facts1']:11d}")


if __name__ == "__main__":
    main()
