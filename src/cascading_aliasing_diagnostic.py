#!/usr/bin/env python3
"""Iteration AA: "алиасинг распределяется не только на эту метрику... каждый
слой выше содержит алиасинг слоя ниже, но пропорционально, и там 'частота
ниже'." Test the CASCADING version of the hierarchy hypothesis directly on
real activations, not diluted through a downstream edit-quality metric:

For each of the 11 real adjacent MLP-key layer pairs (0->1, 1->2, ..., 10->11),
and for several candidate decimation strides k, measure real cosine
similarity between a decimated(k) version of layer L-1's real key and layer
L's real key, averaged over many real sentences. If aliasing cascades and
gets coarser with depth, the BEST-correlating stride should tend to GROW as
L increases (each higher layer "sees" an increasingly decimated, lower-
frequency signal from the one below it).

MLP keys (not residual hidden states) are used specifically because they are
each layer's OWN independently-learned nonlinear transform -- unlike the
residual stream, a layer's real key does NOT trivially/additively contain
the layer below's key by construction, so a real correlation here is not a
confound of GPT-2's residual architecture.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

_DEFAULT_HF_CACHE = str(Path(__file__).resolve().parents[1] / ".hf_cache")
os.environ.setdefault("HF_HOME", _DEFAULT_HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", _DEFAULT_HF_CACHE)
os.environ.setdefault("HF_HUB_CACHE", _DEFAULT_HF_CACHE)

HOTPOT = Path(os.environ.get(
    "HOTPOT_CORPUS_PATH",
    str(Path(__file__).resolve().parents[1] / "benchmarks" / "hotpot_dev_distractor_v1.json"),
))
N_LAYERS = 12
N_SENTENCES = 300
STRIDES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]


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


def real_cosine_rowwise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    an = A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-12)
    bn = B / (np.linalg.norm(B, axis=-1, keepdims=True) + 1e-12)
    return (an * bn).sum(axis=-1)


def real_cosine_on_retained_coords(K_lo: np.ndarray, K_hi: np.ndarray, stride: int) -> float:
    """FIXED (caught live): comparing a zero-padded decimated vector to a
    FULL vector always scores lower cosine as stride grows, trivially --
    zeroing most coordinates destroys norm/content regardless of whether
    real aliasing exists. Correct comparison: restrict BOTH sides to the
    SAME retained (strided) coordinates, so stride only removes information
    symmetrically -- a real test of whether the RETAINED coarse coordinates
    still line up, not an artifact of introducing zeros on one side only."""
    lo_ret = K_lo[:, ::stride]
    hi_ret = K_hi[:, ::stride]
    return float(real_cosine_rowwise(lo_ret, hi_ret).mean())


def real_cosine_shuffled_control(K_lo: np.ndarray, K_hi: np.ndarray, stride: int, seed: int) -> float:
    """Real random-pairing control: same retained coordinates, but K_hi rows
    shuffled to unrelated real sentences -- the chance baseline this
    diagnostic must clear to mean anything."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(K_hi.shape[0])
    return real_cosine_on_retained_coords(K_lo, K_hi[perm], stride)


def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sentences = load_real_sentences(N_SENTENCES)
    print(f"real sentences: {len(sentences)}, layers=0..{N_LAYERS-1}, strides={STRIDES}\n")

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    all_keys = extract_all_layer_keys(model, tok, device, sentences, N_LAYERS)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"{'pair':>8} | " + " ".join(f"k={k:3d}" for k in STRIDES) + " | margin-over-chance @ k=64")
    margins_at_coarsest = []
    for L in range(1, N_LAYERS):
        K_lo = all_keys[L - 1]
        K_hi = all_keys[L]
        sims = [real_cosine_on_retained_coords(K_lo, K_hi, k) for k in STRIDES]
        chance = [real_cosine_shuffled_control(K_lo, K_hi, k, seed=L) for k in STRIDES]
        margin_coarsest = sims[-1] - chance[-1]
        margins_at_coarsest.append(margin_coarsest)
        row = " ".join(f"{s:5.3f}" for s in sims)
        chance_row = " ".join(f"{c:5.3f}" for c in chance)
        print(f"{L-1:3d}->{L:<3d} real | {row}")
        print(f"{'':8s} rand | {chance_row} | margin@k=64: {margin_coarsest:+.4f}")

    print(f"\nreal margin-over-chance at the COARSEST stride (k=64) across depth: {[f'{m:+.4f}' for m in margins_at_coarsest]}")
    xs = np.arange(len(margins_at_coarsest))
    ys = np.array(margins_at_coarsest)
    r = np.corrcoef(xs, ys)[0, 1]
    print(f"real Pearson correlation of (layer depth) vs (real margin-over-chance at coarsest stride): r={r:.3f}")
    print("(a real, growing margin at higher layers would support 'higher layers retain more real coarse\n"
          " structure from the layer below than chance, and this grows with depth' -- the cascading prediction)")


if __name__ == "__main__":
    main()
