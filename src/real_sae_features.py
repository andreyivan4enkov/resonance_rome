#!/usr/bin/env python3
"""Build a REAL, working sparse autoencoder (Anthropic-style: Towards
Monosemanticity) on REAL GPT-2 layer-6 residual-stream activations, from
REAL HotpotQA text. This gives REAL, monosemantic linear-direction features
(SAE dictionary columns) -- what "a feature" actually means, per the
Anthropic finding confirmed earlier this session and per the user's own,
consistently-held definition. Everything downstream (topology/budget
restructuring, one-shot new-feature addition) will operate on THESE
directions, not on raw embedding rows or raw neurons.

SAE architecture (real, standard, not our own theory -- this is a tool being
BUILT to find real features, distinct from the mechanic being tested on them):
  features = ReLU(W_enc @ (x - b_dec) + b_enc)
  x_hat    = W_dec @ features + b_dec
  loss     = ||x - x_hat||^2 + lambda * ||features||_1
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_HOME", r"D:\RLM\_external\sandbox\hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", r"D:\RLM\_external\sandbox\hf_cache")
os.environ.setdefault("HF_HUB_CACHE", r"D:\RLM\_external\sandbox\hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

HOTPOT = Path(r"D:\RLM\benchmarks\hotpot_dev_distractor_v1.json")
LAYER = 6
D_MODEL = 768
D_DICT = 1536
L1_LAMBDA = 3e-3
N_STEPS = 800
LR = 1e-3
BATCH = 256
CKPT = Path(r"D:\RLM\_external\sandbox\results\real_sae_layer6.pt")


class SAE(nn.Module):
    def __init__(self, d_model, d_dict):
        super().__init__()
        self.W_enc = nn.Parameter(torch.randn(d_dict, d_model) * 0.05)
        self.b_enc = nn.Parameter(torch.zeros(d_dict))
        self.W_dec = nn.Parameter(torch.randn(d_model, d_dict) * 0.05)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)

    def decode(self, f):
        return f @ self.W_dec.T + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


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


def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    sentences = load_real_sentences(300)
    print(f"n={len(sentences)} real sentences for SAE training data")

    # Collect REAL layer-6 residual-stream activations across real text
    acts = []
    with torch.no_grad():
        for s in sentences:
            ids = tok(s, return_tensors="pt", truncation=True, max_length=48)["input_ids"].to(device)
            if ids.shape[1] < 4:
                continue
            hs = model.transformer(ids, output_hidden_states=True).hidden_states[LAYER][0]  # (T,768) REAL
            acts.append(hs.cpu())
    X = torch.cat(acts, dim=0).float()  # (N, 768) real activation dataset
    print(f"real activation dataset: {X.shape}")
    mean_norm = X.norm(dim=-1).mean().item()
    print(f"mean real activation norm: {mean_norm:.4f}")

    sae = SAE(D_MODEL, D_DICT).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    N = X.shape[0]

    for step in range(N_STEPS):
        idx = torch.randint(0, N, (BATCH,))
        x = X[idx].to(device)
        x_hat, f = sae(x)
        recon_loss = ((x - x_hat) ** 2).sum(dim=-1).mean()
        l1_loss = f.abs().sum(dim=-1).mean()
        loss = recon_loss + L1_LAMBDA * l1_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        # real, standard SAE hygiene: keep decoder columns unit-norm (prevents trivial shrinkage)
        with torch.no_grad():
            sae.W_dec.data /= sae.W_dec.data.norm(dim=0, keepdim=True).clamp(min=1e-6)
        if step % 100 == 0 or step == N_STEPS - 1:
            with torch.no_grad():
                sparsity = (f > 0).float().sum(dim=-1).mean().item()
                frac_var_explained = 1 - ((x - x_hat) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()
            print(f"step {step}: recon={recon_loss.item():.4f} l1={l1_loss.item():.4f} "
                  f"loss={loss.item():.4f} mean_active_features={sparsity:.1f}/{D_DICT} "
                  f"frac_var_explained={frac_var_explained.item():.4f}")

    torch.save({"state_dict": sae.state_dict(), "d_model": D_MODEL, "d_dict": D_DICT, "layer": LAYER}, CKPT)
    print(f"\nSaved real trained SAE to {CKPT}")


if __name__ == "__main__":
    main()
