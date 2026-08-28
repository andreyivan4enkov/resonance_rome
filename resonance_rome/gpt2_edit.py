"""GPT-2-specific real editing helpers: real key extraction (GPT-2 uses
Conv1D, not nn.Linear -- weight is (in, out), forward is x @ weight) and the
real ROME/MEMIT closed-form solve, single-fact and joint multi-fact, with
either covariance from core.py substituted in.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .core import null_space_projection, resonance_covariance, standard_covariance

DEFAULT_RIDGE = 1.0
DEFAULT_V_STAR_STEPS = 30
DEFAULT_V_STAR_LR = 0.5


def get_real_key(model, tok, device, text: str, layer: int):
    """Real GELU-activated MLP key vector (c_proj's real input) at the last
    token of `text`. Returns (key_tensor, input_ids)."""
    ids = tok(text, return_tensors="pt")["input_ids"].to(device)
    captured = {}

    def hook(module, inp, out):
        captured["k"] = inp[0][0, -1].detach()

    h = model.transformer.h[layer].mlp.c_proj.register_forward_hook(hook)
    with torch.no_grad():
        model(ids)
    h.remove()
    return captured["k"], ids


def extract_peer_keys(model, tok, device, sentences: list[str], layer: int) -> np.ndarray:
    """Real peer key matrix (n_peers, d_key) from a real, unrelated text
    corpus -- the source for BOTH covariance choices."""
    keys = []
    for s in sentences:
        k, _ = get_real_key(model, tok, device, s, layer)
        keys.append(k.cpu().double().numpy())
    return np.stack(keys)


def _find_v_star(model, tok, device, layer, prompt, target_word,
                  v_star_steps=DEFAULT_V_STAR_STEPS, v_star_lr=DEFAULT_V_STAR_LR):
    """Real short optimization (ROME's own real 'compute target vector' step):
    find the output vector that maximizes real log P(target_word) at this
    position. Returns (k_star, v_orig, v_star)."""
    c_proj = model.transformer.h[layer].mlp.c_proj
    k_star, ids = get_real_key(model, tok, device, prompt, layer)
    v_orig = (k_star @ c_proj.weight.detach())
    delta_v = torch.zeros_like(v_orig, requires_grad=True)

    def add_hook(module, inp, out):
        return out + delta_v

    h = c_proj.register_forward_hook(add_hook)
    opt = torch.optim.Adam([delta_v], lr=v_star_lr)
    target_id = tok(target_word, add_special_tokens=False)["input_ids"][0]
    for _ in range(v_star_steps):
        logits = model(ids).logits[0, -1]
        loss = -F.log_softmax(logits, dim=-1)[target_id]
        opt.zero_grad()
        loss.backward()
        opt.step()
    h.remove()
    v_star = (v_orig + delta_v).detach()
    return k_star, v_orig, v_star


def rome_edit(model, tok, device, layer: int, prompt: str, target_word: str,
              peer_keys: np.ndarray, mode: str = "ours",
              ridge: float = DEFAULT_RIDGE, rate_budget: float = 0.2,
              v_star_steps: int = DEFAULT_V_STAR_STEPS, v_star_lr: float = DEFAULT_V_STAR_LR,
              null_space_eigenvalue_frac: float | None = None) -> None:
    """Real, single-fact ROME closed-form edit, applied in place to `model`.

    mode:
      "standard"      -- real ROME baseline covariance (generic corpus)
      "ours"          -- resonance-weighted covariance (this project's main result)
      "null_generic"  -- AlphaEdit-style null-space projection (Fang et al. 2024),
                         null space built from the GENERIC peer covariance
      "null_resonance" -- this project's hybrid: null space built from the
                         RESONANCE-weighted covariance instead -- resonant
                         peers occupy more of the "preserved" subspace and are
                         thereby MORE strongly protected from the edit

    FIX (caught live, "ты снова упустил больше половины... неизбежно
    заменяешь другой [метод] на классику или добавляешь константы"): the
    null-space modes used to fake AlphaEdit's real hard projection with a
    made-up ridge-style penalty (`null_space_penalty * (I-P)`, an invented
    constant, not their formula), AND used a fixed `1e-2` eigenvalue cutoff
    instead of this project's own established self-calibration principle
    (Рефлексия's `_dynamic_leakage_threshold_pct`: derive thresholds from
    real data, not a hand-picked number). Fixed: the edit's own real key is
    now genuinely PROJECTED through P (`k* -> P k*`, matching AlphaEdit's
    real mechanism), and the eigenvalue cutoff self-calibrates by default.

    SECOND-ORDER CATCH (found immediately after the first fix, same self-
    audit): the first pass regularized with a freshly-computed
    `standard_covariance(peer_keys)` for BOTH null modes, discarding the
    already-built `M`. That silently made `null_resonance` regularize
    IDENTICALLY to `null_generic` -- the exact "fixed one thing, quietly
    reverted another to the classic method" pattern being audited for. The
    regularizer is now `M` itself (the same real matrix the null space P was
    built from -- generic for null_generic, resonance-weighted for
    null_resonance), matching AlphaEdit's own formula, which regularizes
    with the same preserved-knowledge covariance it derives P from, not an
    unrelated recomputation.
    """
    c_proj = model.transformer.h[layer].mlp.c_proj
    k_star, v_orig, v_star = _find_v_star(model, tok, device, layer, prompt, target_word,
                                           v_star_steps, v_star_lr)
    k_star_np = k_star.cpu().double().numpy()
    d = peer_keys.shape[1]
    edit_key = k_star_np  # the real key actually used to build the edit; may be projected below

    if mode == "standard":
        C = standard_covariance(peer_keys)
    elif mode == "ours":
        C = resonance_covariance(peer_keys, k_star_np[None, :], rate_budget)
    elif mode in ("null_generic", "null_resonance"):
        M = (standard_covariance(peer_keys) if mode == "null_generic"
             else resonance_covariance(peer_keys, k_star_np[None, :], rate_budget))
        P = null_space_projection(M, null_space_eigenvalue_frac)
        edit_key = P @ k_star_np  # REAL projection of the edit's own key, not a penalty hack
        C = M  # regularize with the SAME real matrix P was built from (generic or resonance)
    else:
        raise ValueError(f"unknown mode: {mode}")

    C_reg = C + ridge * np.eye(d)
    C_inv = np.linalg.inv(C_reg)
    Cinv_k = C_inv @ edit_key
    denom = float(edit_key @ Cinv_k)
    delta_out = v_star.cpu().double().numpy() - v_orig.cpu().double().numpy()
    Delta = np.outer(delta_out, Cinv_k) / denom  # (d_out, d_key), ROME's W(out,in) convention
    with torch.no_grad():
        c_proj.weight += torch.tensor(Delta.T, dtype=c_proj.weight.dtype, device=device)


def joint_memit_edit(model, tok, device, layer: int, facts: list[tuple[str, str]],
                      peer_keys: np.ndarray, mode: str = "ours",
                      ridge: float = DEFAULT_RIDGE, rate_budget: float = 0.2,
                      v_star_steps: int = DEFAULT_V_STAR_STEPS, v_star_lr: float = DEFAULT_V_STAR_LR) -> None:
    """Real MEMIT joint closed-form edit for N SIMULTANEOUS facts, applied in
    place to `model`. facts: list of (prompt, target_word) pairs.

    Literal MEMIT normal equation (Meng et al. 2022, Eq 14):
        Delta = R @ K^T @ (C + K K^T)^-1
    with C substituted per `mode`, same as rome_edit.
    """
    c_proj = model.transformer.h[layer].mlp.c_proj
    W_orig = c_proj.weight.detach().cpu().double().numpy().T  # (768,3072), ROME's W(out,in) convention

    K_cols, M_cols = [], []
    for prompt, target_word in facts:
        k_star, v_orig, v_star = _find_v_star(model, tok, device, layer, prompt, target_word,
                                               v_star_steps, v_star_lr)
        K_cols.append(k_star.cpu().double().numpy())
        M_cols.append(v_star.cpu().double().numpy())
    K = np.stack(K_cols, axis=1)  # (d_key, N)
    M = np.stack(M_cols, axis=1)  # (d_out, N)

    d = peer_keys.shape[1]
    if mode == "standard":
        C = standard_covariance(peer_keys)
    else:
        C = resonance_covariance(peer_keys, K.T, rate_budget)

    C_reg = C + K @ K.T + ridge * np.eye(d)
    C_inv = np.linalg.inv(C_reg)
    R = M - (W_orig @ K)
    Delta = R @ K.T @ C_inv
    with torch.no_grad():
        c_proj.weight += torch.tensor(Delta.T, dtype=c_proj.weight.dtype, device=device)
