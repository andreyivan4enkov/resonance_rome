# Literal method definitions

## Topology + budget (the core resonance mechanic)

For a set of real vectors `V = {v_1, ..., v_n}` (SAE decoder directions, MLP
key vectors, embedding rows — whatever object is being compared):

```
cos_ij = cosine(v_i, v_j)
topo_ij = relu(cos_ij)                       # only constructive resonance counts, never negative pull
topo_ii = 0                                   # no self-coupling
budget0_i = ||v_i||   (or: real activation strength, when available)
cost_ij = -topo_ij
weak(i,j) = the one of {i,j} with the smaller budget0
transfer_ij = min(RATE_BUDGET * relu(cost_ij), budget0_weak(i,j))
budget_final_i = budget0_i + sum_j [ +transfer_ij if i is the strong side,
                                      -transfer_ij if i is the weak side ]
```

`RATE_BUDGET = 0.2` throughout. This is an asymmetric "weak loses to strong,
capped at the weak side's own total budget" transfer rule — never a symmetric
average, never dividing by n. Combination across peers is always a SUM, never
a mean.

## ROME's closed-form edit (real, published, unchanged)

GPT-2's MLP down-projection (`c_proj`, applied to the GELU-activated hidden
state) is treated as a linear key-value associative memory: `v = W @ k`,
`k` = real GELU-activated hidden state ("key"), `v` = its contribution to the
residual stream ("value"), `W = c_proj.weight` (note: GPT-2 uses `Conv1D`,
which stores this transposed relative to `nn.Linear` — `k @ Conv1D.weight`
gives `v` directly).

```
k* = real key vector for the fact's prompt at the target layer
v_orig = W @ k*
v* = v_orig + delta_v, delta_v found by a real short Adam optimization
     (maximizing log P(target_token) at that position)
C = a covariance-like matrix over real "peer" key vectors (see below --
    this is the ONLY thing this project changes)
Delta = (v* - v_orig) (C^-1 k*)^T / (k*^T C^-1 k*)
W_new = W + Delta
```

`C + RIDGE*I` (RIDGE=1.0) is used for invertibility in both variants below,
identically, so the comparison stays fair.

### Standard ROME's C (unchanged, the real baseline)

```
C_std = mean_j (k_j k_j^T)     over real peer key vectors from a generic text corpus
```

A real second-moment (covariance) statistic: protects whatever key directions
are common in real text, regardless of relevance to the new fact.

### This project's substitution

```
C_ours = sum_j [ topo(k_j, k*) * budget_final_j * (k_j k_j^T) ]
```

Same real peer key vectors as `C_std`, reweighted by literal resonance to the
specific new key `k*` instead of generic frequency. SUM, not mean (per the
combination rule above) — trace-matched to `C_std`'s scale when blended (see
the abandoned linear-blend attempt in JOURNEY.md).

## K≥S self-calibration ("Рефлексия"/autopoiesis, adapted)

Adapted from `NeuroStudio_v4/backend/blanket_interface.py`'s
`autopoiesis_k_ge_s_check` (K = real improvement tempo, S = real regression
tempo, K≥S = "operational closure" — a system's own rate of genuine
improvement must not fall behind its own rate of genuine regression for it to
be considered self-sustaining).

For each real (fact, layer, mode) edit attempt:
```
actual_delta = -(ppl_after - ppl_before) / ppl_before
```
Per (layer, mode), across all facts tried at that layer:
```
K_set = { attempts with actual_delta > 0 }, K = mean(actual_delta over K_set)
S_set = { attempts with actual_delta < 0 }, S = mean(|actual_delta| over S_set)
margin = K - S
```
The winning mode at a layer is whichever has the larger real `margin`,
computed from actually-accumulated attempts across multiple real facts — not
a layer index decided in advance from a single observation.

In every real sweep run so far, `K = 0.0` for both modes at every layer: no
edit ever literally IMPROVES baseline perplexity, so the comparison is
entirely decided by which mode does LESS relative harm (`S`).
