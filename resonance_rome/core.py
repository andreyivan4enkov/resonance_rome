"""Pure-math core: the resonance mechanic (topology + asymmetric weak/strong
budget transfer) and the two covariance choices it substitutes into ROME/
MEMIT's closed-form edit. No GPT-2/torch dependency -- real numpy only, so
this module can be unit-tested without a GPU or a downloaded model.

Extracted verbatim (not re-derived) from the formula verified across every
script in `src/` -- see docs/METHODS.md for the literal definition.
"""
from __future__ import annotations

import numpy as np

DEFAULT_RATE_BUDGET = 0.2


def topology_and_budget(V: np.ndarray, rate_budget: float = DEFAULT_RATE_BUDGET) -> tuple[np.ndarray, np.ndarray]:
    """Literal Method-3 formula: relu(cosine) topology, asymmetric weak-to-
    strong budget transfer (never a symmetric average).

    topo_ij = relu(cos(v_i, v_j)), topo_ii = 0
    budget0_i = ||v_i||
    transfer_ij = min(rate_budget * relu(topo_ij), whichever of {i,j} has less budget)
    budget_final_i = budget0_i +/- sum_j transfer_ij (strong side gains, weak side loses)

    FIX (found live via a real unit test, 2026-08-28): earlier scripts across
    this whole project defined `cost = -topo` before taking relu(cost). Since
    topo = relu(cos) is never negative, cost was never positive, so
    relu(cost) was IDENTICALLY ZERO for every input tested -- the weak/strong
    transfer never activated in any prior run; budget_final was silently
    always equal to budget0. `cost` is now `+topo` directly (higher real
    resonance -> more real transfer), so relu(cost) is positive exactly when
    there IS real resonance, matching the literal definition: transfer should
    happen BECAUSE of resonance, not in its absence.

    Returns (topo, budget_final), both shape (n, n) and (n,) respectively.
    """
    n = V.shape[0]
    Vn = V / (np.linalg.norm(V, axis=-1, keepdims=True) + 1e-12)
    cos = Vn @ Vn.T
    topo = np.maximum(cos, 0.0)
    np.fill_diagonal(topo, 0.0)
    budget0 = np.linalg.norm(V, axis=-1)
    cost = topo  # FIXED: was `-topo`, which made relu(cost) always zero
    b_i, b_j = budget0.reshape(n, 1), budget0.reshape(1, n)
    weak_is_i = b_i <= b_j
    weak_budget = np.where(weak_is_i, b_i, b_j)
    transfer = np.minimum(rate_budget * np.maximum(cost, 0.0), weak_budget)
    sign = np.where(weak_is_i, -1.0, 1.0)
    budget_final = budget0 + (sign * transfer).sum(axis=1)
    return topo, budget_final


def standard_covariance(peer_keys: np.ndarray) -> np.ndarray:
    """Real ROME/MEMIT baseline: C = mean_j(k_j k_j^T) over real corpus peer
    keys -- a generic, frequency-based second-moment statistic. Unchanged
    from the published method."""
    return (peer_keys.T @ peer_keys) / peer_keys.shape[0]


def resonance_covariance(peer_keys: np.ndarray, target_keys: np.ndarray,
                          rate_budget: float = DEFAULT_RATE_BUDGET) -> np.ndarray:
    """This project's substitution: C = sum_j [ w_j * k_j k_j^T ], where w_j
    is peer j's own real topology+budget resonance to the target key(s) --
    summed across ALL target keys (not averaged), each peer weighted by ITS
    OWN budget_final (not the target's -- a real bug caught and fixed once
    in this project's history, see docs/JOURNEY.md).

    peer_keys: (n_peers, d) real peer key vectors (e.g. GELU-activated MLP
        hidden states from a real, unrelated text corpus).
    target_keys: (n_targets, d) real key vector(s) for the fact/facts being
        edited -- pass a single row for one fact (ROME) or many rows for a
        simultaneous multi-fact edit (MEMIT).
    """
    n_peers = peer_keys.shape[0]
    n_targets = target_keys.shape[0]
    V_for_topo = np.vstack([peer_keys, target_keys])
    topo, budget_final = topology_and_budget(V_for_topo, rate_budget)
    w = np.zeros(n_peers)
    for ti in range(n_targets):
        target_row = n_peers + ti
        w += topo[target_row, :n_peers] * budget_final[:n_peers]  # peer's OWN budget
    return (peer_keys * w[:, None]).T @ peer_keys
