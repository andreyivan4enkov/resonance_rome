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


def self_calibrated_null_threshold(eigvals: np.ndarray) -> float:
    """Derive the null/occupied cutoff from the REAL eigenvalue spectrum's
    own largest relative gap, instead of a fixed, hand-picked fraction
    (caught live: `eigenvalue_threshold_frac=1e-2` was exactly the kind of
    arbitrary constant this project's own Рефлексия self-calibration
    principle -- `_dynamic_leakage_threshold_pct`, deriving a threshold from
    real accumulated data rather than a fixed number -- already exists to
    avoid). Sort real eigenvalues ascending; the boundary between "null" and
    "occupied" is placed at the real index with the largest ratio jump
    (eigvals[i+1]/eigvals[i]), i.e. the spectrum's own natural elbow."""
    sorted_vals = np.sort(np.clip(eigvals, 1e-12, None))
    ratios = sorted_vals[1:] / sorted_vals[:-1]
    elbow_idx = int(np.argmax(ratios))  # index i: real gap between sorted_vals[i] and [i+1]
    return float(sorted_vals[elbow_idx])


def null_space_projection(M: np.ndarray, eigenvalue_threshold_frac: float | None = None) -> np.ndarray:
    """AlphaEdit-STYLE null-space projection (Fang et al. 2024, arXiv:2410.02355,
    Eq. 8-9) -- a simplified, disclosed re-implementation, not a literal
    reproduction of their exact multi-term closed form (not verified against
    the paper's own equations directly, only a secondary extraction of them).

    Real SVD of a real second-moment matrix M (M = K0 K0^T in AlphaEdit's own
    notation, built from real "preserved knowledge" keys): eigenvectors with
    eigenvalues below `eigenvalue_threshold_frac * max_eigenvalue` are treated
    as the (approximate) null space -- directions M has near-zero real
    presence in. P = U_hat U_hat^T projects any vector/matrix onto that null
    space; an edit constrained to move only within P's range cannot disturb
    what M represents, by construction.

    This project's hybrid: build M from `resonance_covariance` (below)
    instead of a generic corpus second moment -- so what counts as "already
    occupied, must be preserved" is decided by REAL resonance to the fact
    being edited, not by generic frequency alone.

    `eigenvalue_threshold_frac`: leave as None (default) to self-calibrate
    the cutoff from M's own real eigenvalue spectrum (its largest natural
    gap, see `self_calibrated_null_threshold`) -- a fixed fraction is only
    used if explicitly passed, and even then is a disclosed override, not
    the recommended path.
    """
    eigvals, eigvecs = np.linalg.eigh(M)  # M is symmetric PSD by construction
    if eigenvalue_threshold_frac is None:
        threshold = self_calibrated_null_threshold(eigvals)
    else:
        threshold = eigenvalue_threshold_frac * eigvals.max()
    null_mask = eigvals < threshold
    U_hat = eigvecs[:, null_mask]
    return U_hat @ U_hat.T


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
