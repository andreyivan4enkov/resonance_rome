"""resonance_rome: resonance-weighted covariance for ROME/MEMIT factual
model editing.

This package is a clean, reusable distillation of the exact math verified
across every script in `src/` and every result in `results/` -- it is
written AFTER those experiments, not before, specifically so it can be
imported without re-deriving or silently altering any already-verified
formula. If you want the exact, byte-for-byte script that produced a given
number in the README, look in `src/`; if you want to build something new on
top of the same math, import from here.

Public API:
    topology_and_budget(V, rate_budget=0.2) -> (topo, budget_final)
    standard_covariance(peer_keys) -> C
    resonance_covariance(peer_keys, target_keys, rate_budget=0.2) -> C
    null_space_projection(M, eigenvalue_threshold_frac=None) -> P
    self_calibrated_null_threshold(eigvals) -> float
    rome_edit(model, tok, layer, prompt, target_word, peer_keys, mode, ...) -> None
"""

from .core import (
    topology_and_budget,
    standard_covariance,
    resonance_covariance,
    null_space_projection,
    self_calibrated_null_threshold,
)
from .gpt2_edit import get_real_key, extract_peer_keys, rome_edit, joint_memit_edit

__all__ = [
    "topology_and_budget",
    "standard_covariance",
    "resonance_covariance",
    "null_space_projection",
    "self_calibrated_null_threshold",
    "get_real_key",
    "extract_peer_keys",
    "rome_edit",
    "joint_memit_edit",
]

__version__ = "0.1.0"
