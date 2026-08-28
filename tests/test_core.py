"""Real unit tests for the pure-math core -- no GPU, no model download.
Run with: pytest tests/test_core.py
"""
import numpy as np
import pytest

from resonance_rome.core import (
    null_space_projection,
    resonance_covariance,
    standard_covariance,
    topology_and_budget,
)


def _random_vectors(n, d, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d))


def test_topology_is_symmetric_and_bounded():
    V = _random_vectors(12, 8, seed=0)
    topo, _ = topology_and_budget(V)
    assert np.allclose(topo, topo.T), "topo must be symmetric: cos(v_i,v_j) == cos(v_j,v_i)"
    assert np.all(topo >= 0.0) and np.all(topo <= 1.0 + 1e-9), "relu(cos) must lie in [0,1]"
    assert np.allclose(np.diag(topo), 0.0), "no self-coupling"


def test_total_budget_is_conserved():
    """The weak/strong transfer is a real zero-sum exchange between pairs
    (transfer_ij == transfer_ji in magnitude, opposite sign) -- so the TOTAL
    budget across all nodes must be exactly conserved, never created or
    destroyed. This is the literal "conserved budget" property from
    docs/METHODS.md, and a real, checkable invariant of the formula."""
    V = _random_vectors(20, 16, seed=1)
    _, budget_final = topology_and_budget(V)
    budget0 = np.linalg.norm(V, axis=-1)
    assert np.isclose(budget_final.sum(), budget0.sum(), rtol=1e-9), \
        "total budget must be conserved -- transfer only moves budget between nodes, never creates/destroys it"


def test_weak_node_never_loses_more_than_its_own_budget():
    V = _random_vectors(15, 10, seed=2)
    _, budget_final = topology_and_budget(V)
    assert np.all(budget_final >= -1e-9), \
        "a node can lose at most its own full budget (capped by weak_budget) -- budget_final must stay >= 0"


def test_standard_covariance_is_literal_mean_outer_product():
    peer_keys = _random_vectors(30, 6, seed=3)
    C = standard_covariance(peer_keys)
    C_manual = sum(np.outer(k, k) for k in peer_keys) / len(peer_keys)
    assert np.allclose(C, C_manual), "must be the literal mean_j(k_j k_j^T), no hidden weighting"


def test_weak_strong_transfer_actually_activates():
    """Regression test for a real bug found live (2026-08-28) via THIS test
    file: the original formula used `cost = -topo` before relu, and since
    topo = relu(cos) is never negative, relu(cost) was IDENTICALLY ZERO for
    every real input ever tried across this whole project -- transfer never
    activated, budget_final was silently always == budget0. Two vectors that
    are genuinely similar (real, nonzero topo) with different norms MUST now
    show real, nonzero transfer -- budget_final must differ from budget0."""
    similar_a = np.array([1.0, 0.0, 0.0]) * 1.0   # small budget
    similar_b = np.array([1.0, 0.01, 0.0]) * 5.0  # large budget, nearly same direction
    V = np.stack([similar_a, similar_b])
    _, budget_final = topology_and_budget(V)
    budget0 = np.linalg.norm(V, axis=-1)
    assert not np.allclose(budget_final, budget0), \
        "two REALLY resonant (high-topo) vectors with different budgets must show real transfer -- if this " \
        "is exactly budget0, the weak/strong mechanic has regressed to a no-op again"
    assert budget_final[0] < budget0[0], "the weaker (smaller-budget) node must LOSE budget to the stronger one"
    assert budget_final[1] > budget0[1], "the stronger (larger-budget) node must GAIN budget from the weaker one"


def test_weak_strong_transfer_conserved_even_when_active():
    """The conservation property (test_total_budget_is_conserved) must still
    hold now that transfer is real and nonzero, not just trivially true
    because nothing moves."""
    V = _random_vectors(20, 16, seed=1)
    _, budget_final = topology_and_budget(V)
    budget0 = np.linalg.norm(V, axis=-1)
    assert not np.allclose(budget_final, budget0), "sanity: transfer must be real and active for this fixture"
    assert np.isclose(budget_final.sum(), budget0.sum(), rtol=1e-9), \
        "total budget must still be exactly conserved now that transfer actually moves it around"


def test_resonance_covariance_zero_when_no_real_similarity():
    """A peer orthogonal to every target contributes zero weight (relu(cos)=0
    for an exactly orthogonal pair), so it must not appear in C at all."""
    d = 4
    peer_orthogonal = np.array([[1.0, 0.0, 0.0, 0.0]])
    target = np.array([[0.0, 1.0, 0.0, 0.0]])
    C = resonance_covariance(peer_orthogonal, target)
    assert np.allclose(C, 0.0), "an exactly orthogonal peer must contribute zero weight"


def test_null_space_projection_is_idempotent_and_symmetric():
    """A real orthogonal projection matrix must satisfy P^2 == P and P == P^T --
    basic sanity that null_space_projection returns a genuine projector."""
    V = _random_vectors(10, 6, seed=5)
    M = standard_covariance(V)
    P = null_space_projection(M, eigenvalue_threshold_frac=0.5)
    assert np.allclose(P @ P, P, atol=1e-8), "a projection matrix must be idempotent: P@P == P"
    assert np.allclose(P, P.T, atol=1e-8), "a projection matrix built from real symmetric eigenvectors must be symmetric"


def test_null_space_projection_kills_the_dominant_direction():
    """M's top eigenvector (its most heavily 'occupied' real direction) must
    NOT be in the null space -- projecting it through P should collapse it
    close to zero, since P projects onto directions M does NOT occupy."""
    V = _random_vectors(15, 8, seed=6)
    M = standard_covariance(V)
    eigvals, eigvecs = np.linalg.eigh(M)
    top_direction = eigvecs[:, -1]  # largest eigenvalue
    P = null_space_projection(M, eigenvalue_threshold_frac=0.1)
    projected = P @ top_direction
    assert np.linalg.norm(projected) < np.linalg.norm(top_direction) * 0.5, \
        "M's most heavily-occupied direction must be strongly suppressed by its own null-space projection"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
