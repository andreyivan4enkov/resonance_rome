# Changelog

## 1.0.1 — 2026-08-28

- Caught a real methodological substitution in the cluster-mechanism test:
  used standard symmetric spectral clustering instead of this project's own
  asymmetric attractor logic. Redesigned using the already-established
  gravitational-center mechanic (each neuron follows its own real attractor
  to a fixed point). Real result: 48 self-organized clusters, but the
  ridge-vs-null-space energy difference remained small — the fix didn't
  change the conclusion, reinforcing rather than undermining it


## 1.0.0 — 2026-08-28 (finalized for this research phase)

- Final pass: all 9 unit tests green, all file references in README/docs
  verified to exist, working tree clean and pushed
- This closes the resonance-weighted-covariance line of investigation as a
  standalone result. Follow-on work (embedding this logic into a purpose-
  built architecture rather than searching for it inside GPT-2) continues
  in a separate project — see `docs/JOURNEY.md` for the closing rationale


## 0.12.0 — 2026-08-28

- Two direct, real follow-up tests, both honest nulls: (1) cascading/
  frequency-lowering aliasing across all 12 real layer pairs (caught and
  fixed a real confound mid-test before reporting) — weak signal, r=0.145
  vs depth; (2) real spectral-clustering-based test of whether ridge vs
  null-space resonance hit different real neuron clusters — broadly similar
  profiles, inconclusive


## 0.11.0 — 2026-08-28

- Scaled the null-space hybrid test to N_PEERS=2000 (from 200) — confirmed
  the earlier diagnosis: generic and resonance null-space are no longer
  identical at proper scale
- New finding: resonance null-space *inverts* the main result's trade-off
  (better PS at 1.000, worse NS at 0.030) instead of reproducing it — using
  resonance to select the protected subspace behaves differently from using
  it to reweight the ridge-regularized covariance
- Tested a new hierarchy hypothesis (layer below = "working memory"/aliasing
  for the layer above, own weights = "long-term memory", literal decimated
  sum per principle 10) — small, real but inconclusive effect on generic
  null-space PS only (0.850→0.900) at n=10


## 0.10.0 — 2026-08-28

- Implemented and tested the proposed hybrid: resonance-weighted null-space
  projection (`resonance_rome.core.null_space_projection`), inspired by
  AlphaEdit (Fang et al. 2024) — a disclosed, simplified re-implementation
  (SVD + ridge-style penalty), not their exact multi-term closed form
- Real test on 10 COUNTERFACT cases: generic and resonance-weighted null
  space gave IDENTICAL results, and neither beat plain resonance weighting
- Real cause diagnosed: 200 real peer keys against a 3072-dim space leaves
  ~93% of directions exactly zero from sample scarcity alone (AlphaEdit
  itself uses ~100,000 preserved keys) — a test-scale limitation, not
  evidence against the hybrid idea


## 0.9.0 — 2026-08-28

- Portfolio polish: explicit "what this repo demonstrates" (process, not
  just numbers), a Mermaid mechanism diagram, and a real Related Work
  comparison against AlphaEdit (ICLR 2025 Outstanding Paper), "Beyond the
  Covariance Trap" (2026), and other real 2024-2026 work
- Framed the PS (paraphrase) cost as a *predicted* consequence of deduction
  (narrow, resonance-specific) vs induction (broad, frequency-based), not an
  incidental weakness — a falsifiable prediction for future scale tests


Mapped to the 2026-08-27/28 research session (§§8bb–8oo+). Full narrative:
`docs/JOURNEY.md`. Iteration letters H–N were activation-steering / SAE dead
ends; they stay in JOURNEY, not as release tags.

## 0.8.0 — 2026-08-28

- Packaged the verified math into a real library (`resonance_rome/core.py`,
  `resonance_rome/gpt2_edit.py`) with real pytest unit tests (`tests/`), no
  GPU/model needed to run them
- Writing a real test for "total budget is conserved" caught a THIRD real
  bug: `cost = -topo` before `relu` made the weak/strong transfer
  IDENTICALLY ZERO for every input ever tested across this whole project —
  `budget_final` was silently always `== ||v||`. Fixed to `cost = +topo`
- Re-verified the 6-fact joint edit, the N=6..50 scaling curve, AND the
  n=100 real COUNTERFACT benchmark with the fix active: all four checks
  unchanged to within one flipped case (results/*_fixed_transfer.txt) —
  fourth consecutive confirmation the headline trade-off is robust

## 0.7.0 — 2026-08-28

- Real COUNTERFACT benchmark (Meng et al. 2022's own 21,919-case dataset,
  downloaded live) — `src/counterfact_benchmark.py`
- First pass (untuned layer=9): catastrophic NS for both C choices
  (0.005 / 0.065) — layer never tuned for real, diverse facts
- After a real 9-config layer/ridge sweep (layer=4, ridge=1.0): honest
  specificity/generalization trade-off emerges — resonance wins NS (~1.5–2.2x)
  at a real cost on PS, confirmed at n=20 and scaled to n=100

## 0.6.0 — 2026-08-28

- Caught (by direct challenge: "не забыл ли ты опять часть моей логики?")
  a real bug in the multi-fact resonance weighting: it used the EDITED
  FACT's own budget as the multiplier for every peer, instead of each
  PEER's own budget (the single-fact version always had this right)
- Fixed; re-verified the 6-fact joint edit and the N=6..50 scaling curve —
  both unchanged to the 4th decimal place

## 0.5.1 — 2026-08-28

- GitHub / ML-industry packaging to match `moe-orbit-prefetch`: Apache-2.0,
  NOTICE, ATTRIBUTION, CITATION.cff, CODE_OF_CONDUCT, CONTRIBUTING, SECURITY
- Topics, research labels, and git tags for the O→S line
- README.ru.md / README.zh.md landing pages

## 0.5.0 — 2026-08-27

- Scaling curve N=6→15→30→50 joint MEMIT edits (`src/memit_scaling_curve.py`)
- Resonance advantage grows with scale (1.75x → 2.35x less perplexity damage)
- Disclose the separate learning-capacity plateau (~9–10 facts learned, same
  for both C choices) — sandbox §8jj / iteration S

## 0.4.0 — 2026-08-27

- Adapt the *task* to MEMIT’s real regime: simultaneous multi-fact joint edit
  (`src/memit_joint_multi_fact.py`) — sandbox §8ii / iteration R
- Catch and fix a real regularization bug (`C + K K^T`, not a fixed ridge)
- After the fix: resonance C +2.45 ppl / 12 facts vs standard +4.18 / 11;
  both learn 6/6 new facts

## 0.3.0 — 2026-08-27

- Real MEMIT spread formula from the paper (`r_l = (z − h_L) / (L − l + 1)`)
  plus Reflection-gated abort (v3) — sandbox §8hh
- Honest result: neither beats a single-layer edit on one fact

## 0.2.0 — 2026-08-27

- Correct the MEMIT-decomposition claim: v1 was an incomplete adaptation
  (caught by the user); v2 remaining-gap re-estimation made it worse
  — sandbox §8gg / iteration Q

## 0.1.0 — 2026-08-27

- Initial public freeze: resonance-weighted ROME on GPT-2-small, all 12 layers
- Self-calibrated hybrid rule (K≥S / Рефлексия) on 3- and 6-fact samples
- Honest journey docs; tokenization confound (`thunderclap`) documented
  — sandbox §§8bb–8ff / iterations O–P
