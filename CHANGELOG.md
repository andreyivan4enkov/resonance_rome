# Changelog

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
