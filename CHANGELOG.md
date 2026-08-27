# Changelog

Mapped to the 2026-08-27 research session journal
(`D:\RLM\_external\sandbox`, notes/SESSION_SUMMARY_2026-08-27.md §§8bb–8jj).
Iteration letters H–N were activation-steering / SAE dead ends; they stay in
`docs/JOURNEY.md`, not as release tags.

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
