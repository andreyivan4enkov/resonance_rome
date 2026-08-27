# Resonance-Weighted ROME: topology+budget covariance for factual model editing

**Status: real, reproducible pilot result on GPT-2-small. Not benchmarked at
scale, not peer-reviewed, not tested on standard editing benchmarks
(COUNTERFACT/zsRE). A draft, not a finished method.**

## The idea

[ROME](https://github.com/kmeng01/rome) (Meng et al., MIT/Northeastern/Technion,
2022) edits a single fact into a GPT model with a closed-form rank-one update
to one MLP layer's down-projection weight. To avoid damaging unrelated facts,
ROME's formula needs a "what to protect" term: a covariance matrix `C` over
real key vectors from a large generic text corpus (an *inductive*, frequency-based
statistic — protect what's common, regardless of whether it's actually related
to the new fact).

This project replaces that generic corpus statistic with a **resonance-weighted
covariance**: real cosine-similarity topology between the new fact's key vector
and a set of real peer key vectors, combined with an asymmetric "weak/strong"
budget-transfer rule (full definition in
[`docs/METHODS.md`](docs/METHODS.md)). Instead of protecting "whatever is
common," it protects "whatever actually resonates with the new fact" — a
content-aware, *deductive* criterion instead of a generic, *inductive* one.

## What was actually found

Same fact (`"The secret code word is" -> " banana"`), same GPT-2-small, same
ROME closed-form machinery — the ONLY thing changed is the covariance `C`.

| layer | standard ROME (ppl / facts kept) | resonance C (ppl / facts kept) |
|---|---|---|
| 0 | 57.15 / 6  | 57.15 / 6  (resonance loses here) |
| 1 | 60.50 / 4  | 60.95 / 4  (resonance loses here) |
| 2 | 42.66 / 9  | 41.64 / 9  |
| 3 | 37.78 / 10 | 37.05 / 13 |
| 4 | 37.08 / 8  | 36.48 / 9  |
| 5 | 37.33 / 10 | 36.73 / 10 |
| 6 | 36.75 / 10 | 36.48 / 10 |
| 7 | 36.51 / 12 | 36.39 / 13 |
| 8 | 36.59 / 11 | 36.47 / 13 |
| 9 | 36.53 / 13 | 36.39 / 13 |
| 10 | 36.53 / 11 | 36.41 / 13 |
| 11 | 36.71 / 11 | 36.51 / 13 |

(baseline: ppl=36.33, 14/14 facts kept; in every single cell above, the new
fact WAS successfully learned — "banana" became the top-1 completion. Only
collateral damage differs.)

**Pattern**: from layer 2 onward (10 of 12 layers), resonance weighting causes
strictly less collateral damage than the published baseline — never worse, often
by a wide margin. At layers 0-1, the classic corpus statistic wins.

### Is this a fluke of one fact?

Re-ran the same sweep on 6 different new facts (`banana`, `lighthouse`,
`compass`, `pumpkin`, `trumpet`, `volcano` — all single-BPE-token answers, to
avoid a tokenization confound found and documented along the way). Instead of
hand-picking the layer split, a self-calibrated selection rule decided it from
real accumulated evidence per layer (adapted from an existing project's
Markov-blanket "K≥S operational closure" check — real improvement tempo vs
real regression tempo, see `docs/METHODS.md`):

```
n=3 facts: {0: standard, 1: standard, 2: ours, 3: ours, ..., 11: ours}
n=6 facts: {0: ours,     1: standard, 2: ours, 3: ours, ..., 11: ours}
```

Layer 0 flipped between samples (small-sample noise at the edge) — but **the
core pattern (layer 2 onward = resonance wins) reproduced identically across
both independent samples**, and fact-learning success was 100% on the clean
6-fact sample (the one earlier fact that failed everywhere was a tokenization
artifact: its answer was accidentally 3 BPE tokens, not one — see
`docs/JOURNEY.md`).

### Multi-layer decomposition (MEMIT-style) — tried twice, failed worse the second time

A single edit was decomposed across multiple layers (layers 1-4, spanning both
a "standard"-preferring and "resonance"-preferring layer), each layer using its
own self-calibrated best covariance, following MEMIT's (Meng et al. 2022b)
idea of spreading one fact's edit across several layers instead of
concentrating it at one.

**v1 (naive, fixed even split)**: divide the total needed change into 4 equal
shares, computed once from the clean model, add one fixed share at each layer.
Result: worse than a single-layer edit on all 3 facts
(`results/memit_decomposition.txt`).

When asked directly *"did you actually adapt everything to MEMIT?"* — no: v1
never checked how much of the target change earlier layers' edits had already
produced by the time the signal reached the final layer, so four fixed
additions could cumulatively overshoot the real target.

**v2 (fixed to re-estimate the REMAINING gap at each step)**: after each
layer's edit, re-measure the model's REAL current output at the final layer
and only distribute what's still missing among the layers not yet edited —
the direct fix for v1's identified flaw. Result: **substantially worse than
v1**, not better (`results/memit_decomposition_v2_remaining_gap.txt`):

| fact | single-layer | v1 (fixed even split) | v2 (remaining-gap re-estimation) |
|---|---|---|---|
| banana | ppl +0.07 / facts 9 | ppl +2.26 / facts 9 | ppl +11.74 / facts 6 |
| lighthouse | ppl +13.00 / facts 8 | ppl +10.99 / facts 7 | ppl +28.60 / facts 7 |
| compass | ppl +4.00 / facts 11 | ppl +23.04 / facts 7 | ppl **+126.69** / facts 6 |

**Honest diagnosis**: both versions assume a layer's rank-1 weight edit
contributes to the final output roughly *additively*, independent of the
other edited layers. That assumption appears to be wrong here: editing layer
1's weight changes the hidden state feeding into layers 2-4 for EVERY input
(not just the target key), so the real propagated effect of an early edit,
once processed by the (still unedited) later layers' own attention/MLP
nonlinearities, can be far larger or differently-signed than a simple
additive model predicts. v2's remaining-gap re-estimation reacts to that
mismatch by pushing an increasingly large correction onto a shrinking set of
remaining layers — visible directly in the numbers (the error compounds
worst on the *last* fact/layer, compass: +126.69).

**Conclusion**: neither sequential, greedy layer-by-layer decomposition
variant tried here works for this setup. Real MEMIT's actual multi-layer
distribution is not a simple sequential greedy process — it's closer to a
joint solve across the layer range — which has not been attempted here. Both
sequential variants are left as honest, documented negative results; the
single-best-layer edit (the main result above) remains the only version of
this project's approach that actually reduces collateral damage relative to
the published baseline.

## What this does NOT show

- Not tested on models larger than GPT-2-small (124M params).
- Not tested on the field's standard editing benchmarks (COUNTERFACT, zsRE).
- Not tested with many simultaneous edits (MEMIT's actual real-world use case
  is thousands of facts at once; this is single-fact only).
- n=6 facts is still a small sample; the layer-0/1 boundary specifically showed
  sample-dependent noise.
- A literature search (see `docs/JOURNEY.md`) found no identical prior
  combination of resonance/topology-weighted covariance substituted into
  ROME's closed form — it appears to be a real, small, undocumented
  combination, not a rediscovery of a known technique, but this has not been
  independently verified by anyone else.

## Repository layout

```
src/
  rome_resonance_edit.py            single-layer ROME, standard vs resonance C, all-layer sweep
  reflection_autopoiesis_hybrid.py  multi-fact K>=S self-calibration of the per-layer hybrid rule
  real_sae_features.py              sparse autoencoder used in an earlier, separate (largely
                                     negative-result) line of investigation -- see JOURNEY.md
docs/
  METHODS.md      literal mathematical definitions (topology, budget transfer, K>=S check)
  JOURNEY.md       honest chronological account of everything tried, including the failures
results/
  layer_sweep_1fact_banana.txt      raw output, single-fact all-layer sweep
  reflection_sweep_3facts.txt       raw output, 3-fact self-calibration sweep
  reflection_sweep_6facts.txt       raw output, 6-fact self-calibration sweep (clean, no tokenization confound)
  memit_decomposition.txt           raw output, multi-layer decomposition test
```

## Reproducing

Requires `transformers`, `torch` (CUDA optional), a local GPT-2 checkpoint
(downloaded automatically via `transformers` on first run, or point
`HF_HOME`/`TRANSFORMERS_CACHE` at a local cache), and
`benchmarks/hotpot_dev_distractor_v1.json` (real-sentence source for the
peer-key corpus and held-out perplexity set — any real English text corpus
works, this one was just what was on hand).

```bash
python src/rome_resonance_edit.py                    # single-fact, all-layer sweep
python src/reflection_autopoiesis_hybrid.py           # multi-fact self-calibration
```

## Attribution

Built on real, published prior work: ROME (Meng, Bau, Andonian, Belinkov,
2022, MIT/Northeastern/Technion) for the closed-form editing machinery, and
MEMIT (Meng et al., 2022) for the multi-layer decomposition idea. The
resonance/topology+budget weighting and the K≥S self-calibration adaptation
are original to this project.
