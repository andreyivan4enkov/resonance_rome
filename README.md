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

### Multi-layer decomposition (MEMIT-style) — tried, did not help

A single edit was also decomposed across multiple layers (layers 1-4, spanning
both a "standard"-preferring and "resonance"-preferring layer), each layer using
its own self-calibrated best covariance, following MEMIT's (Meng et al. 2022b)
approach of spreading one fact's edit across several layers instead of
concentrating it at one.

**Honest result: this made things WORSE, not better**, on all 3 facts tested
against the single-best-layer baseline (`results/memit_decomposition.txt`):

| fact | single-layer ppl delta / facts kept | MEMIT-style (4 layers) ppl delta / facts kept |
|---|---|---|
| banana | +0.07 / 9 | +2.26 / 9 |
| lighthouse | +13.00 / 8 | +10.99 / 7 |
| compass | +4.00 / 11 | +23.04 / 7 |

The implementation here splits the target residual-stream change *evenly*
across the 4 layers without re-estimating, at each step, how much of that
change earlier layers' edits already produced by the time the residual stream
reaches later layers (real MEMIT does this re-estimation; this simplified
version does not — disclosed in `src/memit_style_decomposition.py`). That
missing step is the most likely reason this naive decomposition compounds
damage instead of reducing it. Left here as an honest negative result, not
hidden — a more faithful MEMIT re-implementation might behave differently,
but that has not been tried.

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
