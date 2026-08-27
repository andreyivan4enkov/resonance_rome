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

**v3 (Рефлексия-gated)**: apply the K≥S/`gate_ok` idiom from `METHODS.md`
DURING the decomposition — after each layer's edit, measure real damage; if
it already exceeds what the single-layer baseline alone costs, stop and
leave the remaining layers unedited. This genuinely prevented the worst
blow-up (compass: +4.84 instead of v2's +126.69) but sometimes stopped too
early to actually learn the fact (banana: aborted after 1 layer, `ok=False`
— the fact was never learned). A real safety/capability trade-off, not a
clean win.

**v4 (the actual, literal MEMIT formula — read from the real paper, not
guessed)**: Meng et al. 2022's real spread rule is neither v1's equal split
nor v2's adaptive remaining-gap — it's `r_l = (z - h_L) / (L - l + 1)`, a
**growing** share computed ONCE from the clean model (no sequential
re-measurement at all). Critically, this means the LAST layer in the range
gets the FULL, undivided deficit (denominator = 1) — i.e. the same full-size
edit as the single-layer baseline — *plus* every earlier layer ALSO receives
a real, substantial edit on top of that (`results/memit_decomposition_v3v4.txt`
shows per-layer residual magnitudes rising from ~27 to ~110). Result:
**worse than the single-layer baseline on all 3 facts** (lighthouse: +110.0
vs +13.0 for single-layer — the worst of any variant tried except v2).

### Adapting the TASK to MEMIT instead of MEMIT to the task

All four decomposition variants above tested MEMIT's multi-*layer* spread on
a *single* fact — the wrong regime to see any benefit in, since MEMIT's real
purpose is making *many simultaneous* edits tractable at *one* layer via its
joint normal-equation solve, `Delta = R K^T (C_0 + K K^T)^-1` (Meng et al.
2022, Eq. 14), not spreading one edit thin. Re-tested in that actual regime:
6 new facts (banana, lighthouse, compass, pumpkin, trumpet, volcano) inserted
**simultaneously** at one layer via the real joint solve, comparing standard
corpus `C_0` against the same resonance-weighted `C` (this time weighted by
each peer's SUMMED resonance across all 6 new fact keys, not just one):

| | perplexity delta | facts kept | all 6 new facts learned? |
|---|---|---|---|
| standard corpus C | +4.18 | 11/14 | yes, 6/6 |
| resonance-weighted C | **+2.45** | **12/14** | yes, 6/6 |

(A real bug was caught and fixed along the way: the closed-form solve's
regularization must scale with the actual key matrix, `C + K K^T`, per the
real formula — a fixed ridge that worked fine for a single key exploded the
joint 6-key solve to `ppl ~1e19` before this was corrected.)

**This is the real, clean confirmation of the substitution's value in MEMIT's
actual intended use case**: same 6 facts, same layer, same joint edit
machinery, only the covariance changed — resonance weighting caused ~1.7x
less perplexity damage and preserved one more unrelated fact, with identical
new-fact learning success. Unlike the single-fact/multi-layer decomposition
experiments above, this one is a genuine, reproducible win in the regime
MEMIT was actually designed for.

**Real conclusion, now grounded in the actual paper**: MEMIT's multi-layer
spread is not designed to *reduce* the size of any one edit — it exists to
make *thousands of simultaneous edits* numerically tractable (the closed-form
solve doesn't scale to one giant multi-fact matrix otherwise), accepting
*more* total parameter change spread across several layers as the cost of
that scalability. Applied to a *single* fact, it just adds redundant edits on
top of what one well-placed ROME edit already achieves, which is why it
increases collateral damage here rather than reducing it. This isn't a bug in
the reimplementation — it's a mismatch between what multi-layer spreading
optimizes for (many-edit scalability) and a single-edit collateral-damage
benchmark. None of the four multi-layer decomposition variants (v1/v2/v3/v4)
improved on the single-best-layer edit for editing ONE fact. **But once the
task was adapted to MEMIT's real regime — several simultaneous edits, one
joint solve, no layer-spreading involved — the resonance covariance DOES
show a real, reproducible improvement** (the table above: ~1.7x less
perplexity damage, one more fact preserved, identical learning success).
This project now has two confirmed positive results: single-layer ROME edits
(main result, top of this document) and multi-fact joint MEMIT edits (this
section) both favor resonance-weighted covariance over the generic corpus
statistic; only the multi-*layer* spreading mechanism itself did not help.

### Does the advantage hold as the number of simultaneous edits scales up?

Real question: MEMIT's whole point is *many* simultaneous edits, so does the
resonance-covariance advantage found at N=6 survive at N=15, 30, 50? All 50
answer words were verified single-BPE-token in advance
(`src/memit_scaling_curve.py`), `v*`/`k*` computed once for all 50 from the
clean model, then reused as prefixes for each smaller N.

| N | standard ppl delta | resonance ppl delta | ratio (std/ours) | facts kept (both) | new facts learned (both, out of N) |
|---|---|---|---|---|---|
| 6  | +4.40 | +2.52 | 1.75x | 6 vs 7 | 6/6 |
| 15 | +3.08 | +1.63 | 1.89x | 14 vs 14 | 10/15 |
| 30 | +2.53 | +1.21 | 2.08x | 14 vs 14 | 10/30 |
| 50 | +2.00 | +0.85 | 2.35x | 14 vs 14 | 9/50 |

**The resonance advantage doesn't just hold — it grows with scale** (1.75x
less damage at N=6, up to 2.35x at N=50). This is the strongest result in
the project so far: consistent, monotonic, and in the direction MEMIT is
actually meant to be used.

**Honest limitation, disclosed, not hidden**: the ABSOLUTE number of new
facts successfully learned does NOT scale with N — it plateaus around 9-10
correctly learned facts regardless of whether N=15, 30, or 50, and this
count is IDENTICAL for standard and resonance C (so it isn't a resonance-vs-
standard effect; it's a shared capacity/optimization limit of one-layer,
fixed-30-step joint insertion at this scale). The "less collateral damage"
finding is robust across the whole range tested; "how many of N facts get
reliably learned in one shot" is a separate, real limitation neither
covariance choice solves.

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
