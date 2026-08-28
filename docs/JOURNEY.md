# Journey: everything tried, including what failed

This project grew out of a long research session testing whether a set of
custom-defined mechanics (asymmetric "topology + budget" resonance, described
in `METHODS.md`) could reduce catastrophic forgetting when teaching an LLM
(GPT-2-small) a single, brand-new, arbitrary fact. Most of the early attempts
failed. They are kept here, honestly, rather than deleted.

## Failed: activation steering (synthetic and transplanted vectors)

First attempt: extract a real linear direction for the new fact from a
trained sparse autoencoder (`src/real_sae_features.py`, 99.82% variance
explained on real GPT-2 layer-6 activations), combine it with real
topology+budget weighting into one steering vector, and inject it additively
into the residual stream via a forward hook (standard activation-steering,
no gradient descent).

Every variant of this failed for a structural reason, not a tuning reason:
- A naive sum of ~500 real active features produced a wildly oversized vector
  (magnitude ~1900+ vs a real activation norm of ~140) that destroyed the
  model at every tested scale.
- A phase-based (Kuramoto) recombination fixed the magnitude blow-up (real,
  verified: the resulting direction was genuinely non-degenerate, not a
  disguised sum) but still never got the new fact into the model's top-5
  predictions at any safe scale.
- Transplanting a REAL, already-correct pattern from a fact the model already
  knows (instead of a synthetic vector) failed a basic fidelity check: adding
  that real pattern back into its OWN native context broke that context too.

**Conclusion reached**: additive activation steering into a single layer is
not a viable mechanism for writing a new fact into a model. It's a real-time
forward-pass perturbation, not a change to what the model has learned.

## First real success: gated gradient descent from a real in-context manifestation

Instead of computing or borrowing a vector, give the model the fact directly
in context (`"The secret code word is banana. The secret code word is"`),
confirm real (unmodified) in-context recall, extract the REAL SAE feature
activations from that real, correct forward pass, and use them (via the
topology+budget rule) to gate ONE real gradient-descent step's contribution
to the embedding and all 12 MLP layers.

Real result over 8 steps: the gated version achieved genuine bare-prompt
recall (no context needed) with ~16x less perplexity damage than standard
fine-tuning, and preserved 10/14 unrelated facts vs standard's 1/14. Not
perfect, but the first version of anything that actually worked.

## Dead end: searching for a "does the model already know this" boundary

An extended side-investigation tried to find a measurable signal, in real SAE
feature activations, that distinguishes a fact the model confidently knows
from one it does not — at the word level, the sentence level, and via the
actual structure of the pairwise feature-topology matrix (not just an
aggregate score), and finally via real Kuramoto phase-synchronization
dynamics on that same feature set.

**All of it failed**, including after catching and fixing a real methodology
bug along the way (a bare, un-prefixed single BPE token at sequence position
0 is a genuinely out-of-distribution input for GPT-2 and produces a
degenerate, near-identical activation regardless of which word it is — this
was found live and fixed, not glossed over). Even with that fixed, and with a
proper unrelated control ("purple elephant"), no measure tried — aggregate
cosine, matrix structure, phase order parameter — reliably separated known
facts from an unrelated control at layer 6. This line was abandoned in favor
of just using the real in-context manifestation directly (see above), per the
observation "мы уже видим активации" (we already have the real activations,
no need to abstractly classify them first).

## The actual result: resonance-weighted ROME

[ROME](https://github.com/kmeng01/rome) already solves "insert one fact,
minimize collateral damage" with a real, published closed-form edit — its
only free "what to protect" choice is a generic corpus covariance. Swapping
that for the same topology+budget resonance weighting (see `METHODS.md` and
the main `README.md`) gave a real, reproducible improvement over the
published baseline at 10 of 12 layers, confirmed on two independent
multi-fact samples via a self-calibrated selection rule (adapted from an
unrelated project's "K≥S operational closure" check, see `METHODS.md`).

## Tried and failed TWICE: multi-layer (MEMIT-style) decomposition

v1: splitting one fact's edit evenly (fixed 1/4 share per layer) across 4
layers made results WORSE on all 3 facts tested against a single-best-layer
edit. Diagnosed live as an incomplete adaptation: v1 never checked how much
of the target change earlier layers' edits had already produced by the time
the signal reaches the final layer — a real, identified gap, not a vague
"MEMIT is different" excuse.

v2: fixed that specific gap — after each layer's edit, re-measure the real
current output at the final layer and distribute only the REMAINING gap
among layers not yet edited. This made things SUBSTANTIALLY WORSE, not
better (one fact's perplexity damage went from +23 to +127). Diagnosis: both
versions assume each layer's edit contributes additively and independently
to the final output; that assumption breaks down once you account for how an
early layer's edit changes the hidden states that flow into the (still
unedited) later layers' own nonlinear processing — v2's reactive
re-estimation then compounds the mismatch onto a shrinking set of remaining
layers instead of correcting it.

Prompted to actually go read the real MEMIT paper instead of guessing again
("сначала изучи все что поможет картировать все что нам нужно по
MEMIT-алгоритмам"), the real formula turned out to be neither v1 nor v2:
`r_l = (z - h_L) / (L - l + 1)` — a growing share, computed once from the
clean model, with the LAST layer getting the FULL undivided deficit. Two more
variants were tried:

- **v3**: adapted the Рефлексия/autopoiesis `gate_ok` idiom to run DURING the
  decomposition — abort further layers as soon as real measured damage
  exceeds the single-layer baseline's own cost. This genuinely prevented the
  worst blow-up on one fact but stopped too early to learn the fact at all on
  another (a real safety/capability trade-off).
- **v4**: the actual literal MEMIT formula. Still worse than the single-layer
  baseline on all 3 facts — because the last layer gets the FULL deficit
  (same size as the single-layer edit) *plus* every earlier layer also gets a
  real, substantial edit on top. MEMIT's real design goal is making
  *thousands of simultaneous edits* numerically tractable, not minimizing
  damage from a *single* edit — applying it to one fact just adds redundant
  modification on top of what one well-placed edit already achieves.

None of the four multi-layer variants (v1/v2/v3/v4) improved on the single
best-layer edit for this project's actual goal (minimizing collateral damage
from one new fact). See `README.md` for the full numbers.

## The real fix: adapt the task to MEMIT, not MEMIT to the task

Pushed back on directly: *"что значит MEMIT не подходит для нашей задачи?
НЕТ! Это значит что мы не адаптировали НАШУ ЗАДАЧУ под MEMIT!"* — correct.
All four decomposition variants tested MEMIT's multi-layer machinery on a
SINGLE fact, which can never show MEMIT's real advantage (making MANY
simultaneous edits tractable). Re-tested properly: 6 new facts inserted
SIMULTANEOUSLY via MEMIT's real joint closed-form solve (one layer, no
spreading), standard corpus C vs resonance-weighted C.

A real bug surfaced immediately: the literal formula's regularization is
`(C_0 + K K^T)^-1`, not a fixed ridge — a ridge that was fine for one key
exploded the 6-key joint solve to `ppl ~1e19`. Fixed by using the real
formula's own `K K^T` term.

After the fix: standard corpus C gave +4.18 ppl / 11 facts kept; resonance C
gave +2.45 ppl / 12 facts kept — both learned all 6 new facts perfectly. A
genuine, reproducible win in MEMIT's actual intended regime, confirming the
user's correction was right: the earlier "MEMIT doesn't help here" framing
was really "we hadn't yet tested the right thing."

## A second bug, found by writing tests for a portfolio pass -- not by chasing results

Asked to make the repository "read like an architect's, not a tinkerer's,"
the obvious no-compute step was distilling the repeated inline math into a
real package with real unit tests. Writing a test for "total budget is
conserved" led to checking the transfer formula line by line: `cost = -topo`
before `relu`. Since `topo` is never negative (it's `relu(cos)`), `cost` was
never positive, so `relu(cost)` was **identically zero for every real input
this whole project ever fed it** -- the asymmetric weak-to-strong transfer,
one of the very first mechanics established in this whole session, had never
actually fired in a single script. `budget_final` was silently always equal
to raw `||v||`.

Fixed to `cost = +topo`. A new unit test confirms transfer is now genuinely
active and conservation still holds. Re-ran the two cheapest real checks
(6-fact joint edit, N=6..50 scaling curve) with the fix -- **both unchanged
to the 3rd decimal place**. This is the THIRD real formula bug caught in this
project (after a ROME-formula misread and a peer/fact-budget mix-up), and
the third time the headline conclusion survived a real correction
essentially untouched. COUNTERFACT has not yet been re-run with this fix.

## Literature check (real search, not from memory)

A live search found real, directly relevant prior work: ROME and MEMIT
themselves (the baseline used throughout), "Model Editing at Scale leads to
Gradual and Catastrophic Forgetting" (confirms even ROME/MEMIT suffer real
collateral damage at scale — consistent with what was found here), and
"Representation Shattering in Transformers" (a mechanistic account of why
editing causes collateral damage). A separate, active research direction —
similarity/cosine-gated gradient projection for continual learning (e.g. an
OGD extension called SFAO) — is conceptually the closest existing family to
the resonance-weighting idea here, though the specific sum-based
topology+budget combination was not found described anywhere.
