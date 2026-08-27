# Attribution (how to use this work)

**License:** [Apache License 2.0](LICENSE) — free to use, modify, fork, and ship. **No payment required.**

This file explains the **attribution expectation** the author asks for. Apache-2.0 already requires keeping copyright / NOTICE on redistribution. Below is how we interpret “small task” vs “inside a larger system”.

## Small / personal / research snippets

If you:

- run the scripts,
- copy a function into a notebook,
- experiment on your laptop,
- cite us in a paper optionally,

then **Apache-2.0 alone is enough**: keep `LICENSE` + `NOTICE` when you redistribute source, and edit freely.

## Substantial systems / products / forks used as a method

If you incorporate this **resonance-weighted covariance** (topology + asymmetric budget transfer as the ROME/MEMIT `C` term) into a larger system, please:

1. Keep `LICENSE` and this repository’s `NOTICE` text in your distribution (Apache §4).
2. State clearly that the method / module comes from:
   - **Repository:** https://github.com/andreyivan4enkov/resonance_rome
   - **Method name:** resonance-weighted covariance for ROME/MEMIT
   - **Author:** Andrey Ivanchenkov (andreyivan4enkov)
3. Prefer a short credit line in docs, for example:

> Resonance-weighted covariance based on
> [resonance_rome](https://github.com/andreyivan4enkov/resonance_rome)
> (Apache-2.0; andreyivan4enkov).

4. Academic citation: use [CITATION.cff](CITATION.cff).

You may still charge for **your** product or support. The author does **not** charge a license fee for this code. Attribution is about **origin of the method**, not about money.

## Forks

Forks on GitHub already show lineage. If you publish a fork as a standalone product, keep NOTICE and mention this upstream (or “based on resonance_rome”).

## What you must not do

- Claim you invented ROME or MEMIT (Meng et al., 2022). This repo only replaces the covariance term.
- Strip NOTICE while shipping a substantial derivative.
- Redistribute **model weights** under this license (weights stay under Hugging Face / GPT-2 terms).
- Present a deleted-failure story as the full result. Negatives live in `docs/JOURNEY.md`.

## Contact

Issues on the GitHub repository.
