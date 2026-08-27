# Contributing

Thanks for considering a contribution. This project follows the same GitHub /
open-ML packaging as [moe-orbit-prefetch](https://github.com/andreyivan4enkov/moe-orbit-prefetch).

## Scope

1. Keep the **resonance-weighted covariance** substitution into real ROME/MEMIT.
   Do not replace it with cosine-as-the-method, standard attention, or a
   renamed classic baseline.
2. Do not commit model weights, secrets, Hugging Face caches, or `*.pt`.
3. If you change equations, update [docs/METHODS.md](docs/METHODS.md) in the same PR.
4. New **claims** need a lab artifact under `results/` — no invented PASS.
5. Prefer **ours → classic** order in benches. If ours is broken, stop; do not
   run the classic baseline to “still show a table”.
6. Keep honest negatives. Failed variants stay in `docs/JOURNEY.md` and
   `results/`; do not delete them to clean the story.
7. Architecture-specific work lives on a named branch (`gpt2-small` now).
   Do not dump a different model family onto this branch.

## Dev setup

```bash
pip install -r requirements.txt
python src/rome_resonance_edit.py
```

Weights download via `transformers` on first run (not stored in git).

## License

Contributions are under **Apache-2.0**. Keep `LICENSE` / `NOTICE`. Substantial
products: see [ATTRIBUTION.md](ATTRIBUTION.md).

## Conduct

[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
