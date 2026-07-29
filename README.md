# fairkit

A cross-dataset fairness benchmarking and bias-mitigation toolkit, refactored
from three Colab notebooks originally built around the ProPublica COMPAS
dataset.

> Work in progress — this README is a placeholder created on Day 1 of the
> refactor. It will be filled in with real results (Day 3), the CLI
> quickstart (Day 6), and the full writeup (Day 7).

## Repo structure
```
fairness-toolkit/
├── src/fairkit/        # library code
│   ├── datasets/       # per-dataset loaders (compas, adult, german)
│   ├── train.py        # model training
│   ├── mitigate.py     # bias mitigation wrappers
│   └── evaluate.py     # fairness + accuracy evaluation
├── tests/              # pytest suite
├── notebooks/archive/  # original exploratory notebooks (kept for reference)
├── results/            # benchmark output (csv/png)
├── cli.py              # command-line entry point
├── requirements.txt
└── pyproject.toml
```

## Status
- [x] Day 1 — repo skeleton, environment, `load_compas()` loader + tests
- [x] Day 2 — `train.py` / `evaluate.py` / `mitigate.py`
- [ ] Day 3 — full COMPAS benchmark + results table
- [ ] Day 4 — Adult Income dataset
- [ ] Day 5 — German Credit dataset + cross-dataset comparison
- [ ] Day 6 — CLI packaging
- [ ] Day 7 — final polish, docs, publish
