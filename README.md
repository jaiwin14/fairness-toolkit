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

## Results: COMPAS

Full results: [`results/compas_results.csv`](results/compas_results.csv) · Chart: [`results/compas_tradeoff.png`](results/compas_tradeoff.png)

Baseline models show real, measurable bias against the unprivileged group
(sex_Male=0) before any mitigation — demographic parity difference ranges
0.14–0.33 and disparate impact ratio ranges 0.63–0.82 (1.0 = fair) across
models and targets. Reproducible via `scripts/run_compas_benchmark.py`.

**Findings:**
- **Equalized Odds** and **Reject Option Classification** both substantially
  reduce demographic parity / equalized odds difference (down to ~0.01–0.05,
  DI ratio pushed to ~0.91–0.99) for a modest accuracy cost — and Reject
  Option Classification sometimes matches or slightly *beats* baseline
  accuracy while still improving fairness.
- **Calibrated Equalized Odds** keeps accuracy close to baseline (~0.65) but
  actually *increases* demographic parity / equalized odds difference in
  this benchmark. This isn't a bug — it optimizes a different fairness
  criterion (per-group score calibration under a false-negative-rate cost)
  than the metrics in this table measure, a known tension in fairness ML:
  satisfying one fairness definition can conflict with another
  (Kleinberg/Chouldechova impossibility results). Worth knowing before
  picking a mitigation technique based on a single metric.
- **Adversarial Debiasing** (in-processing) gets the best of both worlds
  here — accuracy comparable to or better than baseline (0.67–0.69) with
  low demographic parity / equalized odds difference (~0.03–0.04).

| Target | Model | Mitigation | Accuracy | DP diff | EO diff | DI ratio |
|---|---|---|---|---|---|---|
| is_recid | logreg | none | 0.671 | 0.258 | 0.237 | 0.66 |
| is_recid | logreg | reject_option_classification | 0.671 | 0.017 | 0.084 | 1.038 |
| is_recid | logreg | equalized_odds | 0.633 | 0.033 | 0.004 | 0.949 |
| is_recid | logreg | calibrated_equalized_odds | 0.653 | 0.498 | 0.66 | 0.502 |
| is_recid | svc | none | 0.668 | 0.154 | 0.141 | 0.764 |
| is_recid | svc | reject_option_classification | 0.686 | 0.038 | 0.011 | 0.942 |
| is_recid | svc | equalized_odds | 0.655 | 0.038 | 0.005 | 0.931 |
| is_recid | svc | calibrated_equalized_odds | 0.649 | 0.501 | 0.659 | 0.499 |
| is_recid | gbc | none | 0.676 | 0.213 | 0.215 | 0.7 |
| is_recid | gbc | reject_option_classification | 0.684 | 0.05 | 0.04 | 0.921 |
| is_recid | gbc | equalized_odds | 0.658 | 0.038 | 0.003 | 0.931 |
| is_recid | gbc | calibrated_equalized_odds | 0.649 | 0.502 | 0.66 | 0.498 |
| is_recid | xgboost | none | 0.67 | 0.165 | 0.149 | 0.77 |
| is_recid | xgboost | reject_option_classification | 0.668 | 0.023 | 0.032 | 0.961 |
| is_recid | xgboost | equalized_odds | 0.651 | 0.036 | 0.005 | 0.94 |
| is_recid | xgboost | calibrated_equalized_odds | 0.648 | 0.447 | 0.604 | 0.553 |
| is_recid | adversarial_debiasing_nn | adversarial_debiasing | 0.689 | 0.041 | 0.027 | 0.936 |
| two_year_recid | logreg | none | 0.663 | 0.331 | 0.376 | 0.63 |
| two_year_recid | logreg | reject_option_classification | 0.663 | 0.045 | 0.012 | 0.91 |
| two_year_recid | logreg | equalized_odds | 0.6 | 0.022 | 0.01 | 0.972 |
| two_year_recid | logreg | calibrated_equalized_odds | 0.652 | 0.438 | 0.597 | 0.562 |
| two_year_recid | svc | none | 0.679 | 0.213 | 0.234 | 0.748 |
| two_year_recid | svc | reject_option_classification | 0.683 | 0.048 | 0.026 | 0.919 |
| two_year_recid | svc | equalized_odds | 0.638 | 0.028 | 0.043 | 0.964 |
| two_year_recid | svc | calibrated_equalized_odds | 0.666 | 0.375 | 0.549 | 0.625 |
| two_year_recid | gbc | none | 0.685 | 0.214 | 0.19 | 0.723 |
| two_year_recid | gbc | reject_option_classification | 0.688 | 0.011 | 0.032 | 0.982 |
| two_year_recid | gbc | equalized_odds | 0.653 | 0.035 | 0.007 | 0.949 |
| two_year_recid | gbc | calibrated_equalized_odds | 0.669 | 0.441 | 0.623 | 0.559 |
| two_year_recid | xgboost | none | 0.674 | 0.135 | 0.111 | 0.82 |
| two_year_recid | xgboost | reject_option_classification | 0.671 | 0.004 | 0.036 | 0.992 |
| two_year_recid | xgboost | equalized_odds | 0.653 | 0.034 | 0.011 | 0.952 |
| two_year_recid | xgboost | calibrated_equalized_odds | 0.66 | 0.385 | 0.553 | 0.615 |
| two_year_recid | adversarial_debiasing_nn | adversarial_debiasing | 0.673 | 0.043 | 0.012 | 0.94 |

## Status
- [x] Day 1 — repo skeleton, environment, `load_compas()` loader + tests
- [x] Day 2 — `train.py` / `evaluate.py` / `mitigate.py`
- [x] Day 3 — full COMPAS benchmark + results table + tradeoff chart
- [ ] Day 4 — Adult Income dataset
- [ ] Day 5 — German Credit dataset + cross-dataset comparison
- [ ] Day 6 — CLI packaging
- [ ] Day 7 — final polish, docs, publish
