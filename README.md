# fairkit

A cross-dataset bias-mitigation benchmarking toolkit: trains classifiers
on three well-known fairness datasets (COMPAS, Adult Income, German
Credit), measures how biased they are against protected groups, applies
four standard mitigation techniques from AIF360, and honestly reports
which techniques actually work — and which don't, despite looking fine on
a single dataset.

Originally three exploratory Colab notebooks built around the ProPublica
COMPAS recidivism dataset (`notebooks/archive/`). Rebuilt into a tested, installable Python package with a CLI.

## The headline finding

Tested across 3 datasets × 4 models × 3 post-processing mitigation
techniques (36 combinations total): **`equalized_odds` never made fairness
worse — 0/36. `calibrated_equalized_odds` made it worse in 9/12 dataset×model
cases where it was compared head-to-head.** A technique that looks like a
safe default on one dataset can quietly backfire on another. Full analysis
in [Cross-dataset comparison](#cross-dataset-comparison) below.

## Quickstart (CLI)

```bash
git clone https://github.com/jaiwin14/fairness-toolkit.git
cd fairness-toolkit
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Requires Python 3.10–3.13 (TensorFlow, a dependency for one mitigation
technique, doesn't yet support 3.14).

```bash
$ fairkit list-datasets
  adult     target=income
  compas    target=two_year_recid
  german    target=credit_risk

$ fairkit run --dataset compas --model xgboost --mitigation eqodds
Running xgboost on compas (mitigation: equalized_odds)...

  accuracy                      = 0.6528
  demographic_parity_difference = 0.0335
  equalized_odds_difference     = 0.0105
  disparate_impact_ratio        = 0.9519  (1.0 = fair)

$ fairkit benchmark --dataset compas
=== dataset: compas ===
  [logreg  ] baseline           acc=0.663
  ...
Saved 17 rows to results/compas_cli_results.csv

$ fairkit benchmark --dataset all --output results/   # all 3 datasets
```

`--mitigation` accepts a full technique name or a short alias: `roc`,
`eqodds`, `ceo`, `adv`, or `none`. Run `fairkit --help`,
`fairkit run --help`, or `fairkit benchmark --help` for full options.

```bash
pytest tests/ -v   # 45 tests
```

## Methodology

**Datasets** (all binary classification, one or more protected attributes):
| Dataset | Task | Protected attribute(s) used | Rows |
|---|---|---|---|
| [COMPAS](https://github.com/propublica/compas-analysis) | predict recidivism | `sex`, `race` | 7,214 |
| [Adult Income](https://archive.ics.uci.edu/dataset/2/adult) | predict income >$50K | `sex`, `race` | 30,162 |
| [German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) | predict credit risk | `sex`, `age` | 1,000 |

**Models**: Logistic Regression, SVM, Gradient Boosting, XGBoost — all
scikit-learn-compatible, trained identically across all three datasets via
one shared `train_models()` function.

**Fairness metrics** ([`fairlearn`](https://fairlearn.org/)): demographic
parity difference, equalized odds difference, disparate impact ratio
(1.0 = fair).

**Mitigation techniques** ([AIF360](https://aif360.res.ibm.com/)):
- *Post-processing* (adjust an already-fitted model's predictions):
  Reject Option Classification, Equalized Odds, Calibrated Equalized Odds
- *In-processing* (trains its own model with fairness built into
  training): Adversarial Debiasing

**Design principle**: the training/evaluation/mitigation code has zero
dataset-specific branching. Each dataset only supplies a loader and its
own `favorable_label`/`privileged_value` convention (e.g. COMPAS treats
label 0 as favorable; Adult and German treat label 1 as favorable — the
same mitigation functions handle both correctly). See `src/fairkit/registry.py`.

## Cross-dataset comparison

Full table: [`results/cross_dataset_comparison.csv`](results/cross_dataset_comparison.csv) · Chart: [`results/cross_dataset_comparison.png`](results/cross_dataset_comparison.png)

The question this was built to answer: **does bias mitigation help
consistently, or is it dataset-dependent?** Counted across all 12
(dataset × model) combinations, by how often each technique made the
disparate impact ratio *worse* instead of better:

| Mitigation | Times fairness got worse | Mean improvement (Adult / COMPAS / German) |
|---|---|---|
| `equalized_odds` | **0 / 12** | 1.093 / 0.229 / 0.058 |
| `reject_option_classification` | 1 / 12 | 1.513 / 0.221 / 0.035 |
| `calibrated_equalized_odds` | **9 / 12** | 0.476 / -0.140 / -0.218 |

**Findings:**
- **`equalized_odds` is the most reliable technique tested** — never made
  fairness worse on any dataset or model. `reject_option_classification`
  is close behind (one regression, German Credit + xgboost).
- **`calibrated_equalized_odds` is dataset-dependent in a genuinely bad
  way** — improved fairness on Adult but *worsened* it on COMPAS and
  German Credit in most cases. It optimizes a different fairness criterion
  (per-group score calibration under a false-negative-rate cost) than the
  metrics in this table measure — a real instance of the
  Kleinberg/Chouldechova impossibility result that satisfying one fairness
  definition can conflict with another. Don't reach for it by default;
  check what it actually optimizes for your use case first.
- **Mitigation benefit scales with how biased the baseline already was**,
  not with the technique alone. Adult's baseline bias was severe (DI ratio
  routinely 2–3), so mitigation shows huge absolute improvements there.
  German Credit's baseline bias was often already small (DI ratio close to
  1.0), leaving little genuine signal to correct — which is also why
  `calibrated_equalized_odds` does the most damage there (over-correcting
  where there wasn't much to fix). **A technique's usefulness can't be
  judged in isolation from how biased the starting model was.**

## Results by dataset

<details>
<summary><b>COMPAS</b> — baseline DP diff 0.14–0.33, DI ratio 0.63–0.82 (click to expand full table)</summary>

Full results: [`results/compas_results.csv`](results/compas_results.csv) · Chart: [`results/compas_tradeoff.png`](results/compas_tradeoff.png)

Equalized Odds and Reject Option Classification substantially reduce bias
(DP/EO diff down to ~0.01–0.05) for a modest accuracy cost — Reject Option
Classification sometimes matches or slightly *beats* baseline accuracy
while still improving fairness. Adversarial Debiasing gets the best of
both worlds: accuracy comparable to or better than baseline (0.67–0.69)
with low DP/EO diff (~0.03–0.04).

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

</details>

<details>
<summary><b>Adult Income</b> — baseline DI ratio ~3.0 (strong gender/race income gap) (click to expand full table)</summary>

Full results: [`results/adult_results.csv`](results/adult_results.csv)

Baseline models are much more accurate (0.80–0.86) than on COMPAS, but
show *stronger* bias by disparate impact: the privileged group (male /
White) gets the favorable prediction at roughly **3x** the rate of the
unprivileged group before mitigation — consistent with the well-documented
income gap in this dataset. `equalized_odds` and
`reject_option_classification` bring DI ratio down to ~1.1–1.8 for a
modest accuracy cost (2–7 points). `svc` starts far less biased than the
other three models (DI ratio 1.88 at baseline) — model-dependent, not just
dataset-dependent.

| Model | Mitigation | Accuracy | DP diff | EO diff | DI ratio |
|---|---|---|---|---|---|
| logreg | none | 0.848 | 0.176 | 0.11 | 3.196 |
| logreg | reject_option_classification | 0.781 | 0.048 | 0.161 | 1.144 |
| logreg | equalized_odds | 0.822 | 0.087 | 0.003 | 1.64 |
| logreg | calibrated_equalized_odds | 0.824 | 0.101 | 0.068 | 2.255 |
| svc | none | 0.797 | 0.048 | 0.021 | 1.878 |
| svc | reject_option_classification | 0.798 | 0.044 | 0.036 | 1.689 |
| svc | equalized_odds | 0.793 | 0.045 | 0.013 | 1.779 |
| svc | calibrated_equalized_odds | 0.797 | 0.054 | 0.01 | 2.113 |
| gbc | none | 0.858 | 0.169 | 0.085 | 3.121 |
| gbc | reject_option_classification | 0.81 | 0.04 | 0.161 | 1.128 |
| gbc | equalized_odds | 0.831 | 0.093 | 0.011 | 1.703 |
| gbc | calibrated_equalized_odds | 0.839 | 0.116 | 0.045 | 2.448 |
| xgboost | none | 0.864 | 0.183 | 0.074 | 2.968 |
| xgboost | reject_option_classification | 0.825 | 0.043 | 0.169 | 1.149 |
| xgboost | equalized_odds | 0.841 | 0.105 | 0.001 | 1.669 |
| xgboost | calibrated_equalized_odds | 0.847 | 0.134 | 0.057 | 2.442 |
| adversarial_debiasing_nn | adversarial_debiasing | 0.829 | 0.048 | 0.174 | 1.327 |

</details>

<details>
<summary><b>German Credit</b> — smaller dataset, less baseline bias, mitigation can backfire (click to expand full table)</summary>

Full results: [`results/german_results.csv`](results/german_results.csv)

Much smaller dataset (1,000 rows) with noticeably less baseline bias than
COMPAS or Adult for some models — `svc` starts with disparate impact ratio
already close to 1.0 (0.99), while `logreg` and `gbc` show more (DI ratio
~1.14 each, `xgboost` ~1.03). Useful contrast case: when baseline bias is
already small, post-processing mitigation has little genuine signal to
correct and can do more harm than good (see Cross-dataset comparison above).

| Model | Mitigation | Accuracy | DP diff | EO diff | DI ratio |
|---|---|---|---|---|---|
| logreg | none | 0.695 | 0.093 | 0.13 | 1.143 |
| logreg | reject_option_classification | 0.655 | 0.045 | 0.1 | 0.903 |
| logreg | equalized_odds | 0.605 | 0.007 | 0.05 | 1.011 |
| logreg | calibrated_equalized_odds | 0.685 | 0.35 | 0.45 | 1.538 |
| svc | none | 0.71 | 0.01 | 0.03 | 0.99 |
| svc | reject_option_classification | 0.735 | 0.002 | 0.05 | 1.003 |
| svc | equalized_odds | 0.7 | 0.01 | 0.025 | 0.99 |
| svc | calibrated_equalized_odds | 0.72 | 0.024 | 0.125 | 1.026 |
| gbc | none | 0.76 | 0.09 | 0.075 | 1.136 |
| gbc | reject_option_classification | 0.775 | 0.007 | 0.05 | 0.99 |
| gbc | equalized_odds | 0.745 | 0.031 | 0.05 | 1.045 |
| gbc | calibrated_equalized_odds | 0.72 | 0.333 | 0.6 | 1.5 |
| xgboost | none | 0.745 | 0.021 | 0.025 | 1.029 |
| xgboost | reject_option_classification | 0.735 | 0.045 | 0.1 | 0.932 |
| xgboost | equalized_odds | 0.74 | 0.014 | 0.025 | 1.019 |
| xgboost | calibrated_equalized_odds | 0.725 | 0.093 | 0.15 | 1.124 |
| adversarial_debiasing_nn | adversarial_debiasing | 0.7 | 0.0 | 0.0 | 1.0 |

</details>

## Repo structure

```
fairness-toolkit/
├── src/fairkit/
│   ├── datasets/        # compas.py, adult.py, german.py loaders
│   ├── train.py         # train_models(): LogReg/SVC/GBC/XGBoost
│   ├── evaluate.py      # evaluate_model(): accuracy + fairness metrics
│   ├── mitigate.py      # 4 AIF360 mitigation techniques, dataset-agnostic
│   ├── registry.py      # per-dataset conventions (favorable_label, etc.)
│   ├── benchmark.py     # dataset-agnostic benchmark runner (CLI's engine)
│   └── cli.py           # `fairkit` command implementation
├── tests/                # 45 pytest tests
├── scripts/
│   ├── verify_day{1..6}.py       # one checkpoint script per build day
│   ├── run_{compas,adult,german}_benchmark.py
│   ├── plot_compas_tradeoff.py
│   └── compare_benchmarks.py     # cross-dataset comparison
├── notebooks/archive/    # original 3 exploratory notebooks (untouched)
├── data/                 # raw dataset files + provenance (data/README.md)
├── results/               # all benchmark CSVs, charts
├── cli.py                # thin shim -> fairkit.cli (also runnable directly)
├── requirements.txt
└── pyproject.toml
```

## Reproducibility

Every claim above is backed by a script you can re-run yourself — nothing
here is hand-typed. Each build day has its own checkpoint script:

```bash
pytest tests/ -v                # 45 tests
python scripts/verify_day1.py   # ... through verify_day6.py
```

All results were regenerated from a completely clean install (fresh venv,
`pip install -r requirements.txt`, `pip install -e .`) before being copied
into this README, specifically to catch packaging or environment issues a
"works on my machine" result would miss.

## Data sources & citation

- **COMPAS**: ProPublica's [compas-analysis](https://github.com/propublica/compas-analysis) repository.
- **Adult Income**: Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20
- **German Credit**: Hofmann, H. (1994). *Statlog (German Credit Data)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77

Both UCI datasets are CC BY 4.0. See `data/README.md` for exact file
provenance (both were fetched from public GitHub mirrors rather than UCI's
archive directly — see that file for why).

## License

MIT — see [LICENSE](LICENSE).
