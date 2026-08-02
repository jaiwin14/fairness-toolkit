## Bias mitigation doesn't work the way I assumed — here's what a week of benchmarking taught me

Most fairness-in-ML writeups pick one dataset, apply one mitigation technique, and call it a day. I wanted to know something more useful: if a bias mitigation technique looks good on one dataset, can you trust it on another?

So I rebuilt a set of exploratory notebooks (originally built around the ProPublica COMPAS recidivism dataset) into a proper benchmarking toolkit — trained 4 classifiers (Logistic Regression, SVM, Gradient Boosting, XGBoost) across 3 well-known fairness datasets (COMPAS, UCI Adult Income, German Credit), and tested 4 bias mitigation techniques from IBM's AIF360 library against each one: Reject Option Classification, Equalized Odds, Calibrated Equalized Odds, and Adversarial Debiasing.

**The finding that surprised me:** across all 12 dataset × model combinations tested, `equalized_odds` post-processing never once made fairness worse. `calibrated_equalized_odds` — which looked perfectly reasonable on the Adult Income dataset — actively worsened fairness in 9 out of 12 cases elsewhere. It's not a broken technique; it optimizes a genuinely different fairness criterion (per-group score calibration) than the metric I was measuring against, which is exactly the kind of real-world "fairness definitions can conflict with each other" tension the fairness ML literature warns about (Kleinberg/Chouldechova impossibility results) — but it only becomes visible once you test across more than one dataset.

I also found that a technique's usefulness scales with how biased the model already was to begin with — mitigation on Adult Income (severe baseline bias, ~3x disparate impact) showed huge improvements, while the same technique on German Credit (much smaller baseline bias) sometimes made things worse by over-correcting.

Beyond the fairness findings, this was also a real software engineering exercise: generalizing the pipeline so the exact same code correctly handles two datasets with *opposite* favorable-outcome conventions (COMPAS treats "did not reoffend" as favorable=0; Adult and German treat the positive outcome as favorable=1) turned up a genuine scores-direction bug in my own mitigation code that I caught by benchmarking, not by code review.

Full repo, all results reproducible from a clean install, plus a CLI (`fairkit run`, `fairkit benchmark`): [github.com/jaiwin14/fairness-toolkit](https://github.com/jaiwin14/fairness-toolkit)

#MachineLearning #ResponsibleAI #FairnessInML #DataScience
