# Experiments

## Baseline
Started with simple modality + anatomy matching rules — if the prior shared the same imaging type and body region as the current study, mark it relevant. Got the structure right but precision was poor (~78% accuracy); too many irrelevant priors were being surfaced.

## What Worked
Switched to a learned model with 15 hand-crafted features per (current, prior) pair: token Jaccard similarity, exact description match, same modality/anatomy flags, contrast agreement, temporal buckets (within 30/180/365 days), and empirical base rates per modality. Fed these into a LogisticRegression + HistGradientBoosting ensemble and tuned the decision threshold on a held-out split. That combination pushed accuracy to **91.97%** on the full public eval (27,614 priors). Added a post-hoc penalty for cross-modality, cross-anatomy pairs which cut a handful of obvious false positives.

## What Failed
- `lbfgs` solver for logistic regression hit repeated numpy overflow during training and never converged — switched to `saga` which trained cleanly in under a minute.
- Running one `predict_proba` call per prior study was ~40× slower than batching the whole request into a single numpy call. Would have been a timeout risk on large evaluator payloads.

## How I Would Improve It
- Add a lightweight medical text encoder (e.g. BioBERT) to catch semantic matches that token overlap misses — "MRI HEAD" vs "MRI BRAIN" score low on Jaccard but are the same study.
- Expand the modality/anatomy keyword lexicons with more real-world RIS abbreviations to reduce unclassified "other" studies.
- Use cross-validated threshold tuning instead of a single validation split for a more stable operating point.
