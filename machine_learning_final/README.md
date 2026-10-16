# Final ROS classifier pipeline

## Run

```powershell
conda activate magicrhino
python run_final_pipeline.py            # ~12 minutes, deterministic
python verify_pipeline.py               # audit the saved outputs
```

`--clean` removes existing outputs first. The configuration lives in
`config/analysis_config.json` and is actually loaded by the script. 
Input data has 391 samples (138 ROS-positive / 253 negative, 2,428
features, identifier `Index`, target `Y`).

## Procedure

- Six families × five seeds (0, 42, 2024, 2025, 12345); per seed all families
  share the same stratified 5 outer folds.
- Each outer training partition gets a stratified 5-fold inner randomized
  search: 6 candidates from the original parameter distributions, refit on
  mean inner ROC-AUC. Median imputation, variance removal, ANOVA feature
  selection, and scaling are fitted inside every inner fold.
- The MCC-optimal decision threshold is chosen on the inner OOF probabilities
  of the selected candidate, using only outer-training data; it is locked
  before the outer test fold is scored. No pooled outer threshold is ever
  optimized.
- Top 10 of the 30 model/seed combinations (ranked by pooled outer OOF ROC-AUC)
  are refit on all 391 samples by repeating the identical selection procedure
  on the full data. The deployment artifact is the RandomForest/seed-0
  combination (rank 3 overall, chosen for family-level stability, not for its
  individual score). The deployment fit is always exported, even if its
  combination falls outside the configured top-N ranking (`rank_among_top10`
  is then null in its sidecar).

## Family results (nested internal validation)

| Family | Outer OOF ROC-AUC | Outer MCC |
|---|---:|---:|
| RandomForest | 0.739 ± 0.013 | 0.389 ± 0.023 |
| XGBoost | 0.737 ± 0.019 | 0.327 ± 0.043 |
| KNearestNeighbors | 0.729 ± 0.020 | 0.336 ± 0.037 |
| LightGBM | 0.719 ± 0.016 | 0.304 ± 0.026 |
| GaussianNB | 0.709 ± 0.016 | 0.323 ± 0.017 |
| LogisticRegression | 0.699 ± 0.014 | 0.287 ± 0.016 |

## Outputs

| File | Meaning |
|---|---|
| `outputs/tables/model_comparison_summary.csv` | Family mean ± SD across seeds |
| `outputs/tables/model_comparison_by_seed.csv` | Pooled outer OOF metrics per model/seed |
| `outputs/tables/outer_oof_predictions.csv` | 11,730 outer predictions with per-fold thresholds |
| `outputs/tables/inner_threshold_predictions.csv` | Inner OOF probabilities used for each fold's threshold |
| `outputs/tables/outer_fold_metrics.csv` | Per-fold metrics, parameters, thresholds |
| `outputs/tables/inner_search_candidates.csv` | All 900 inner search candidates |
| `outputs/tables/random_forest_seed0_outer_oof_predictions.csv` | Deployment combination's 391 OOF predictions |
| `outputs/tables/random_forest_seed0_roc_curve.csv` | ROC curve plotting data |
| `outputs/tables/random_forest_seed0_pr_curve.csv` | Precision-Recall curve plotting data |
| `outputs/tables/random_forest_seed0_threshold_metrics.csv` | Threshold sweep; nearest sweep point to each locked fold threshold flagged |
| `outputs/tables/analysis_report.json` | Configuration, hashes, versions, results, deployment details |
| `outputs/models/random_forest_seed0.joblib` (+ `.json`) | Deployment model and its metadata |
| `outputs/models/top10_manifest.json` | All top-10 artifacts |
| `outputs/models/top10/rankNN_*` | Top-10 full-data models with sidecars |

## Using the deployment model

```python
import json, joblib, sys
sys.path.insert(0, r"machine_learning_final")  # for the XGBoost tags adapter, if loading top-10 XGBoost artifacts
model = joblib.load(r"machine_learning_final/outputs/models/random_forest_seed0.joblib")
threshold = json.load(open(r"machine_learning_final/outputs/models/random_forest_seed0.json"))["threshold"]
probability = model.predict_proba(new_features)[:, 1]   # same 2,428 columns, same names/order
prediction = (probability >= threshold).astype(int)
```

Do not use `model.predict()`: it applies sklearn's default 0.5 cutoff instead
of the saved MCC-optimal threshold (0.4787). The RandomForest artifact loads
with sklearn alone; the top-10 XGBoost artifacts additionally require
`run_final_pipeline` to be importable (local tags adapter for XGBoost 2.1.2 /
scikit-learn 1.9, fixed upstream in XGBoost 2.1.4).