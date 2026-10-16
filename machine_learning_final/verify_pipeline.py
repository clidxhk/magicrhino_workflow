"""Audit the final pipeline outputs: split isolation, threshold replay, metric replay,
ranking consistency, and artifact loading."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from run_final_pipeline import ROOT, choose_threshold, deployment_stem, load_data, metrics

TABLES = ROOT / "outputs/tables"
MODELS = ROOT / "outputs/models"


def replay_integrity(config, frame, x, y):
    outer = pd.read_csv(TABLES / "outer_oof_predictions.csv")
    inner = pd.read_csv(TABLES / "inner_threshold_predictions.csv")
    folds = pd.read_csv(TABLES / "outer_fold_metrics.csv")
    assert not outer.duplicated(["seed", "model", "sample_position"]).any()
    checked_folds = 0
    for seed in config["comparison_seeds"]:
        splits = list(StratifiedKFold(n_splits=config["outer_folds"], shuffle=True,
                                      random_state=seed).split(x, y))
        for name in config["models"]:
            group = outer[(outer.seed == seed) & (outer.model == name)]
            assert set(group.sample_position) == set(range(len(frame))), (seed, name)
            for fold, (_, test) in enumerate(splits, 1):
                rows = group[group.outer_fold == fold]
                assert set(rows.sample_position) == set(test.tolist()), (seed, name, fold)
                assert (rows.outer_prediction.to_numpy()
                        == (rows.outer_probability.to_numpy() >= rows.threshold.iloc[0]).astype(int)).all()
                development = inner[(inner.seed == seed) & (inner.model == name)
                                    & (inner.outer_fold == fold)]
                assert set(development.sample_position) == set(range(len(frame))) - set(test)
                replayed = choose_threshold(development.Y.to_numpy(),
                                            development.inner_probability.to_numpy())
                assert np.isclose(replayed, rows.threshold.iloc[0]), (seed, name, fold)
                fold_row = folds[(folds.seed == seed) & (folds.model == name)
                                 & (folds.outer_fold == fold)].iloc[0]
                assert np.isclose(fold_row.threshold, rows.threshold.iloc[0])
                checked_folds += 1
    return outer, checked_folds


def replay_scores(outer, frame):
    by_seed = pd.read_csv(TABLES / "model_comparison_by_seed.csv")
    summary = pd.read_csv(TABLES / "model_comparison_summary.csv")
    assert len(by_seed) == len(config["models"]) * len(config["comparison_seeds"])
    for _, row in by_seed.iterrows():
        group = outer[(outer.seed == row.seed) & (outer.model == row.model)]
        recomputed = metrics(group.Y, group.outer_probability, group.outer_prediction)
        for name, value in recomputed.items():
            assert np.isclose(value, row[name]), (row.model, row.seed, name)
    for _, row in summary.iterrows():
        family = by_seed[by_seed.model == row.model]
        assert np.isclose(family.roc_auc.mean(), row.mean_outer_roc_auc)
        assert np.isclose(family.roc_auc.std(), row.sd_outer_roc_auc)
        assert np.isclose(family.mcc.mean(), row.mean_outer_mcc)
    return by_seed, summary


def check_ranking_and_artifacts(by_seed, config, frame, x, y):
    manifest = json.loads((MODELS / "top10_manifest.json").read_text(encoding="utf-8"))
    expected = by_seed.sort_values("roc_auc", ascending=False).head(config["n_top_models"])
    assert [(r["model"], r["seed"]) for r in manifest["models"]] == \
           list(zip(expected.model, expected.seed))
    for record in manifest["models"]:
        sidecar = json.loads((MODELS / record["file"]).with_suffix(".json").read_text(encoding="utf-8"))
        assert sidecar["model"] == record["model"] and sidecar["seed"] == record["seed"]
        assert len(sidecar["selected_features"]) > 0
        joblib.load(MODELS / record["file"])  # Must load in this environment.
    stem = deployment_stem(config)
    deployment = json.loads((MODELS / f"{stem}.json").read_text(encoding="utf-8"))
    assert deployment["family"] == config["deployment_model"]
    model = joblib.load(MODELS / f"{stem}.joblib")
    probability = model.predict_proba(x)[:, 1]
    assert probability.shape == (len(frame),)
    assert 0.0 < deployment["threshold"] < 1.0
    assert len(deployment["input_columns"]) == x.shape[1]
    report = json.loads((TABLES / "analysis_report.json").read_text(encoding="utf-8"))
    assert report["deployment_model"]["threshold"] == deployment["threshold"]
    assert set(report["model_family_summary"]) == set(config["models"])
    rf_curve = pd.read_csv(TABLES / f"{stem}_roc_curve.csv")
    rf_oof = pd.read_csv(TABLES / f"{stem}_outer_oof_predictions.csv")
    deploy_rows = by_seed[(by_seed.model == config["deployment_model"])
                          & (by_seed.seed == config["deployment_seed"])]
    assert np.isclose(roc_auc_score(rf_oof.Y, rf_oof.outer_probability),
                      float(deploy_rows.roc_auc.iloc[0]))
    sweep = pd.read_csv(TABLES / f"{stem}_threshold_metrics.csv")
    fold_thresholds = rf_oof.threshold.unique()
    assert sweep.is_nearest_to_locked_fold_threshold.sum() >= len(fold_thresholds) - 1  # two locked thresholds may share a nearest sweep point
    flagged = sweep[sweep.is_nearest_to_locked_fold_threshold].threshold.to_numpy()
    assert all(min(abs(flagged - locked)) < 0.01 for locked in fold_thresholds)
    assert len(rf_curve) >= 2 and {"false_positive_rate", "true_positive_rate"} < set(rf_curve.columns)
    return report


if __name__ == "__main__":
    config = json.loads((ROOT / "config/analysis_config.json").read_text(encoding="utf-8"))
    _, frame, x, y = load_data(config)
    outer, checked_folds = replay_integrity(config, frame, x, y)
    print(f"PASS: {checked_folds} outer folds; split isolation, threshold replay, decision replay")
    by_seed, _ = replay_scores(outer, frame)
    print("PASS: pooled by-seed metrics and family summary replayed from saved predictions")
    report = check_ranking_and_artifacts(by_seed, config, frame, x, y)
    print("PASS: top-10 ranking, sidecars, deployment artifact, and analysis report consistent")
