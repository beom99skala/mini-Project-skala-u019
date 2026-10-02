"""Day 2 ESS cycle-life regression, using only information from cycles 1-100.

Usage:
    python train.py --data-dir /path/to/mini-PJ-dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


FILES = {
    "Batch 1": "2017-05-12_batch1.mat",
    "Batch 2": "2018-02-20_batch2.mat",
    "Batch 3": "2018-04-12_batch3.mat",
}

FEATURE_SETS = {
    "core": ["log10_delta_q_var", "delta_q_min_ah", "qd_change_ah", "ir_mean_ohm"],
    "core_plus_capacity_slope": [
        "log10_delta_q_var", "delta_q_min_ah", "qd_change_ah", "ir_mean_ohm",
        "qd_slope_2_100_ah_per_cycle", "qd_slope_91_100_ah_per_cycle",
    ],
    "core_plus_delta_shape": [
        "log10_delta_q_var", "delta_q_min_ah", "qd_change_ah", "ir_mean_ohm",
        "delta_q_mean_ah", "delta_q_skew", "log10_abs_delta_q_min",
    ],
    "paper_inspired_combined": [
        "log10_delta_q_var", "delta_q_min_ah", "qd_change_ah", "ir_mean_ohm",
        "qd_slope_2_100_ah_per_cycle", "qd_slope_91_100_ah_per_cycle",
        "qd_cycle_2_ah", "qd_max_minus_cycle_2_ah",
        "delta_q_mean_ah", "delta_q_skew", "log10_abs_delta_q_min",
    ],
    "expanded": [
        "log10_delta_q_var", "delta_q_min_ah", "delta_q_abs_area_ahv",
        "qd_change_ah", "ir_mean_ohm", "tmax_mean_c", "charge_time_mean_min",
    ],
}


def read_array(handle: h5py.File, ref: h5py.Reference) -> np.ndarray:
    return np.asarray(handle[ref][()], dtype=float).squeeze()


def mean_at(summary: h5py.Group, key: str, mask: np.ndarray) -> float:
    values = np.asarray(summary[key][()], dtype=float).squeeze()[mask]
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def capacity_slope(cycle_no: np.ndarray, qd: np.ndarray, start: int, end: int) -> float:
    mask = (cycle_no >= start) & (cycle_no <= end) & np.isfinite(qd)
    return float(np.polyfit(cycle_no[mask], qd[mask], 1)[0]) if mask.sum() >= 2 else np.nan


def load_batch(path: Path, batch_name: str) -> pd.DataFrame:
    rows = []
    with h5py.File(path, "r") as handle:
        batch = handle["batch"]
        for idx in range(batch["cycle_life"].shape[0]):
            life = read_array(handle, batch["cycle_life"][idx, 0])
            life = float(life) if life.size == 1 else np.nan
            policy_codes = np.asarray(handle[batch["policy_readable"][idx, 0]][()], dtype=int).ravel()
            charging_policy = "".join(chr(code) for code in policy_codes)
            summary = handle[batch["summary"][idx, 0]]
            cycles = handle[batch["cycles"][idx, 0]]
            cycle_no = np.asarray(summary["cycle"][()], dtype=float).squeeze()
            early = (cycle_no >= 2) & (cycle_no <= 100)
            initial = (cycle_no >= 2) & (cycle_no <= 10)
            around_100 = (cycle_no >= 95) & (cycle_no <= 100)

            # Qdlin is indexed by recorded cycle: entries 9 and 99 are cycles 10 and 100.
            q10 = read_array(handle, cycles["Qdlin"][9, 0])
            q100 = read_array(handle, cycles["Qdlin"][99, 0])
            voltage = read_array(handle, batch["Vdlin"][idx, 0])
            if not (len(q10) == len(q100) == len(voltage)):
                raise ValueError(f"Qdlin/voltage length mismatch: {batch_name} cell {idx + 1}")
            delta = q100 - q10
            order = np.argsort(voltage)
            variance = float(np.nanvar(delta))
            delta_mean = float(np.nanmean(delta))
            delta_std = float(np.nanstd(delta))
            delta_skew = float(np.nanmean(((delta - delta_mean) / delta_std) ** 3)) if delta_std > 0 else np.nan

            qd_initial = mean_at(summary, "QDischarge", initial)
            qd_100 = mean_at(summary, "QDischarge", around_100)
            qd_all = np.asarray(summary["QDischarge"][()], dtype=float).squeeze()
            qd_cycle_2 = float(qd_all[cycle_no == 2][0]) if np.any(cycle_no == 2) else np.nan
            rows.append({
                "batch": batch_name,
                "cell_index": idx + 1,
                "charging_policy": charging_policy,
                "cycle_life": life,
                "final_qd_ah_for_qc_only": float(qd_all[-1]),
                "log10_delta_q_var": float(np.log10(max(variance, 1e-12))),
                "delta_q_min_ah": float(np.nanmin(delta)),
                "log10_abs_delta_q_min": float(np.log10(max(abs(np.nanmin(delta)), 1e-12))),
                "delta_q_mean_ah": delta_mean,
                "delta_q_skew": delta_skew,
                "delta_q_abs_area_ahv": float(np.trapezoid(np.abs(delta[order]), voltage[order])),
                "qd_change_ah": qd_100 - qd_initial,
                "qd_slope_2_100_ah_per_cycle": capacity_slope(cycle_no, qd_all, 2, 100),
                "qd_slope_91_100_ah_per_cycle": capacity_slope(cycle_no, qd_all, 91, 100),
                "qd_cycle_2_ah": qd_cycle_2,
                "qd_max_minus_cycle_2_ah": float(np.nanmax(qd_all[early]) - qd_cycle_2),
                "ir_mean_ohm": mean_at(summary, "IR", early),
                "tmax_mean_c": mean_at(summary, "Tmax", early),
                "charge_time_mean_min": mean_at(summary, "chargetime", early),
            })
    return pd.DataFrame(rows)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAPE_pct": round(100 * mean_absolute_percentage_error(y_true, y_pred), 2),
        "MAE_cycles": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE_cycles": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "R2": round(r2_score(y_true, y_pred), 3),
    }


def candidates():
    elastic = TransformedTargetRegressor(
        regressor=Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", ElasticNet(max_iter=30000, random_state=42)),
        ]),
        func=np.log,
        inverse_func=np.exp,
        check_inverse=False,
    )
    forest = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(random_state=42, n_jobs=1)),
    ])
    xgboost = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(
            objective="reg:squarederror", tree_method="hist", random_state=42,
            n_jobs=1, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=5.0, verbosity=0,
        )),
    ])
    return {
        "ElasticNet_log_target": (
            elastic,
            {"regressor__model__alpha": [0.001, 0.01, 0.1, 1.0],
             "regressor__model__l1_ratio": [0.2, 0.5, 0.8]},
        ),
        "RandomForest": (
            forest,
            {"model__n_estimators": [100, 300],
             "model__max_features": [0.7, 1.0],
             "model__max_depth": [2, 4],
             "model__min_samples_leaf": [2, 4]},
        ),
        "XGBoost": (
            xgboost,
            {"model__n_estimators": [100, 300],
             "model__learning_rate": [0.03, 0.1],
             "model__max_depth": [2, 3],
             "model__min_child_weight": [2, 5]},
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing the three MAT files")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for name, filename in FILES.items():
        path = args.data_dir / filename
        if path.exists():
            frames.append(load_batch(path, name))
        elif name != "Batch 3":
            raise FileNotFoundError(path)
    data = pd.concat(frames, ignore_index=True)
    data["quality_status"] = "included"
    data.loc[data.cycle_life.isna(), "quality_status"] = "target_missing"
    data.loc[
        data.cycle_life.notna() & (data.final_qd_ah_for_qc_only > 0.885),
        "quality_status",
    ] = "EOL_not_observed"
    # The paper's Batch 3 quality exclusions map to these original positions
    # in the supplied 2018-04-12 file (38, then 3/43/44 after other filters).
    data.loc[
        data.batch.eq("Batch 3") & data.cell_index.isin([3, 38, 43, 44]),
        "quality_status",
    ] = "paper_Batch3_quality_exclusion"
    data.to_csv(args.output_dir / "features_all_cells.csv", index=False)
    data.loc[~data.quality_status.eq("included"), [
        "batch", "cell_index", "cycle_life", "final_qd_ah_for_qc_only", "quality_status"
    ]].to_csv(args.output_dir / "quality_exclusions.csv", index=False)
    valid = data.loc[data.quality_status.eq("included") & (data.cycle_life > 100)].copy()
    b1 = valid.loc[valid.batch == "Batch 1"].reset_index(drop=True)
    b2 = valid.loc[valid.batch == "Batch 2"].reset_index(drop=True)
    b3 = valid.loc[valid.batch == "Batch 3"].reset_index(drop=True)
    if len(b1) < 20 or len(b2) < 10:
        raise ValueError("Insufficient labeled cells for the required Batch 1/2 split")

    holdout_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, holdout_idx = next(
        holdout_split.split(b1, b1.cycle_life, groups=b1.charging_policy)
    )
    train, holdout = b1.iloc[train_idx], b1.iloc[holdout_idx]
    assert set(train.charging_policy).isdisjoint(set(holdout.charging_policy))
    cv = GroupShuffleSplit(n_splits=15, test_size=0.2, random_state=42)
    baseline = DummyRegressor(strategy="median")
    baseline_cv = -cross_val_score(
        baseline, train[FEATURE_SETS["core"]], train.cycle_life,
        scoring="neg_mean_absolute_percentage_error", cv=cv, groups=train.charging_policy, n_jobs=1,
    ).mean() * 100

    rows = []
    best = None
    primary = None
    best_score = float("inf")
    for feature_set, feature_cols in FEATURE_SETS.items():
        for model_name, (estimator, grid) in candidates().items():
            search = GridSearchCV(
                estimator, grid, scoring="neg_mean_absolute_percentage_error",
                cv=cv, n_jobs=1, refit=True, error_score="raise",
            )
            search.fit(train[feature_cols], train.cycle_life, groups=train.charging_policy)
            cv_mape = -search.best_score_ * 100
            rows.append({
                "model": model_name, "feature_set": feature_set,
                "cv_MAPE_pct": round(cv_mape, 2),
                "cv_MAPE_fold_std_pct": round(
                    100 * search.cv_results_["std_test_score"][search.best_index_], 2
                ),
                "best_params": json.dumps(search.best_params_, sort_keys=True),
            })
            if cv_mape < best_score:
                best_score = cv_mape
                best = (model_name, feature_set, feature_cols, search.best_estimator_, search.best_params_)
            if model_name == "ElasticNet_log_target" and feature_set == "core":
                primary = (model_name, feature_set, feature_cols, search.best_estimator_, search.best_params_, cv_mape)

    pd.DataFrame(rows).sort_values("cv_MAPE_pct").to_csv(args.output_dir / "cv_comparison.csv", index=False)
    if primary is None or best is None:
        raise RuntimeError("Primary or exploratory model was not evaluated")
    model_name, feature_set, feature_cols, chosen, params, primary_cv_mape = primary
    exploratory_name, exploratory_set, _, _, exploratory_params = best

    shift_rows = []
    for feature in FEATURE_SETS["paper_inspired_combined"]:
        b1_values = b1[feature].dropna()
        b2_values = b2[feature].dropna()
        iqr = b1_values.quantile(0.75) - b1_values.quantile(0.25)
        shift_rows.append({
            "feature": feature,
            "batch1_median": b1_values.median(),
            "batch2_median": b2_values.median(),
            "median_shift_over_batch1_iqr": (
                (b2_values.median() - b1_values.median()) / iqr if iqr > 0 else np.nan
            ),
        })
    pd.DataFrame(shift_rows).to_csv(args.output_dir / "feature_shift_batch1_batch2.csv", index=False)
    holdout_pred = chosen.predict(holdout[feature_cols])
    holdout_predictions = holdout[["batch", "cell_index", "cycle_life"]].copy()
    holdout_predictions["predicted_cycle_life"] = np.round(holdout_pred, 1)
    holdout_predictions["absolute_percentage_error_pct"] = np.round(
        100 * np.abs(holdout_pred - holdout.cycle_life) / holdout.cycle_life, 2
    )
    holdout_predictions.to_csv(args.output_dir / "batch1_holdout_predictions.csv", index=False)
    baseline.fit(train[feature_cols], train.cycle_life)
    baseline_holdout = metrics(holdout.cycle_life, baseline.predict(holdout[feature_cols]))

    final_model = clone(chosen).fit(b1[feature_cols], b1.cycle_life)
    b2_pred = final_model.predict(b2[feature_cols])
    baseline.fit(b1[feature_cols], b1.cycle_life)
    baseline_b2 = metrics(b2.cycle_life, baseline.predict(b2[feature_cols]))
    predictions = b2[["batch", "cell_index", "cycle_life"]].copy()
    predictions["predicted_cycle_life"] = np.round(b2_pred, 1)
    predictions["absolute_percentage_error_pct"] = np.round(
        100 * np.abs(b2_pred - b2.cycle_life) / b2.cycle_life, 2
    )
    predictions.to_csv(args.output_dir / "batch2_predictions.csv", index=False)

    report = {
        "data": {"batch1_labeled": len(b1), "batch2_labeled": len(b2),
                 "batch3_labeled": len(b3), "batch1_train": len(train),
                 "batch1_holdout": len(holdout),
                 "batch1_train_policy_groups": train.charging_policy.nunique(),
                 "batch1_holdout_policy_groups": holdout.charging_policy.nunique(),
                 "quality_exclusions": data.loc[~data.quality_status.eq("included")]
                     .groupby(["batch", "quality_status"]).size()
                     .rename("cells").reset_index().to_dict(orient="records")},
        "selection": {"model": model_name, "feature_set": feature_set,
                      "features": feature_cols, "parameters": params,
                      "CV_MAPE_pct": round(primary_cv_mape, 2),
                      "median_baseline_CV_MAPE_pct": round(baseline_cv, 2)},
        "exploratory_best_batch1_cv": {
            "model": exploratory_name, "feature_set": exploratory_set,
            "parameters": exploratory_params, "CV_MAPE_pct": round(best_score, 2),
            "note": "Ablation only; not selected as the primary cross-batch model",
        },
        "batch1_holdout": metrics(holdout.cycle_life, holdout_pred),
        "batch1_holdout_median_baseline": baseline_holdout,
        "batch2_final_test": metrics(b2.cycle_life, b2_pred),
        "batch2_median_baseline": baseline_b2,
    }
    if len(b3):
        report["batch3_optional_external"] = metrics(b3.cycle_life, final_model.predict(b3[feature_cols]))
    train_mape = report["selection"]["CV_MAPE_pct"]
    valid_mape = report["batch1_holdout"]["MAPE_pct"]
    test_mape = report["batch2_final_test"]["MAPE_pct"]
    reporting_rows = [
        {"구분": "Train (Batch 1 CV)", "MAPE (%)": train_mape, "비고": "15 group-aware CV splits"},
        {"구분": "Valid (Batch 1 Hold-out)", "MAPE (%)": valid_mape, "비고": "정책 그룹 분리"},
        {"구분": "Test (Batch 2)", "MAPE (%)": test_mape, "비고": "최종 평가"},
        {"구분": "Gap (Train-Valid)", "MAPE (%)": round(valid_mape - train_mape, 2), "비고": "Valid - Train"},
        {"구분": "Gap (Valid-Test)", "MAPE (%)": round(test_mape - valid_mape, 2), "비고": "Test - Valid"},
        {"구분": "Gap (Target-Test)", "MAPE (%)": round(test_mape - 9.1, 2), "비고": "Test - 논문 목표 9.1%"},
    ]
    if len(b3):
        b3_mape = report["batch3_optional_external"]["MAPE_pct"]
        reporting_rows.extend([
            {"구분": "Test (Batch 3, optional)", "MAPE (%)": b3_mape,
             "비고": "원 저자 기준 품질 제외 후 탐색적 평가"},
            {"구분": "Gap (Batch2-Batch3)", "MAPE (%)": round(b3_mape - test_mape, 2),
             "비고": "Batch 3 - Batch 2"},
            {"구분": "Gap (Target-Test, Batch 3)", "MAPE (%)": round(b3_mape - 9.1, 2),
             "비고": "Batch 3 - 논문 목표 9.1%"},
        ])
    pd.DataFrame(reporting_rows).to_csv(args.output_dir / "performance_reporting.csv", index=False)
    report["gaps_percentage_points"] = {
        "valid_minus_train": reporting_rows[3]["MAPE (%)"],
        "test_minus_valid": reporting_rows[4]["MAPE (%)"],
        "test_minus_paper_target": reporting_rows[5]["MAPE (%)"],
    }
    if len(b3):
        report["gaps_percentage_points"].update({
            "batch3_minus_batch2": reporting_rows[7]["MAPE (%)"],
            "batch3_minus_paper_target": reporting_rows[8]["MAPE (%)"],
        })
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
