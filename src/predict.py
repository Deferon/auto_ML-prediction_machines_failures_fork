"""Inference and business recommendations."""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.config import (
    ARTIFACTS_DIR,
    CAT_FEATURES,
    ID_COL,
    PRODUCT_ID_COL,
    RISK_THRESHOLDS,
    TYPE_COL,
)
from src.etl.features import build_features, get_feature_matrix
from src.etl.load import load_test, load_train
from src.logging_setup import configure_logging
from src.monitoring import (
    build_inference_monitoring_summary,
    compare_distributions,
    compute_data_quality_report,
    infrastructure_snapshot,
    save_monitoring_report,
)

logger = logging.getLogger(__name__)


def assign_risk_level_vector(probs: pd.Series) -> pd.Series:
    """Assign fixed operational thresholds, including singleton and tied scores."""
    if not np.isfinite(probs).all() or not probs.between(0, 1).all():
        raise ValueError("Probabilities must be finite and between 0 and 1")
    return pd.cut(
        probs,
        bins=[-np.inf, RISK_THRESHOLDS["low"], RISK_THRESHOLDS["medium"], np.inf],
        labels=["Низкий", "Средний", "Высокий"],
        right=False,
    ).astype(str)


def predict(
    model_path: Path | None = None,
    output_dir: Path | None = None,
    use_train_for_drift: bool = True,
    data_path: Path | None = None,
    reference_path: Path | None = None,
) -> pd.DataFrame:
    """
    Выполняет предсказание вероятности отказа оборудования, для полученной вероятности присвает риск, выраженный перечислением ["Низкий", "Средний", "Высокий"].
    Формирует рекоммендации по улучшению качества модели и, при необходимости, рассчитывает дрейф данных.
    Для презентации работоспособности используются тестовые данные
    """
    output_dir = output_dir or ARTIFACTS_DIR
    model_path = model_path or output_dir / "model.cbm"
    started = time.perf_counter()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run: python -m src.train")

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    test_raw = load_test(data_path)
    test_df = build_features(test_raw, is_train=False)
    X, _ = get_feature_matrix(test_df, include_target=False)
    if model.feature_names_ != list(X.columns):
        raise ValueError("Model feature schema mismatch. Retrain with: poetry run ml-train")

    cat_idx = [X.columns.get_loc(c) for c in CAT_FEATURES if c in X.columns]
    pool = Pool(X, cat_features=cat_idx)
    infra_before = infrastructure_snapshot()
    inference_started = time.perf_counter()
    probabilities = model.predict_proba(pool)[:, 1]
    inference_time = time.perf_counter() - inference_started
    infra_after = infrastructure_snapshot()

    result = test_df[[ID_COL, PRODUCT_ID_COL, TYPE_COL]].copy()

    result["failure_probability"] = probabilities
    for col in [
        "efficiency [%]",
        "Tool wear [min]",
        "delta_temperature [K]",
        "Power [kW]",
        "air_mass",
    ]:
        if col in test_df.columns:
            result[col] = test_df[col].values
    if "Power [kW]" in result.columns:
        result["mechanical_power [kW]"] = result["Power [kW]"]
    if "air_mass" in result.columns:
        result["air_mass [kg/s]"] = result["air_mass"]

    result["risk_level"] = assign_risk_level_vector(pd.Series(probabilities, index=result.index))

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    result.to_csv(predictions_path, index=False)

    recommendations = build_recommendations(result)
    rec_path = output_dir / "maintenance_recommendations.csv"
    recommendations.to_csv(rec_path, index=False)

    drift = {}
    if use_train_for_drift:
        train_df = build_features(load_train(reference_path), is_train=True)
        drift = compare_distributions(train_df, test_df)
    report = build_inference_monitoring_summary(
        predictions_rows=len(result),
        high_risk_count=int((result["risk_level"] == "Высокий").sum()),
        drift=drift,
        test_quality=compute_data_quality_report(test_df, "test"),
        infrastructure={"before": infra_before, "after": infra_after},
        inference_time_sec=inference_time,
        pipeline_time_sec=time.perf_counter() - started,
    )
    if not use_train_for_drift:
        report["drift"] = {"overall_status": "not_evaluated", "features": {}, "alerts": []}
    save_monitoring_report(report, output_dir / "inference_monitoring.json")
    logger.info("Saved %d predictions to %s", len(result), predictions_path)

    return result


def build_recommendations(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    На основании полученных предсказаний формирует рекоммендации для последующего улучшения качества модели
    """
    rows = []

    if "efficiency [%]" in predictions.columns:
        low_eff = (predictions["efficiency [%]"] < 50).mean()
        if low_eff > 0.1:
            rows.append(
                {
                    "Проблема": "Низкая эффективность системы",
                    "Решение": "Настроить систему при КПД < 50.0%",
                    "Приоритет": "Высокий",
                }
            )

    if "Tool wear [min]" in predictions.columns:
        wear_threshold = predictions["Tool wear [min]"].quantile(0.90)
        high_wear = (predictions["Tool wear [min]"] > wear_threshold).sum()
        if high_wear > 0:
            rows.append(
                {
                    "Проблема": "Критический износ инструмента",
                    "Решение": f"Плановые замены при износе > {wear_threshold:.0f} мин",
                    "Приоритет": "Высокий",
                }
            )

    high_risk = (predictions["risk_level"] == "Высокий").sum()
    if high_risk > 0:
        rows.append(
            {
                "Проблема": "Высокий прогноз отказа",
                "Решение": "Целевое обслуживание оборудования с risk_level=Высокий",
                "Приоритет": "Высокий",
            }
        )

    if not rows:
        rows.append(
            {
                "Проблема": "Стабильное состояние",
                "Решение": "Продолжить мониторинг KPI",
                "Приоритет": "Низкий",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--reference-data", type=Path, default=None)
    parser.add_argument("--no-drift", action="store_true")
    args = parser.parse_args()
    result = predict(
        model_path=args.model,
        output_dir=args.output_dir,
        data_path=args.data,
        reference_path=args.reference_data,
        use_train_for_drift=not args.no_drift,
    )
    print(f"Predictions saved: {len(result)} rows")


if __name__ == "__main__":
    main()
