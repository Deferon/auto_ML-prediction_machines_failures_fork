"""Train CatBoost model with MLflow tracking."""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.catboost
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import (
    ARTIFACTS_DIR,
    CAT_FEATURES,
    CATBOOST_PARAMS,
    FEATURE_SCHEMA_VERSION,
    MODEL_FEATURES,
    RANDOM_STATE,
    SMOKE_SAMPLE_SIZE,
    TARGET_COL,
    TRAIN_TEST_SIZE,
)
from src.etl.features import build_features, get_feature_matrix
from src.etl.load import load_train
from src.logging_setup import configure_logging
from src.mlflow_setup import setup_mlflow
from src.monitoring import (
    build_training_monitoring_summary,
    compute_data_quality_report,
    infrastructure_snapshot,
    save_monitoring_report,
)
from src.plots import (
    save_confusion_matrix,
    save_feature_importance,
    save_infrastructure_chart,
    save_model_metrics_chart,
    save_roc_curve,
)

logger = logging.getLogger(__name__)


def train_model(
    sample_size: int | None = None,
    output_dir: Path | None = None,
    data_path: Path | None = None,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Train, evaluate and persist a model with reproducible feature metadata."""
    if sample_size is not None and sample_size < 10:
        raise ValueError("sample_size must be at least 10")
    output_dir = output_dir or ARTIFACTS_DIR
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_train(data_path)
    if df[TARGET_COL].nunique() != 2 or df[TARGET_COL].value_counts().min() < 2:
        raise ValueError("Training requires both target classes with at least two rows each")
    if sample_size is not None and sample_size < len(df):
        df, _ = train_test_split(
            df, train_size=sample_size, stratify=df[TARGET_COL], random_state=RANDOM_STATE
        )
    if df[TARGET_COL].nunique() != 2 or df[TARGET_COL].value_counts().min() < 2:
        raise ValueError("Sample must contain at least two rows of each target class")

    df = build_features(df, is_train=True)
    X, y = get_feature_matrix(df, include_target=True)
    if y is None:
        raise ValueError("Training target is missing")

    cat_idx = [X.columns.get_loc(c) for c in CAT_FEATURES if c in X.columns]
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TRAIN_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx)

    setup_mlflow(tracking_uri=tracking_uri, artifact_dir=output_dir / "mlartifacts")
    logger.info("Training with %d rows; validating with %d rows", len(X_train), len(X_val))

    infra_before = infrastructure_snapshot()
    t0 = time.perf_counter()

    with mlflow.start_run(run_name="catboost_train"):
        for key, value in CATBOOST_PARAMS.items():
            mlflow.log_param(key, value)
        mlflow.log_param("sample_size", len(df))
        mlflow.log_param("n_features", len(X.columns))
        mlflow.log_param("feature_schema_version", FEATURE_SCHEMA_VERSION)

        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        train_time = time.perf_counter() - t0
        mlflow.log_metric("train_time_sec", train_time)

        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_val, y_proba)),
            "train_time_sec": train_time,
        }
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(name, value)

        infra_after = infrastructure_snapshot()
        mlflow.log_metric("cpu_percent_before", infra_before["cpu_percent"])
        mlflow.log_metric("cpu_percent_after", infra_after["cpu_percent"])
        mlflow.log_metric("ram_used_percent_before", infra_before["ram_used_percent"])
        mlflow.log_metric("ram_used_percent_after", infra_after["ram_used_percent"])

        model_path = output_dir / "model.cbm"
        model.save_model(str(model_path))
        mlflow.log_artifact(str(model_path))
        metadata = {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "features": MODEL_FEATURES,
            "categorical_features": CAT_FEATURES,
            "random_seed": RANDOM_STATE,
            "training_rows": len(X_train),
            "validation_rows": len(X_val),
        }
        metadata_path = output_dir / "model_metadata.json"
        save_monitoring_report(metadata, metadata_path)
        mlflow.log_artifact(str(metadata_path))

        importance = dict(zip(X.columns, model.get_feature_importance().tolist(), strict=False))
        metrics["feature_importance_top5"] = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        )

        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_artifact(str(metrics_path))
        save_confusion_matrix(y_val, y_pred, plots_dir / "confusion_matrix.png")
        save_roc_curve(y_val, y_proba, plots_dir / "roc_curve.png")
        save_feature_importance(importance, plots_dir / "feature_importance.png")
        for plot in plots_dir.glob("*.png"):
            mlflow.log_artifact(str(plot))

        data_report = compute_data_quality_report(df, "train_processed")
        data_report["infrastructure"] = {
            "before": infra_before,
            "after": infra_after,
        }
        save_monitoring_report(data_report, output_dir / "data_quality.json")
        mlflow.log_artifact(str(output_dir / "data_quality.json"))

        monitoring_summary = build_training_monitoring_summary(metrics, data_report)
        summary_path = output_dir / "monitoring_summary.json"
        save_monitoring_report(monitoring_summary, summary_path)
        mlflow.log_artifact(str(summary_path))

        save_model_metrics_chart(plots_dir / "model_metrics.png", metrics)
        save_infrastructure_chart(
            plots_dir / "infrastructure_training.png",
            stage="training",
            before=infra_before,
            after=infra_after,
            duration_sec=train_time,
            duration_label="Training",
        )
        mlflow.log_artifact(str(plots_dir / "model_metrics.png"))
        mlflow.log_artifact(str(plots_dir / "infrastructure_training.png"))

        mlflow.catboost.log_model(model, "model", input_example=X_val.head(5))

    logger.info("Model and metrics saved to %s", output_dir)
    return metrics


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Train machine failure model")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Use a stratified subset (default: full dataset)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Train on {SMOKE_SAMPLE_SIZE} rows",
    )
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()
    if args.sample_size is not None and args.sample_size < 10:
        parser.error("--sample-size must be at least 10")
    sample = (
        args.sample_size
        if args.sample_size is not None
        else (SMOKE_SAMPLE_SIZE if args.smoke else None)
    )
    metrics = train_model(
        sample_size=sample,
        output_dir=args.output_dir,
        data_path=args.data,
        tracking_uri=args.tracking_uri,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
