import os
import pytest
from pathlib import Path
import json
import mlflow
import dagshub
from utils.mlflow_utils import (
    get_metrics_from_stage,
    get_run_by_stage,
    CHAMPION,
    CHALLENGER,
)
from typing import Literal

ROOT_DIR = Path(__file__).parent.parent
threshold_values_path = ROOT_DIR / "thresholds.json"

NOISE_MULTIPLIER = 2
MIN_METRICS_TO_PASS = 4


def load_thresholds(
    thresholds_type: Literal["noise_thresholds", "historical_thresholds"],
    thresholds_path: Path | str,
) -> dict:
    if isinstance(thresholds_path, str):
        thresholds_path = Path(thresholds_path)

    if thresholds_path.exists():
        with open(thresholds_path, "r") as file:
            return json.load(file)[thresholds_type]

    return {}


# Initialize DagsHub and MLflow.
dagshub.init(repo_owner="atulkrs", repo_name="llmops-rag-app", mlflow=True)

# Set the tracking server.
mlflow.set_tracking_uri("https://dagshub.com/atulkrs/llmops-rag-app.mlflow")

# Fetch the experiment.
experiment = mlflow.get_experiment_by_name("rag-app")

if experiment is None:
    pytest.skip(
        "rag-app MLflow experiment not found; promotion gate requires a live experiment.",
        allow_module_level=True,
    )

experiment_id = experiment.experiment_id

try:
    challenger_run = get_run_by_stage(CHALLENGER, experiment_id)
except RuntimeError:
    if os.getenv("CI") == "true":
        raise RuntimeError(
            "CI quality gate requires exactly one stage='challenger' run."
        )

    pytest.skip(
        "No stage='challenger' run exists; gate is evaluated after CI creates a challenger.",
        allow_module_level=True,
    )

latest_metrics = challenger_run.data.metrics

noise_thresholds = load_thresholds(
    thresholds_type="noise_thresholds",
    thresholds_path=threshold_values_path,
)

champion_metrics = get_metrics_from_stage(
    stage_name=CHAMPION,
    experiment_id=experiment_id,
)

champion_metrics_names = list(champion_metrics.keys())
latest_metrics_names = list(latest_metrics.keys())


def test_similar_metric_names():
    assert champion_metrics_names == latest_metrics_names, (
        "comparison metrics differ, use same metrics for comparison only"
    )


def test_promotion_majority_vote():
    verdicts = []
    better_or_equal_count = 0
    tolerance_breaches = []

    for metric in latest_metrics_names:
        if metric not in noise_thresholds:
            pytest.fail(
                f"No noise threshold for metric {metric!r}; "
                "thresholds.json and the eval metrics are out of sync"
            )

        stage_value = champion_metrics[metric]
        latest_value = latest_metrics[metric]
        tolerance = NOISE_MULTIPLIER * noise_thresholds[metric]
        delta = latest_value - stage_value

        is_better_or_equal = delta >= 0
        is_within_tolerance = delta >= -tolerance

        if is_better_or_equal:
            better_or_equal_count += 1

        if not is_within_tolerance:
            tolerance_breaches.append(metric)

        status = (
            "BETTER"
            if is_better_or_equal
            else "NOISE"
            if is_within_tolerance
            else "FAIL"
        )

        verdicts.append(
            f"  [{status}] {metric!r}: "
            f"latest={latest_value:.4f} "
            f"champion={stage_value:.4f} "
            f"delta={delta:+.4f} "
            f"tolerance=+/-{tolerance:.4f}"
        )

    total = len(latest_metrics_names)
    detail = "\n".join(verdicts)

    assert not tolerance_breaches, (
        f"Promotion gate failed: metric(s) {tolerance_breaches} "
        f"regressed beyond their noise tolerance.\n"
        f"Per-metric detail:\n{detail}"
    )

    assert better_or_equal_count >= MIN_METRICS_TO_PASS, (
        f"Promotion gate failed: {better_or_equal_count}/{total} "
        f"metrics better-than-or-equal-to champion "
        f"(need >= {MIN_METRICS_TO_PASS}/{total}); remaining metrics "
        f"must still fall within noise tolerance.\n"
        f"Per-metric detail:\n{detail}"
    )
