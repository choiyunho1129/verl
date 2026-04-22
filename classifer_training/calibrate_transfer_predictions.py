from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, Ridge

from classifer_training.single_rollout_hidden_utils import reg_metrics, save_diagnostics_plot
from classifer_training.utils import write_jsonl


def _load_prediction_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.expanduser().resolve().open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["task_id"])] = row
    return rows


def _aligned_arrays(
    primary_path: Path,
    aux_path: Path | None,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray | None, dict[str, dict[str, Any]]]:
    primary_rows = _load_prediction_rows(primary_path)
    aux_rows = _load_prediction_rows(aux_path) if aux_path is not None else None
    task_ids = sorted(primary_rows.keys() if aux_rows is None else set(primary_rows).intersection(aux_rows))
    if not task_ids:
        raise ValueError(f"No aligned prediction rows for {primary_path} and {aux_path}.")

    y_true = np.asarray([float(primary_rows[task_id]["value_true"]) for task_id in task_ids], dtype=np.float64)
    primary_pred = np.asarray([float(primary_rows[task_id]["value_pred"]) for task_id in task_ids], dtype=np.float64)
    aux_pred = (
        None
        if aux_rows is None
        else np.asarray([float(aux_rows[task_id]["value_pred"]) for task_id in task_ids], dtype=np.float64)
    )
    return task_ids, y_true, primary_pred, aux_pred, primary_rows


def _score_features(primary_pred: np.ndarray, aux_pred: np.ndarray) -> np.ndarray:
    diff = primary_pred - aux_pred
    return np.stack(
        [
            primary_pred,
            aux_pred,
            diff,
            np.abs(diff),
            primary_pred * aux_pred,
            np.minimum(primary_pred, aux_pred),
            np.maximum(primary_pred, aux_pred),
        ],
        axis=1,
    )


def _fit_calibrator(
    method: str,
    y_cal: np.ndarray,
    primary_cal: np.ndarray,
    aux_cal: np.ndarray | None,
) -> dict[str, Any]:
    if method == "raw":
        return {"method": method}
    if method == "mean_shift":
        return {"method": method, "offset": float(np.mean(y_cal) - np.mean(primary_cal))}
    if method == "linear":
        model = LinearRegression()
        model.fit(primary_cal.reshape(-1, 1), y_cal)
        return {"method": method, "model": model}
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(primary_cal, y_cal)
        return {"method": method, "model": model}
    if method == "linear_stack_prompt":
        if aux_cal is None:
            raise ValueError("linear_stack_prompt requires --calibration_aux_predictions and --eval_aux_predictions.")
        model = Ridge(alpha=1.0)
        model.fit(_score_features(primary_cal, aux_cal), y_cal)
        return {"method": method, "model": model}
    raise ValueError(f"Unsupported calibration method: {method}")


def _predict_calibrated(
    calibrator: dict[str, Any],
    primary_pred: np.ndarray,
    aux_pred: np.ndarray | None,
) -> np.ndarray:
    method = str(calibrator["method"])
    if method == "raw":
        pred = primary_pred
    elif method == "mean_shift":
        pred = primary_pred + float(calibrator["offset"])
    elif method in {"linear", "isotonic"}:
        pred = calibrator["model"].predict(primary_pred.reshape(-1, 1) if method == "linear" else primary_pred)
    elif method == "linear_stack_prompt":
        if aux_pred is None:
            raise ValueError("linear_stack_prompt requires aux predictions at inference.")
        pred = calibrator["model"].predict(_score_features(primary_pred, aux_pred))
    else:
        raise ValueError(f"Unsupported calibration method: {method}")
    return np.clip(np.asarray(pred, dtype=np.float64).reshape(-1), 0.0, 1.0)


def _extra_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = reg_metrics(y_true, y_pred)
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        out["pearson"] = float(pearsonr(y_true, y_pred).statistic)
        out["spearman"] = float(spearmanr(y_true, y_pred).statistic)
    else:
        out["pearson"] = 0.0
        out["spearman"] = 0.0
    out["gt_mean"] = float(np.mean(y_true))
    out["pred_mean"] = float(np.mean(y_pred))
    out["bias"] = float(np.mean(y_pred - y_true))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a lightweight calibration layer on transfer predictions.")
    parser.add_argument("--calibration_predictions", type=Path, required=True)
    parser.add_argument("--eval_predictions", type=Path, required=True)
    parser.add_argument("--calibration_aux_predictions", type=Path)
    parser.add_argument("--eval_aux_predictions", type=Path)
    parser.add_argument(
        "--method",
        choices=["raw", "mean_shift", "linear", "isotonic", "linear_stack_prompt"],
        default="isotonic",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cal_task_ids, y_cal, primary_cal, aux_cal, _ = _aligned_arrays(
        args.calibration_predictions,
        args.calibration_aux_predictions,
    )
    eval_task_ids, y_eval, primary_eval, aux_eval, eval_rows = _aligned_arrays(
        args.eval_predictions,
        args.eval_aux_predictions,
    )

    calibrator = _fit_calibrator(args.method, y_cal, primary_cal, aux_cal)
    eval_pred = _predict_calibrated(calibrator, primary_eval, aux_eval)

    output_rows: list[dict[str, Any]] = []
    for task_id, true_value, pred_value in zip(eval_task_ids, y_eval.tolist(), eval_pred.tolist(), strict=True):
        base = dict(eval_rows[task_id])
        base["value_true"] = float(true_value)
        base["value_pred_raw"] = float(base["value_pred"])
        base["value_pred"] = float(pred_value)
        output_rows.append(base)

    write_jsonl(args.output_dir / "predictions_calibrated.jsonl", output_rows)
    save_diagnostics_plot(
        args.output_dir / "prediction_diagnostics_calibrated.png",
        output_rows,
        f"Calibrated Transfer: {args.method}",
    )

    serializable_calibrator = {
        key: value for key, value in calibrator.items() if key != "model"
    }
    if "model" in calibrator:
        joblib.dump(calibrator["model"], args.output_dir / "calibrator_model.joblib")
        serializable_calibrator["model_path"] = "calibrator_model.joblib"

    summary = {
        "setting": "transfer_prediction_calibration",
        "method": args.method,
        "num_calibration_prompts": int(len(cal_task_ids)),
        "num_eval_prompts": int(len(eval_task_ids)),
        "calibration_predictions": str(args.calibration_predictions.expanduser().resolve()),
        "eval_predictions": str(args.eval_predictions.expanduser().resolve()),
        "calibration_aux_predictions": None
        if args.calibration_aux_predictions is None
        else str(args.calibration_aux_predictions.expanduser().resolve()),
        "eval_aux_predictions": None
        if args.eval_aux_predictions is None
        else str(args.eval_aux_predictions.expanduser().resolve()),
        "calibrator": serializable_calibrator,
        "raw_eval_metrics": _extra_metrics(y_eval, primary_eval),
        "calibrated_eval_metrics": _extra_metrics(y_eval, eval_pred),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
