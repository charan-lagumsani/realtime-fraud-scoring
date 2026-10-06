"""Stream evaluation: precision/recall over a simulated live stream plus
p50/p99 scoring-latency statistics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def summarize(y_true: list[int], y_pred: list[int],
              latencies_ms: list[float]) -> dict:
    lat = np.asarray(latencies_ms, dtype=float)
    return {
        "n_scored": len(y_true),
        "frauds_in_stream": int(sum(y_true)),
        "frauds_caught": int(sum(1 for t, p in zip(y_true, y_pred) if t and p)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "latency_ms_p50": float(np.percentile(lat, 50)),
        "latency_ms_p99": float(np.percentile(lat, 99)),
        "latency_ms_mean": float(lat.mean()),
        "throughput_txn_per_s": float(1000.0 / lat.mean()) if lat.mean() > 0 else 0.0,
    }


def print_report(title: str, report: dict) -> None:
    print(f"\n=== {title} ===")
    for key, val in report.items():
        if isinstance(val, float):
            print(f"  {key:22s} {val:.4f}")
        elif isinstance(val, dict):
            continue
        else:
            print(f"  {key:22s} {val}")
