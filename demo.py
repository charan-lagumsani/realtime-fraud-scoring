"""End-to-end demo: train on historical transactions, then score a simulated
live stream and report fraud-detection quality + serving latency.

Usage:
    python demo.py [--train-txns 20000] [--stream-txns 3000]
"""

from __future__ import annotations

import argparse
import time

from src.stream import TransactionStream
from src.features import OnlineFeatureStore, FEATURE_NAMES
from src.model import build_training_frame, train_model, save_model, load_model
from src.evaluate import summarize, print_report

MODEL_PATH = "data/fraud_model.pkl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-txns", type=int, default=20000)
    ap.add_argument("--stream-txns", type=int, default=3000)
    args = ap.parse_args()

    print("realtime-fraud-scoring — streaming fraud detection demo")
    print(f"features: {', '.join(FEATURE_NAMES)}")

    # 1. Historical data + training -------------------------------------
    print(f"\n[1/3] Generating {args.train_txns} historical transactions...")
    history = TransactionStream(seed=7).generate(args.train_txns)
    X, y = build_training_frame(history)
    print(f"      feature matrix: {X.shape}, fraud rate: {y.mean():.3%}")

    print("[2/3] Training gradient-boosting classifier...")
    clf, train_report = train_model(X, y)
    print_report("Holdout evaluation (training phase)", train_report)
    top_feats = sorted(train_report["feature_importance"].items(),
                       key=lambda kv: kv[1], reverse=True)[:3]
    print("      top features:", ", ".join(f"{k}={v:.2f}" for k, v in top_feats))
    save_model(clf, MODEL_PATH)
    clf = load_model(MODEL_PATH)  # prove the artifact round-trips

    # 2. Live stream simulation -----------------------------------------
    print(f"\n[3/3] Simulating live stream of {args.stream_txns} transactions...")
    stream = TransactionStream(seed=99).generate(args.stream_txns)
    store = OnlineFeatureStore()
    y_true, y_pred, latencies = [], [], []
    for txn in stream:
        feats = store.featurize(txn)
        t0 = time.perf_counter()
        proba = float(clf.predict_proba([feats])[0][1])
        latencies.append((time.perf_counter() - t0) * 1000.0)
        y_true.append(1 if txn.is_fraud else 0)
        y_pred.append(1 if proba >= 0.5 else 0)

    stream_report = summarize(y_true, y_pred, latencies)
    print_report("Live-stream evaluation", stream_report)
    print("\nServe it live:  uvicorn api:app --reload")
    print("  curl -X POST localhost:8000/score -H 'Content-Type: application/json' \\")
    print("       -d '{\"card_id\":\"card_0001\",\"amount\":999.0,"
          "\"merchant_category\":\"travel\"}'")


if __name__ == "__main__":
    main()
