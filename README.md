# realtime-fraud-scoring — Streaming Fraud Detection

A production-style **real-time fraud scoring system**: a simulated transaction
stream, online feature engineering with per-card rolling windows, a trained
gradient-boosting classifier, and a FastAPI scoring service with p50/p99
latency tracking.

The simulator is deliberately adversarial: fraud amounts overlap legitimate
spend, so the model must learn **velocity patterns** (burst rate, time-since-last,
amount-vs-history) — not just "big amount = fraud".

## Architecture

```
TransactionStream (simulated Kafka topic)
        │  chronological card transactions w/ hidden fraud labels
        ▼
OnlineFeatureStore ──► per-card rolling state (deque, 1h window)
        │  features BEFORE the txn lands (like Flink/Tecton at serve time):
        │  amount, txn_count_10m/1h, total/avg/max_amt_1h,
        │  time_since_last_s, amount_vs_avg_ratio, is_night, is_risky_merchant
        ▼
HistGradientBoostingClassifier (class_weight=balanced)
        │
        ├──► demo.py ──► stream simulation + precision/recall + p50/p99 latency
        │
        └──► api.py ──► FastAPI: POST /score  GET /metrics  GET /health
```

## Quickstart

```bash
pip install -r requirements.txt

# End-to-end demo: train on 20k txns, score a 3k-txn live stream
python demo.py

# Serve the scoring API
uvicorn api:app --reload
curl -X POST localhost:8000/score -H 'Content-Type: application/json' \
     -d '{"card_id":"card_0001","amount":750.0,"merchant_category":"travel"}'
```

## Sample output

```
=== Holdout evaluation (training phase) ===
  precision              0.7500
  recall                 0.9231
  f1                     0.8276

=== Live-stream evaluation ===
  n_scored               3000
  frauds_in_stream       223
  frauds_caught          206
  precision              0.7687
  recall                 0.9238
  f1                     0.8391
  latency_ms_p50         3.9596
  latency_ms_p99         260.9574
  throughput_txn_per_s   66.0343
```

Top features by permutation importance: `is_risky_merchant`, `amount`,
`time_since_last_s` — velocity signals, as intended.

## How it works

- **Stream** (`src/stream.py`) — per-card spending profiles generate legitimate
  traffic; fraud arrives as bursts of rapid transactions on compromised cards.
  Ground-truth labels exist only for evaluation, never for scoring.
- **Features** (`src/features.py`) — the store computes features from history
  *before* appending the current transaction, mirroring production streaming
  feature stores where serving features must exclude the event being scored.
- **Model** (`src/model.py`) — `HistGradientBoostingClassifier` with balanced
  class weights; permutation importance for honest feature attribution.
- **API** (`api.py`) — scores single transactions, tracks in-memory p50/p99
  latency across requests; `/health` reports model load state.

## Tech stack

Python, scikit-learn (HistGradientBoosting), FastAPI + uvicorn, NumPy, pickle
model artifacts. No external services — the stream is simulated in-process.
