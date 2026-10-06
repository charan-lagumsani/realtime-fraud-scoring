"""FastAPI scoring service for real-time fraud detection.

POST /score  -> fraud probability for one transaction (with latency tracking)
GET  /metrics -> p50/p99 serving latency and request counts
GET  /health  -> liveness probe
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import OnlineFeatureStore
from src.model import load_model
from src.stream import Transaction

MODEL_PATH = Path(__file__).parent / "data" / "fraud_model.pkl"

app = FastAPI(title="realtime-fraud-scoring",
              description="Streaming fraud scoring API")

_model = load_model(MODEL_PATH) if MODEL_PATH.exists() else None
_store = OnlineFeatureStore()
_latencies_ms: list[float] = []


class ScoreRequest(BaseModel):
    card_id: str = Field(..., examples=["card_0001"])
    amount: float = Field(..., gt=0, examples=[999.0])
    merchant_category: str = Field(..., examples=["travel"])


class ScoreResponse(BaseModel):
    card_id: str
    fraud_probability: float
    is_fraud: bool
    latency_ms: float


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    if _model is None:
        raise HTTPException(503, "model not trained yet — run python demo.py first")
    txn = Transaction(txn_id="live", card_id=req.card_id, amount=req.amount,
                      merchant_category=req.merchant_category,
                      timestamp=datetime.utcnow())
    t0 = time.perf_counter()
    feats = _store.featurize(txn)
    proba = float(_model.predict_proba([feats])[0][1])
    latency_ms = (time.perf_counter() - t0) * 1000.0
    _latencies_ms.append(latency_ms)
    return ScoreResponse(card_id=req.card_id, fraud_probability=proba,
                         is_fraud=proba >= 0.5, latency_ms=latency_ms)


@app.get("/metrics")
def metrics():
    lat = np.asarray(_latencies_ms or [0.0])
    return {
        "requests": len(_latencies_ms),
        "latency_ms_p50": float(np.percentile(lat, 50)),
        "latency_ms_p99": float(np.percentile(lat, 99)),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}
