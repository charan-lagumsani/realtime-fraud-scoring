"""Online feature engineering for streaming fraud scoring.

`OnlineFeatureStore` keeps per-card rolling state in memory and computes
velocity features for each incoming transaction *before* the transaction is
added to the card's history — exactly how a production streaming feature
store (Flink/Flink-ML, Tecton) serves features at scoring time.
"""

from __future__ import annotations

from collections import deque
from datetime import timedelta

from .stream import Transaction, RISKY_MERCHANTS

WINDOW_10M = timedelta(minutes=10)
WINDOW_1H = timedelta(hours=1)

FEATURE_NAMES = [
    "amount",
    "txn_count_10m",
    "txn_count_1h",
    "total_amt_1h",
    "avg_amt_1h",
    "max_amt_1h",
    "time_since_last_s",
    "amount_vs_avg_ratio",
    "is_night",
    "is_risky_merchant",
]


class OnlineFeatureStore:
    def __init__(self):
        # card_id -> deque of (timestamp, amount); kept sorted by arrival time
        self.history: dict[str, deque] = {}

    def _prune(self, card_id: str, now):
        dq = self.history[card_id]
        while dq and now - dq[0][0] > WINDOW_1H:
            dq.popleft()

    def featurize(self, txn: Transaction) -> list[float]:
        """Compute features from history, then append txn to history."""
        dq = self.history.setdefault(txn.card_id, deque())
        self._prune(txn.card_id, txn.timestamp)

        past = [(ts, amt) for ts, amt in dq]
        recent_10m = [amt for ts, amt in past if txn.timestamp - ts <= WINDOW_10M]
        amounts_1h = [amt for _, amt in past]

        if past:
            last_ts = past[-1][0]
            time_since_last = (txn.timestamp - last_ts).total_seconds()
            avg_1h = sum(amounts_1h) / len(amounts_1h)
        else:
            time_since_last = 3600.0  # no history: treat as "long ago"
            avg_1h = txn.amount

        feats = [
            txn.amount,
            float(len(recent_10m)),
            float(len(amounts_1h)),
            float(sum(amounts_1h)),
            float(avg_1h),
            float(max(amounts_1h)) if amounts_1h else 0.0,
            float(time_since_last),
            float(txn.amount / avg_1h) if avg_1h > 0 else 1.0,
            float(txn.timestamp.hour < 6 or txn.timestamp.hour >= 23),
            float(txn.merchant_category in RISKY_MERCHANTS),
        ]
        dq.append((txn.timestamp, txn.amount))
        assert len(feats) == len(FEATURE_NAMES)
        return feats
