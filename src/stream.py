"""Simulated real-time transaction stream.

Generates a chronological stream of card transactions with ground-truth fraud
labels. Legitimate traffic follows per-card spending profiles; fraud appears as
bursts of high-velocity, high-amount transactions — the pattern real fraud
rings exhibit when testing stolen cards.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class Transaction:
    txn_id: str
    card_id: str
    amount: float
    merchant_category: str
    timestamp: datetime
    is_fraud: bool = False  # ground truth — never visible to the scorer


MERCHANT_CATEGORIES = [
    "grocery", "fuel", "restaurant", "travel", "retail",
    "online_marketplace", "entertainment", "pharmacy",
]

RISKY_MERCHANTS = {"online_marketplace", "travel", "entertainment"}


class TransactionStream:
    """Yields transactions in chronological order, like a Kafka topic would."""

    def __init__(self, n_cards: int = 400, fraud_rate: float = 0.02, seed: int = 42):
        self.rng = random.Random(seed)
        self.n_cards = n_cards
        self.fraud_rate = fraud_rate
        # Per-card spending profiles for legitimate behavior.
        self.profiles = {
            f"card_{i:04d}": {
                "mean_amount": self.rng.uniform(15, 220),
                "categories": self.rng.sample(MERCHANT_CATEGORIES, k=3),
            }
            for i in range(n_cards)
        }

    def _legit_txn(self, txn_id: int, card_id: str, ts: datetime) -> Transaction:
        p = self.profiles[card_id]
        # Occasional large legitimate purchase (travel/retail) so amount
        # alone cannot separate fraud — velocity features must do the work.
        if self.rng.random() < 0.02:
            amount = self.rng.uniform(500, 1500)
            category = self.rng.choice(["travel", "retail"])
        else:
            amount = max(1.0, self.rng.gauss(p["mean_amount"],
                                             p["mean_amount"] * 0.4))
            category = self.rng.choice(p["categories"])
        return Transaction(
            txn_id=f"txn_{txn_id:07d}",
            card_id=card_id,
            amount=round(amount, 2),
            merchant_category=category,
            timestamp=ts,
            is_fraud=False,
        )

    def _fraud_burst(self, start_id: int, card_id: str, ts: datetime, n: int):
        """A fraud burst: several rapid high-amount txns on one card."""
        txns = []
        for j in range(n):
            txns.append(Transaction(
                txn_id=f"txn_{start_id + j:07d}",
                card_id=card_id,
                # Overlaps legitimate spend: amount alone is not enough.
                amount=round(self.rng.uniform(60, 900), 2),
                merchant_category=self.rng.choice(sorted(RISKY_MERCHANTS)),
                timestamp=ts + timedelta(seconds=self.rng.randint(20, 180)),
                is_fraud=True,
            ))
        return txns

    def generate(self, n_transactions: int, start: datetime | None = None):
        """Generate a list of `n_transactions` chronologically ordered txns."""
        start = start or datetime(2026, 1, 5, 0, 0, 0)
        txns: list[Transaction] = []
        txn_id = 0
        ts = start
        cards = list(self.profiles)
        while len(txns) < n_transactions:
            ts += timedelta(seconds=self.rng.randint(5, 90))
            card_id = self.rng.choice(cards)
            if self.rng.random() < self.fraud_rate:
                burst = self._fraud_burst(txn_id, card_id, ts,
                                          n=self.rng.randint(2, 5))
                txns.extend(burst)
                txn_id += len(burst)
                ts = burst[-1].timestamp
            else:
                txns.append(self._legit_txn(txn_id, card_id, ts))
                txn_id += 1
        txns.sort(key=lambda t: t.timestamp)
        return txns[:n_transactions]
