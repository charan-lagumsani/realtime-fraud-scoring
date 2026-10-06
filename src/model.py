"""Fraud classifier: training, persistence, and scoring."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from .features import OnlineFeatureStore, FEATURE_NAMES
from .stream import Transaction


def build_training_frame(transactions: list[Transaction]):
    """Replay transactions through the feature store to build X, y."""
    store = OnlineFeatureStore()
    X, y = [], []
    for txn in transactions:
        X.append(store.featurize(txn))
        y.append(1 if txn.is_fraud else 0)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def train_model(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """Train a gradient-boosting fraud classifier with a holdout report."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_depth=6,
        class_weight="balanced", random_state=seed,
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(clf, X_te, y_te, n_repeats=3,
                                  random_state=seed, scoring="f1")
    report = {
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
        "n_test": int(len(y_te)),
        "fraud_rate_test": float(y_te.mean()),
        "feature_importance": {
            name: float(imp)
            for name, imp in zip(FEATURE_NAMES, perm.importances_mean)
        },
    }
    return clf, report


def save_model(clf, path: str | Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(clf, f)


def load_model(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)
