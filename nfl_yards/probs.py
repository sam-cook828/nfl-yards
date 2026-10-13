"""Probability of reaching yardage thresholds, e.g. P(receiving yards >= 60).

One classifier per yard type is trained on rows stacked across thresholds (a `threshold` feature, label
yards >= threshold). The threshold features carry a decreasing monotonic constraint, so P(>=50) can never
fall below P(>=60).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss

THRESHOLDS = {
    "passing_yards": [125, 150, 175, 200, 225, 250, 275, 300, 325, 350],
    "rushing_yards": [25, 40, 50, 60, 70, 80, 90, 100],
    "receiving_yards": [25, 40, 50, 60, 70, 80, 90, 100],
}


def stack(df: pd.DataFrame, feats: list[str], target: str, thresholds: list[int]) -> pd.DataFrame:
    """One copy of the feature rows per threshold, plus the threshold and its gap to recent form."""
    parts = []
    for t in thresholds:
        x = df[feats].copy()
        x["threshold"] = t
        x["thr_minus_r8"] = t - df[f"{target}_r8"]
        parts.append(x)
    return pd.concat(parts, ignore_index=True)


def labels(df: pd.DataFrame, target: str, thresholds: list[int]) -> np.ndarray:
    return np.concatenate([(df[target] >= t).to_numpy() for t in thresholds]).astype(int)


def make_model(columns: list[str], n_iter: int) -> HistGradientBoostingClassifier:
    mono = {c: -1 for c in ("threshold", "thr_minus_r8")}
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0,
        max_iter=n_iter, monotonic_cst=[mono.get(c, 0) for c in columns], random_state=0)


def fit(tr, va, te, feats, target):
    """Pick boosting rounds on `va`, refit on tr+va, score on `te` against base-rate-by-threshold."""
    th = THRESHOLDS[target]
    Xtr, ytr = stack(tr, feats, target, th), labels(tr, target, th)
    Xva, yva = stack(va, feats, target, th), labels(va, target, th)
    probe = make_model(list(Xtr.columns), 400).fit(Xtr, ytr)
    losses = [log_loss(yva, p[:, 1]) for p in probe.staged_predict_proba(Xva)]
    best_n = int(np.argmin(losses)) + 1

    trva = pd.concat([tr, va])
    final = make_model(list(Xtr.columns), best_n).fit(stack(trva, feats, target, th), labels(trva, target, th))
    Xte, yte = stack(te, feats, target, th), labels(te, target, th)
    p = final.predict_proba(Xte)[:, 1]
    base = np.concatenate([np.full(len(te), (trva[target] >= t).mean()) for t in th])
    metrics = dict(best_iter=best_n, log_loss=round(log_loss(yte, p), 4), base_rate_log_loss=round(log_loss(yte, base), 4),
                   brier=round(brier_score_loss(yte, p), 4), base_rate_brier=round(brier_score_loss(yte, base), 4))
    # calibration: mean predicted vs observed hit rate in probability deciles
    cal = pd.DataFrame({"p": p, "y": yte})
    cal["bin"] = pd.cut(cal["p"], [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    metrics["calibration"] = {str(k): [round(v.p.mean(), 3), round(v.y.mean(), 3), len(v)]
                              for k, v in cal.groupby("bin", observed=True) if len(v)}
    return final, metrics


def predict(model, df: pd.DataFrame, feats: list[str], target: str) -> pd.DataFrame:
    th = THRESHOLDS[target]
    p = model.predict_proba(stack(df, feats, target, th)[list(model.feature_names_in_)])[:, 1]
    return pd.DataFrame(p.reshape(len(th), len(df)).T, index=df.index, columns=[f"p_{t}" for t in th])
