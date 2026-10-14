"""Expected-yards model: predicts a player's mean yardage for a game (squared-error loss)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def _make(n_iter: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=40,
        l2_regularization=1.0, max_iter=n_iter, random_state=0)


def fit(tr, va, te, feats, target):
    """Pick boosting rounds on `va`, refit on tr+va, score on `te` against the 8-game-average baseline."""
    probe = _make(1500).fit(tr[feats], tr[target])
    rmse = [np.sqrt(np.mean((p - va[target]) ** 2)) for p in probe.staged_predict(va[feats])]
    best_n = int(np.argmin(rmse)) + 1

    trva = pd.concat([tr, va])
    model = _make(best_n).fit(trva[feats], trva[target])
    pred, y = model.predict(te[feats]), te[target]
    base = te["baseline"]
    metrics = dict(n_train=len(tr), n_test=len(te), best_iter=best_n,
                   test_rmse=round(float(np.sqrt(np.mean((pred - y) ** 2))), 2),
                   baseline_rmse=round(float(np.sqrt(np.mean((base - y) ** 2))), 2),
                   test_mae=round(float(np.abs(pred - y).mean()), 2),
                   baseline_mae=round(float(np.abs(base - y).mean()), 2),
                   mean_bias=round(float((pred - y).mean()), 2))
    metrics["rmse_improvement_pct"] = round(100 * (1 - metrics["test_rmse"] / metrics["baseline_rmse"]), 1)
    return model, metrics


def predict(model, df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return np.clip(model.predict(df.reindex(columns=feats)), 0, None)
