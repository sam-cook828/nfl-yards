"""Expected-yards model: predicts a player's mean yardage for a game (squared-error loss).

Boosted trees shrink extreme players toward the middle (elite receivers come out too low) and can carry a
season-level bias (passing ran high). A linear calibration step corrects both: it is fit on out-of-sample
predictions from the five seasons before the test set, each made by a model trained only on earlier seasons,
and blends the tree prediction with the player's 8-game average.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

CALIBRATION_SEASONS = 5


def _make(n_iter: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=40,
        l2_regularization=1.0, max_iter=n_iter, random_state=0)


class CalibratedRegressor:
    """Tree model plus a linear correction: yards = a * tree prediction + b * 8-game average + c."""

    def __init__(self, trees: HistGradientBoostingRegressor, calibrator: LinearRegression):
        self.trees, self.calibrator = trees, calibrator

    def predict(self, X: pd.DataFrame, baseline: pd.Series) -> np.ndarray:
        raw = self.trees.predict(X)
        return self.calibrator.predict(np.c_[raw, baseline.to_numpy()])


def _calibrator(trva: pd.DataFrame, feats: list[str], target: str, n_iter: int) -> LinearRegression:
    seasons = sorted(trva["season"].unique())[-CALIBRATION_SEASONS:]
    oof = []
    for s in seasons:
        past, cur = trva[trva.season < s], trva[trva.season == s]
        oof.append(pd.DataFrame({"pred": _make(n_iter).fit(past[feats], past[target]).predict(cur[feats]),
                                 "baseline": cur["baseline"].to_numpy(), "y": cur[target].to_numpy()}))
    oof = pd.concat(oof)
    return LinearRegression().fit(oof[["pred", "baseline"]].to_numpy(), oof["y"])


def _scores(pred, te, target) -> dict:
    y, base = te[target].to_numpy(), te["baseline"].to_numpy()
    top = base >= np.percentile(base, 98)  # the very best players by recent form
    return dict(test_rmse=round(float(np.sqrt(np.mean((pred - y) ** 2))), 2),
                test_mae=round(float(np.abs(pred - y).mean()), 2),
                mean_bias=round(float((pred - y).mean()), 2),
                top2pct_bias=round(float((pred[top] - y[top]).mean()), 2))


def fit(tr, va, te, feats, target):
    """Pick boosting rounds on `va`, fit the calibration, refit on tr+va, score on `te` against the baseline."""
    probe = _make(1500).fit(tr[feats], tr[target])
    rmse = [np.sqrt(np.mean((p - va[target]) ** 2)) for p in probe.staged_predict(va[feats])]
    best_n = int(np.argmin(rmse)) + 1

    trva = pd.concat([tr, va])
    trees = _make(best_n).fit(trva[feats], trva[target])
    model = CalibratedRegressor(trees, _calibrator(trva, feats, target, best_n))

    y, base = te[target], te["baseline"]
    metrics = dict(n_train=len(tr), n_test=len(te), best_iter=best_n,
                   calibration=dict(tree_weight=round(float(model.calibrator.coef_[0]), 3),
                                    avg_weight=round(float(model.calibrator.coef_[1]), 3),
                                    intercept=round(float(model.calibrator.intercept_), 2)),
                   baseline_rmse=round(float(np.sqrt(np.mean((base - y) ** 2))), 2),
                   baseline_mae=round(float(np.abs(base - y).mean()), 2),
                   uncalibrated=_scores(trees.predict(te[feats]), te, target),
                   **_scores(model.predict(te[feats], base), te, target))
    metrics["rmse_improvement_pct"] = round(100 * (1 - metrics["test_rmse"] / metrics["baseline_rmse"]), 1)
    return model, metrics


def predict(model: CalibratedRegressor, df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return np.clip(model.predict(df.reindex(columns=feats), df["baseline"]), 0, None)
