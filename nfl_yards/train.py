"""Train one LightGBM model per yard type; compare against a rolling-average baseline."""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingRegressor

from .data import load_all
from .features import TARGETS, build, feature_cols

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-season", type=int, default=2024, help="early-stopping season; later seasons are the test set")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    raw = load_all(refresh=args.refresh)
    print(f"loaded {len(raw):,} player-weeks, seasons {raw.season.min()}-{raw.season.max()}")
    MODELS_DIR.mkdir(exist_ok=True)
    report = {}

    for target in TARGETS:
        df = build(raw, target)
        feats = feature_cols(df, target)
        tr, va, te = (df[df.season < args.val_season], df[df.season == args.val_season], df[df.season > args.val_season])
        # absolute_error loss targets MAE; early stopping uses a time-ordered tail of train, then val season is a check
        model = HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=40,
            l2_regularization=1.0, max_iter=1500, early_stopping=False, random_state=0)
        best_n, best_mae = 50, np.inf
        model.set_params(max_iter=1)
        model.fit(tr[feats], tr[target])
        for n in range(50, 1501, 50):
            model.set_params(max_iter=n, warm_start=False).fit(tr[feats], tr[target])
            m = np.abs(model.predict(va[feats]) - va[target]).mean()
            if m < best_mae:
                best_n, best_mae = n, m
            elif n - best_n >= 200:
                break
        model.set_params(max_iter=best_n).fit(pd.concat([tr, va])[feats], pd.concat([tr, va])[target])
        pred = model.predict(te[feats])
        mae, base_mae = np.abs(pred - te[target]).mean(), np.abs(te["baseline"] - te[target]).mean()
        report[target] = dict(n_train=len(tr), n_test=len(te), best_iter=best_n,
                              test_mae=round(mae, 2), baseline_mae=round(base_mae, 2),
                              improvement_pct=round(100 * (1 - mae / base_mae), 1))
        print(target, report[target])
        joblib.dump({"model": model, "features": feats}, MODELS_DIR / f"{target}.joblib")

    (MODELS_DIR / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
