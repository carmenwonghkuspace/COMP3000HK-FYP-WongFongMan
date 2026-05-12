#!/usr/bin/env python3
"""
Build web/model.json from result/best_model.pkl without re-running main.py.

The browser only understands the TF-IDF + logistic JSON pack (same schema as
main.py writes). Arbitrary sklearn/joblib objects cannot be dumped to JSON as-is.

- If the bundled model is LogisticRegression: only the .pkl is required.
- If it is RF / XGBoost / etc.: main.py fits a logistic surrogate on the training
  TF-IDF matrix; this script needs result/processed_tfidf.csv (from the same
  main.py run that produced the .pkl) to reproduce that surrogate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

KEEP_TOKENS = {"url", "num"}


def build_browser_pack(bundle: dict, processed_csv: Path | None) -> dict:
    best_model = bundle["model"]
    tfidf = bundle["vectorizer"]
    best_name = bundle.get("model_name", "unknown")
    feature_names = np.asarray(bundle["feature_names"])
    rng = int(bundle.get("random_state", 42))
    raw_sw = bundle.get("stop_words") or []
    stop_sorted = sorted(set(raw_sw))

    browser_surrogate_lr = not isinstance(best_model, LogisticRegression)
    if browser_surrogate_lr:
        if processed_csv is None or not processed_csv.is_file():
            raise FileNotFoundError(
                "The winning model is not LogisticRegression; the browser pack uses "
                "a logistic surrogate fit on training TF-IDF rows (same as main.py). "
                "Provide --processed-csv pointing to result/processed_tfidf.csv from "
                "the same pipeline run as this .pkl, or run main.py once to create it."
            )
        df = pd.read_csv(processed_csv)
        for col in ("dataset", "label"):
            if col not in df.columns:
                raise ValueError(f"{processed_csv} must contain a '{col}' column.")
        dtrain = df[df["dataset"] == "train"]
        fn_list = feature_names.tolist()
        missing = [c for c in fn_list if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV missing {len(missing)} feature columns (e.g. {missing[:3]}). "
                "Use processed_tfidf.csv from the same run as best_model.pkl."
            )
        x_train = dtrain[fn_list].to_numpy(dtype=np.float64)
        y_train = dtrain["label"].to_numpy()
        lr_browser = LogisticRegression(
            max_iter=2000, C=1.0, solver="liblinear", random_state=rng
        )
        lr_browser.fit(x_train, y_train)
    else:
        lr_browser = best_model

    idf_browser = getattr(tfidf, "idf_", None)
    if idf_browser is None:
        idf_browser = np.ones(len(feature_names))
    idf_browser = np.asarray(idf_browser).ravel()

    return {
        "version": 1,
        "best_model_cv": best_name,
        "inference_mode": (
            "surrogate_logistic" if browser_surrogate_lr else "best_logistic_regression"
        ),
        "note": (
            "Probabilities match the CV-winning model only when it is "
            "sklearn LogisticRegression; otherwise this uses an LR surrogate "
            "fit on identical TF-IDF features for consistent browser inference."
        ),
        "ngram_range": list(tfidf.ngram_range),
        "sublinear_tf": bool(tfidf.sublinear_tf),
        "smooth_idf": bool(getattr(tfidf, "smooth_idf", True)),
        "norm": (getattr(tfidf, "norm", None) or "l2"),
        "intercept": float(np.asarray(lr_browser.intercept_).ravel()[0]),
        "coef": np.asarray(lr_browser.coef_).ravel().astype(np.float64).tolist(),
        "feature_names": feature_names.tolist(),
        "idf": idf_browser.astype(np.float64).tolist(),
        "stop_words": stop_sorted,
        "keep_tokens": sorted(KEEP_TOKENS),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pkl", type=Path, default=Path("result/best_model.pkl"))
    ap.add_argument("--out-web", type=Path, default=Path("web/model.json"))
    ap.add_argument("--out-result", type=Path, default=Path("result/browser_model.json"))
    ap.add_argument(
        "--processed-csv",
        type=Path,
        default=Path("result/processed_tfidf.csv"),
        help="Training TF-IDF rows (needed only when the CV winner is not LogisticRegression).",
    )
    args = ap.parse_args()

    bundle = joblib.load(args.pkl)
    pack = build_browser_pack(bundle, args.processed_csv)

    args.out_web.parent.mkdir(parents=True, exist_ok=True)
    args.out_result.parent.mkdir(parents=True, exist_ok=True)
    for dest in (args.out_web, args.out_result):
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(pack, fh, separators=(",", ":"))
        print(f"Wrote {dest.resolve()}")


if __name__ == "__main__":
    main()
