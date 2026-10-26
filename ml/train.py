"""Train the diabetes risk model and write model/model.joblib + model/metrics.json.

Usage:
    python -m ml.train
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from app.features import RAW_FEATURES, AddEngineeredFeatures, ZeroToNaN

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "diabetes.csv"
MODEL_DIR = ROOT / "model"
SEED = 42

COLUMNS = {
    "Pregnancies": "pregnancies",
    "Glucose": "glucose",
    "BloodPressure": "blood_pressure",
    "SkinThickness": "skin_thickness",
    "Insulin": "insulin",
    "BMI": "bmi",
    "DiabetesPedigreeFunction": "diabetes_pedigree",
    "Age": "age",
}


def build_pipeline() -> Pipeline:
    # Imputation happens inside the pipeline, so medians come from training folds only.
    return Pipeline(
        [
            ("zero_to_nan", ZeroToNaN()),
            ("impute", SimpleImputer(strategy="median")),
            ("features", AddEngineeredFeatures()),
            ("model", RandomForestClassifier(random_state=SEED, class_weight="balanced")),
        ]
    )


def main() -> None:
    df = pd.read_csv(DATA).rename(columns=COLUMNS)
    X = df[RAW_FEATURES].to_numpy(dtype=float)
    y = df["Outcome"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    search = GridSearchCV(
        build_pipeline(),
        param_grid={
            "model__n_estimators": [100, 200],
            "model__max_depth": [4, 6, 8, None],
            "model__min_samples_leaf": [1, 3, 5],
        },
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "sklearn_version": sklearn.__version__,
        "rows": {"train": int(len(y_train)), "test": int(len(y_test))},
        "positive_rate": round(float(y.mean()), 3),
        "best_params": search.best_params_,
        "cv_roc_auc": round(float(search.best_score_), 3),
        "test": {
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 3),
            "accuracy": round(float(accuracy_score(y_test, pred)), 3),
            "precision": round(float(precision_score(y_test, pred)), 3),
            "recall": round(float(recall_score(y_test, pred)), 3),
            "f1": round(float(f1_score(y_test, pred)), 3),
        },
    }

    # Refit on all rows for the served model; the metrics above are from the held-out split.
    final = build_pipeline().set_params(**search.best_params_).fit(X, y)

    importances = final.named_steps["model"].feature_importances_
    names = RAW_FEATURES + AddEngineeredFeatures().get_feature_names()
    metrics["feature_importance"] = {
        n: round(float(v), 3) for n, v in sorted(zip(names, importances), key=lambda p: -p[1])
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_DIR / "model.joblib", compress=3)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics["test"], indent=2))


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
