import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.features import RAW_FEATURES
from app.schemas import PatientFeatures, Prediction

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "model.joblib"
METRICS_PATH = ROOT / "model" / "metrics.json"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Diabetes Risk API",
    description=(
        "Random-forest classifier trained on the Pima Indians Diabetes dataset, served with "
        "FastAPI. For learning and demonstration only, not medical advice."
    ),
    version="2.0.0",
)


@lru_cache
def get_model():
    return joblib.load(MODEL_PATH)


@lru_cache
def get_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text())


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


@app.get("/model", tags=["model"])
def model_info():
    """Held-out test metrics, tuned hyperparameters and feature importance."""
    return get_metrics()


@app.post("/predict", response_model=Prediction, tags=["model"])
def predict(features: PatientFeatures):
    row = np.array([[getattr(features, name) for name in RAW_FEATURES]], dtype=float)
    probability = float(get_model().predict_proba(row)[0, 1])
    return Prediction(
        risk="high" if probability >= 0.5 else "low",
        probability=probability,
        model_version=get_metrics()["trained_at"],
    )
