import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.features import RAW_FEATURES, AddEngineeredFeatures, ZeroToNaN
from app.main import app

client = TestClient(app)

EXAMPLE = {
    "pregnancies": 2,
    "glucose": 138,
    "blood_pressure": 72,
    "skin_thickness": 30,
    "insulin": 120,
    "bmi": 33.1,
    "diabetes_pedigree": 0.45,
    "age": 41,
}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_index_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_predict_returns_probability():
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["probability"] <= 1
    assert body["risk"] == ("high" if body["probability"] >= 0.5 else "low")


def test_higher_glucose_raises_risk():
    low = client.post("/predict", json={**EXAMPLE, "glucose": 85, "bmi": 22.0}).json()["probability"]
    high = client.post("/predict", json={**EXAMPLE, "glucose": 190, "bmi": 40.0}).json()["probability"]
    assert high > low


@pytest.mark.parametrize(
    "field,value", [("glucose", -1), ("age", 5), ("bmi", 120), ("pregnancies", 1.5)]
)
def test_out_of_range_input_is_rejected(field, value):
    assert client.post("/predict", json={**EXAMPLE, field: value}).status_code == 422


def test_missing_field_is_rejected():
    body = {k: v for k, v in EXAMPLE.items() if k != "insulin"}
    assert client.post("/predict", json=body).status_code == 422


def test_model_card_has_test_metrics():
    card = client.get("/model").json()
    assert set(card["test"]) == {"roc_auc", "accuracy", "precision", "recall", "f1"}
    assert card["rows"]["train"] + card["rows"]["test"] == 768


def test_zero_marks_missing_values():
    X = np.array([[1, 0, 0, 0, 0, 0, 0.5, 30]], dtype=float)
    out = ZeroToNaN().transform(X)
    assert out[0, RAW_FEATURES.index("pregnancies")] == 1  # 0 pregnancies is a real value
    assert np.isnan(out[0, RAW_FEATURES.index("glucose")])
    assert np.isnan(out[0, RAW_FEATURES.index("bmi")])


def test_engineered_features():
    X = np.array([[0, 100, 70, 20, 99, 31.0, 0.5, 55]], dtype=float)
    out = AddEngineeredFeatures().transform(X)
    bmi_category, age_group, ratio = out[0, -3:]
    assert bmi_category == 3  # obese
    assert age_group == 2  # 50+
    assert ratio == pytest.approx(1.0)
