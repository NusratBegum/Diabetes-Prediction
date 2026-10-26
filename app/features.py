"""Feature engineering shared by training and serving.

The model is a scikit-learn Pipeline, so everything here runs inside it and the
API only has to pass the eight raw measurements.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

RAW_FEATURES = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
]

# In the Pima dataset a 0 in these columns means "not measured", not a real value.
ZERO_MEANS_MISSING = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]

ENGINEERED_FEATURES = ["bmi_category", "age_group", "glucose_insulin_ratio"]


class ZeroToNaN(BaseEstimator, TransformerMixin):
    """Replace 0 with NaN in the columns where 0 marks a missing measurement."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        for name in ZERO_MEANS_MISSING:
            col = X[:, RAW_FEATURES.index(name)]
            col[col == 0] = np.nan
        return X


class AddEngineeredFeatures(BaseEstimator, TransformerMixin):
    """Append BMI category, age group and glucose/insulin ratio (after imputation)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        bmi = X[:, RAW_FEATURES.index("bmi")]
        age = X[:, RAW_FEATURES.index("age")]
        glucose = X[:, RAW_FEATURES.index("glucose")]
        insulin = X[:, RAW_FEATURES.index("insulin")]

        bmi_category = np.digitize(bmi, [18.5, 25, 30])  # under / normal / over / obese
        age_group = np.digitize(age, [30, 50])
        ratio = glucose / (insulin + 1)
        return np.column_stack([X, bmi_category, age_group, ratio])

    @staticmethod
    def get_feature_names() -> list[str]:
        return list(ENGINEERED_FEATURES)
