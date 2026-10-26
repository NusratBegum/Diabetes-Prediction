from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    """The eight measurements from the Pima Indians Diabetes dataset.

    0 is accepted for glucose, blood pressure, skin thickness, insulin and BMI and
    is treated as "not measured" (imputed by the model), as in the training data.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pregnancies": 2,
                "glucose": 138,
                "blood_pressure": 72,
                "skin_thickness": 30,
                "insulin": 120,
                "bmi": 33.1,
                "diabetes_pedigree": 0.45,
                "age": 41,
            }
        }
    )

    pregnancies: int = Field(ge=0, le=20)
    glucose: float = Field(ge=0, le=250, description="Plasma glucose (mg/dL), 2-hour OGTT")
    blood_pressure: float = Field(ge=0, le=160, description="Diastolic blood pressure (mm Hg)")
    skin_thickness: float = Field(ge=0, le=100, description="Triceps skinfold (mm)")
    insulin: float = Field(ge=0, le=900, description="2-hour serum insulin (µU/mL)")
    bmi: float = Field(ge=0, le=70)
    diabetes_pedigree: float = Field(ge=0, le=3, description="Diabetes pedigree function")
    age: int = Field(ge=21, le=100, description="Training data covers ages 21-81")


class Prediction(BaseModel):
    risk: str = Field(description='"high" or "low" at the 0.5 threshold')
    probability: float = Field(ge=0, le=1, description="Predicted probability of diabetes")
    model_version: str
