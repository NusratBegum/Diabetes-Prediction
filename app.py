import gradio as gr
import joblib
import numpy as np

model = joblib.load('diabetes_model.pkl')

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, 
                     insulin, bmi, diabetes_pedigree, age):

    if bmi < 18.5:
        bmi_category = 0
    elif bmi < 25:
        bmi_category = 1
    elif bmi < 30:
        bmi_category = 2
    else:
        bmi_category = 3

    if age < 30:
        age_group = 0
    elif age < 50:
        age_group = 1
    else:
        age_group = 2

    glucose_insulin_ratio = glucose / (insulin + 1)

    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                          insulin, bmi, diabetes_pedigree, age, 
                          bmi_category, age_group, glucose_insulin_ratio]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    if prediction == 1:
        result = "HIGH RISK - Diabetes Predicted"
        confidence = probability[1] * 100
    else:
        result = "LOW RISK - No Diabetes"
        confidence = probability[0] * 100

    return result + "\nConfidence: " + str(round(confidence, 2)) + "%"

demo = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Slider(0, 17, step=1, label="Pregnancies"),
        gr.Slider(0, 200, step=1, value=120, label="Glucose"),
        gr.Slider(0, 130, step=1, value=70, label="Blood Pressure"),
        gr.Slider(0, 100, step=1, value=20, label="Skin Thickness"),
        gr.Slider(0, 900, step=1, value=80, label="Insulin"),
        gr.Slider(0, 70, step=0.1, value=25.0, label="BMI"),
        gr.Slider(0, 2.5, step=0.01, value=0.5, label="Diabetes Pedigree Function"),
        gr.Slider(21, 81, step=1, value=30, label="Age")
    ],
    outputs=gr.Textbox(label="Result", lines=6, max_lines=6),
    title="Diabetes Prediction",
    description="Enter your health data to predict diabetes risk."
)

demo.launch()
