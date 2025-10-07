from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import joblib
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import io
import google.generativeai as genai

# ————— Gemini API key (hardcoded, for testing) —————
GEMINI_API_KEY = "AIzaSyDqomTOI7QIcr-bfGP9dLclAW_03n85JuY"
genai.configure(api_key=GEMINI_API_KEY)

# ————— FastAPI setup —————
app = FastAPI(title="DeepReef + Gemini 2.5")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ————— Load models & metadata —————
cnn_model = tf.keras.models.load_model("coral_mobilenet_best.h5")
rf = joblib.load("coral_bleaching_model.pkl")
label_encoder: LabelEncoder = joblib.load("label_encoder.pkl")
X_train_columns = joblib.load("train_columns.pkl")

def preprocess_image(uploaded_file: UploadFile):
    TARGET_SIZE = (160, 160)
    img_bytes = uploaded_file.file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(TARGET_SIZE)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return arr

@app.post("/analyze_coral")
async def analyze_coral(
    file: UploadFile = File(...),
    Temperature_Mean: float = Form(...),
    Windspeed: float = Form(...),
    TSA: float = Form(...),
    Ocean_Name: str = Form(...),
    Exposure: str = Form(...)
):
    try:
        # CNN inference
        arr = preprocess_image(file)
        preds = cnn_model.predict(arr)[0]
        class_labels = ['Bleached Coral', 'Healthy Coral']
        idx = int(np.argmax(preds))
        coral_status = class_labels[idx]
        confidence = float(preds[idx])

        # RF inference
        df = pd.DataFrame([{
            'Temperature_Mean': Temperature_Mean,
            'Windspeed': Windspeed,
            'TSA': TSA,
            'Ocean_Name': Ocean_Name,
            'Exposure': Exposure
        }])
        df_enc = pd.get_dummies(df, columns=['Ocean_Name', 'Exposure'], drop_first=True)
        for col in X_train_columns:
            if col not in df_enc.columns:
                df_enc[col] = 0
        df_enc = df_enc[X_train_columns]
        severity_pred = rf.predict(df_enc)
        severity = label_encoder.inverse_transform(severity_pred)[0]

        # Compose Gemini prompt
        prompt = f"""
        Coral image result: {coral_status} (confidence: {confidence*100:.2f}%).
        Environmental parameters:
         • Temperature_Mean = {Temperature_Mean}
         • Windspeed = {Windspeed}
         • TSA = {TSA}
         • Ocean_Name = {Ocean_Name}, Exposure = {Exposure}
        RF bleaching severity: {severity}.

        Provide a concise scientific summary (3-4 sentences) interpreting these findings,
        possible explanations, and risk assessment.
        """

        # Call Gemini 2.5 Flash
        model = genai.GenerativeModel("gemini-2.5-flash")
        gemini_resp = model.generate_content(prompt)
        summary = gemini_resp.text

        return {
            "image_prediction": coral_status,
            "confidence": round(confidence * 100, 2),
            "bleaching_severity": severity,
            "gemini_summary": summary
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "DeepReef + Gemini 2.5 ready", "endpoint": "/analyze_coral"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
